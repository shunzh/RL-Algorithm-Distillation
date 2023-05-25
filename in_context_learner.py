import math
import os
import pprint
import numpy as np

import torch
import wandb
from transformers import TrainingArguments, Trainer
import gym

import envs
from lifetime_dataset import LifetimeDataset, collate_fn, prepare_for_prediction, collect_from_pkl
from model import PolicyTransformer, PolicyTransformerConfig

import warnings

from utils import find_config_file

warnings.filterwarnings("ignore", category=DeprecationWarning, module='gym')


def train_model(
        env_id: str,
        model_config: str,
        subsample_gap: int,
        file_prefix: str,
        output_dir: str,
        resume: bool,
        device: str,
) -> None:
    """
    Args:
        env_id: gym environment id
        model_config: path to model config, that configus the Transformer model
        file_prefix: prefix of the pickle files that contain the lifetimes
        device: device to run the model on
    """
    # add reward threshold to exclude failed lifetimes
    lifetimes = collect_from_pkl(file_prefix, subsample_gap=subsample_gap)

    config = PolicyTransformerConfig.from_json_file(model_config)
    model = PolicyTransformer(config)

    model.to(device)

    split_idx = math.ceil(len(lifetimes) * 0.95)
    train_dataset = LifetimeDataset(lifetimes[:split_idx], context_len=config.context_len)
    if split_idx == len(lifetimes):
        warnings.warn("Not enough data to split into train and eval sets, using train set for eval")
        eval_dataset = train_dataset
    else:
        eval_dataset = LifetimeDataset(lifetimes[split_idx:], context_len=config.context_len)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        per_device_train_batch_size=1024,
        per_device_eval_batch_size=1024,
        warmup_ratio=0.1,
        learning_rate=3e-4, # same as the peak learning rate in the paper
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        max_grad_norm=1,
    )

    wandb.init(
        project='alg-distill',
        name=env_id + '-train',
        config=config.to_dict(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )

    trainer.train(
        resume_from_checkpoint=resume, # will use the latest checkpoint in output_dir if True
    )

    trainer.save_model()


@torch.no_grad()
def eval_model(
        env_id: str,
        lifetime_num: int,
        episode_num: int,
        model_config: str,
        model_path: str,
        device: str,
        log_interval: int = 100,
        temperature: float = 1.,
        baseline: str = None,
) -> None:
    config = PolicyTransformerConfig.from_json_file(model_config)
    model = PolicyTransformer(config)
    model.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin')))
    model.eval()
    model.to(device)

    context_len = model.config.context_len

    states = []
    actions = []
    rewards = []
    timesteps = []

    wandb.init(
        project='alg-distill',
        name=env_id + '-eval',
        config=config.to_dict(),
    )

    for lifetime_idx in range(lifetime_num):
        env = gym.make(env_id)
        ep_len = env.episode_length

        for i in range(episode_num):
            traj = []
            obs = env.reset()
            for t in range(ep_len):
                states += [obs]
                timesteps += [t]
                if baseline == 'random':
                    action = env.action_space.sample()
                else:
                    model_input = prepare_for_prediction(states, actions, rewards, timesteps,
                                                        env, context_len, device)
                    action = model.predict(**model_input, temperature=temperature) # temperature > 0 to allow exploration
                new_obs, reward, done, _ = env.step(action)
                actions += [action]
                rewards += [reward]
                traj += [(obs, action, reward, t)]
                obs = new_obs

                if done:
                    break

            mean_reward = np.mean(rewards[-ep_len:])
            print(f"Lifetime {lifetime_idx}, episode {i}, mean reward {mean_reward}")
            if (i+1) % log_interval == 0:
                log_interval_mean_reward = np.mean(rewards[-log_interval:])
                wandb.log({f'mean_reward': log_interval_mean_reward})
                env.render(trajectory=traj,
                           log_name=f'l{lifetime_idx}/e{i:05}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='DarkRoom-v0')
    parser.add_argument('--subsample_gap', type=int, default=1,
                        help='subsample steps with indices of 0, i, 2*i, ..., where i is the subsample_gap')
    parser.add_argument('--file_prefix', type=str, default='darkroom_normal')
    parser.add_argument('--model_path', type=str, default='output')
    parser.add_argument('--resume', action='store_true',
                        help='resume training from the latest checkpoint in model_path')

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--lifetime_num', type=int, default=20)
    parser.add_argument('--episode_num', type=int, default=2000)
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--baseline', type=str, default=None)

    args = parser.parse_args()

    model_config = find_config_file(env_id=args.env_id, alg='dt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pprint.pprint(vars(args))
    print(f'Using device: {device}')
    print(f'Using model config: {model_config}')

    if not args.eval:
        train_model(
            env_id=args.env_id,
            model_config=model_config,
            subsample_gap=args.subsample_gap,
            file_prefix=args.file_prefix,
            output_dir=args.model_path,
            resume=args.resume,
            device=device,
        )
    else:
        eval_model(
            env_id=args.env_id,
            lifetime_num=args.lifetime_num,
            episode_num=args.episode_num,
            model_config=model_config,
            model_path=args.model_path,
            temperature=args.temperature,
            baseline=args.baseline,
            device=device,
        )
