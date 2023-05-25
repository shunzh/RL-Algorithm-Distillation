import json
import pickle
import pprint

import gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

import wandb

import envs
from stable_baselines3 import A2C

from utils import find_config_file


class SaveTrajCallBack(BaseCallback):
    def __init__(self, verbose=0, env=None, lifetime_idx=None, log=False, log_interval=1000):
        super().__init__(verbose)
        self.env = env
        self.lifetime_idx = lifetime_idx
        self.log = log
        self.log_interval = log_interval

        self.trajectories = []

    def _on_step(self) -> bool:
        episode_length = self.env.episode_length
        learning_step = self.num_timesteps - 1 # starts from 0

        state = self.locals.get('obs_tensor')[0].tolist()
        action = self.locals.get('actions')[0].tolist()
        reward = self.locals.get('rewards')[0].tolist()
        # fixme cannot deal with episodes with different lengths for now
        timestep = learning_step % episode_length

        self.trajectories.append((state, action, reward, timestep))

        log_interval = self.log_interval
        if (learning_step + 1) % log_interval == 0:
            mean_reward = np.mean([t[2] for t in self.trajectories[-log_interval:]])
            print(f'Time step: {self.num_timesteps}, Mean reward: {mean_reward}')

            if self.log:
                wandb.log({'mean_reward': mean_reward})
                # render trajectory and log on wandb
                # disabling for now, using too much bandwidth
                # self.env.render(trajectory=self.trajectories[-episode_length:],
                #                 log_name=f'l{self.lifetime_idx}/t{learning_step:05}')

        return True


def collect_lifetimes(
        env_id: str,
        lifetime_num: int = 10,
        lifetime_start_idx: int = 0,
        total_steps: int = 10000,
        output_prefix: str = None,
        alg_config: str = None,
        log: bool = False,
) -> None:
    """
    Collect trajectories from an environment using an RL algorithm.

    Args:
        env_id: The ID of the Gym environment to use.
        lifetime_num: The number of lifetimes to collect.
        total_steps: The total number of training steps.

    Returns:
        A list of lifetimes, where each lifetime is a concatenated list of trajectories:
        [(o0^0, a0^0, r0^0, t0^0), (o1^0, a1^0, r1^0, t1^0), ..., , (oT^0, aT^0, rT^0, tT^0),
         (o0^1, a0^1, r0^1, t0^1), (o1^1, a1^1, r1^1, t1^1), ..., (oT^1, aT^1, rT^1, t1^1),
         ...]
        where _t denotes timestep, ^i denotes the index of trajectory.
    """
    agent_config = json.load(open(alg_config, 'r'))

    if log:
        wandb.init(
            project='alg-distill',
            name=env_id + '-collect',
            monitor_gym=True,
            config={
                'env_id': env_id,
                'lifetime_num': lifetime_num,
                'total_steps': total_steps,
                **agent_config,
            }
        )

    for lifetime_idx in range(lifetime_start_idx, lifetime_num):
        print(f'Running lifetime {lifetime_idx}...')

        # Create the Gym environment
        env = gym.make(env_id)

        # Create the A2C agent and train it on the environment
        alg = A2C(env=env, **agent_config)
        callback = SaveTrajCallBack(env=env, lifetime_idx=lifetime_idx, log=log)

        alg.learn(total_timesteps=total_steps, callback=callback)

        if output_prefix:
            pickle.dump(callback.trajectories, open(f"{output_prefix}_{lifetime_idx}.pkl", 'wb'))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='DarkRoom-v0')
    parser.add_argument('--lifetime_num', type=int, default=1000)
    parser.add_argument('--lifetime_start_idx', type=int, default=0)
    parser.add_argument('--total_steps', type=int, default=20000)
    parser.add_argument('--output_prefix', type=str, default='darkroom_normal')
    parser.add_argument('--log', action='store_true')
    args = parser.parse_args()

    alg_config = find_config_file(args.env_id, 'a2c')

    pprint.pprint(vars(args))
    print(f'alg_config: {alg_config}')

    collect_lifetimes(
        env_id=args.env_id,
        lifetime_num=args.lifetime_num,
        lifetime_start_idx=args.lifetime_start_idx,
        total_steps=args.total_steps,
        output_prefix=args.output_prefix,
        alg_config=alg_config,
        log=args.log,
    )
