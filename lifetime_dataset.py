import os
import pickle

import gym
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm


def collect_from_pkl(file_prefix: str, subsample_gap: int = 1):
    """
    Read all pkl files with prefix file_prefix and concatenate them
    """
    filenames = []
    lifetimes = []
    # find all files with prefix file_prefix, end with pkl
    for file in os.listdir('.'):
        if file.startswith(file_prefix) and file.endswith('.pkl'):
            filenames.append(file)

    print(f'Found {len(filenames)} lifetime files, collecting...')
    for filename in tqdm(filenames):
        try:
            with open(filename, 'rb') as f:
                lifetime = pickle.load(f)
                lifetime = lifetime[::subsample_gap]
                lifetimes.append(lifetime)
        except EOFError:
            print(f'Failed to load {filename}, skipped.')

    print(f'{len(lifetimes)} lifetimes collected.')

    return lifetimes


class LifetimeDataset(Dataset):
    def __init__(self, lifetimes: list, context_len: int = 10):
        self.lifetimes = lifetimes
        self.context_len = context_len

        # excluding empty trajectory at the beginning of each lifetime,
        # nothing to predict without the first state in a lifetime
        self.lifetime_lens = [len(trajs) - 1 for trajs in self.lifetimes]
        self.total_len = sum(self.lifetime_lens)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        """
        Get the context up to index idx
        """
        # find the lifetime index and step index
        for lifetime_idx in range(len(self.lifetimes)):
            if idx < self.lifetime_lens[lifetime_idx]:
                break
            idx -= self.lifetime_lens[lifetime_idx]
        # excluding empty trajectory at the beginning of each lifetime
        # that is, never use lifetime[:0] as context, nothing to predict in this case
        step_idx = idx + 1

        # get the context_len previous steps as context in the same lifetime
        context = self.lifetimes[lifetime_idx][max(0, step_idx - self.context_len):step_idx]

        return {
            'states': [s[0] for s in context],
            'actions': [s[1] for s in context],
            'rewards': [s[2] for s in context],
            'timesteps': [s[3] for s in context],
        }


def collate_fn(batch):
    """
    If shorter than max length, padding on the left and mask.
    This only applies to the beginning of each lifetime.
    """
    states = [torch.FloatTensor(s['states']) for s in batch]
    actions = [torch.FloatTensor(s['actions']) for s in batch]
    rewards = [torch.FloatTensor(s['rewards']) for s in batch]
    timesteps = [torch.tensor(s['timesteps']) for s in batch]

    max_length = max(len(s) for s in states)

    attention_mask = [torch.tensor([0] * (max_length - len(s['states']))
                                   + [1] * len(s['states']))
                      for s in batch]

    for idx in range(len(batch)):
        if len(states[idx]) < max_length:
            # if this datum is shorter than the max length in batch, padding on the left
            # need to pad in the first dimension, so that the seq_len is consistent
            # but F.pad pads starting from the last dimension, so a bit twisted here
            pad = (max_length - len(states[idx]), 0)

            states[idx] = F.pad(states[idx], (states[idx].dim() - 1) * (0, 0) + pad)
            actions[idx] = F.pad(actions[idx], (actions[idx].dim() - 1) * (0, 0) + pad)
            rewards[idx] = F.pad(rewards[idx], pad)
            timesteps[idx] = F.pad(timesteps[idx], pad)

        if actions[idx].dim() == 1: # (seq_len,)
            # actions here are scalar, should create a dummy dimension
            actions[idx] = actions[idx].unsqueeze(-1)

        # create a dummy dimension for reward
        rewards[idx] = rewards[idx].unsqueeze(-1)

    return {
        'states': torch.stack(states), # (batch_size, seq_len, state_dim)
        'actions': torch.stack(actions), # (batch_size, seq_len, action_dim)
        'rewards': torch.stack(rewards), # (batch_size, seq_len, 1)
        'timesteps': torch.stack(timesteps), # (batch_size, seq_len)
        'attention_mask': torch.stack(attention_mask), # (batch_size, seq_len)

        # this tells the huggingface trainer that eval loss will be returned, although labels are not explicitly provided
        'return_loss': True,
    }


def prepare_for_prediction(
        states: list,
        actions: list,
        rewards: list,
        timesteps: list,
        env: gym.Env,
        context_len: int = 10,
        device: str = 'cuda',
):
    """
    Prepare data for prediction.
    """
    datum = {
        'states': states[-context_len:],
        # adding dummy action and reward to make the length consistent
        # transformer would mask these out anyway
        'actions': actions[-context_len + 1:] + [env.action_space.sample()], # better to use 0 values?
        'rewards': rewards[-context_len + 1:] + [0],
        'timesteps': timesteps[-context_len:],
    }
    model_input = collate_fn([datum])
    model_input['return_loss'] = False

    # move to device
    for k in model_input:
        if hasattr(model_input[k], 'to'):
            model_input[k] = model_input[k].to(device)

    return model_input
