"""
A Gym class for the following environment.

Dark Room: a 2D discrete POMDP where an agent spawns in a room and must ﬁnd a goal location. The agent only knows its
own (x, y) coordinates but does not know the goal location and must infer it from the reward. The room size is 9 × 9,
the possible actions are one step left, right, up, down, and no-op, the episode length is 20, and the agent resets at
the center of the map. We test two environment variants – Dark Room where the agent receives r = 1 every time the goal
is reached and Dark Room Hard, a hard exploration variant with a 17 × 17 size and sparse reward (r = 1 exactly once for
reaching the goal). When not r = 1, then r = 0.
"""
import warnings

import gym
import wandb
from gym import spaces
import numpy as np
from matplotlib import pyplot as plt

# gym warnings are annoying
warnings.filterwarnings("ignore")


class DarkRoom(gym.Env):
    def __init__(self, size=9, episode_length=20):
        self.size = size
        self.episode_length = episode_length

        self.observation_space = spaces.Box(low=0, high=self.size-1, shape=(2,), dtype=int)
        self.action_space = spaces.Discrete(5)

        self.goal_pos = self.generate_goal_pos()
        self.center_pos = [self.size // 2, self.size // 2]

        print(f'Goal position: {self.goal_pos}')

        self.reset()

    def generate_goal_pos(self):
        return [np.random.randint(self.size), np.random.randint(self.size)]

    def reset(self):
        # reset agent and goal to random positions
        self.agent_pos = self.center_pos.copy()
        self.steps_left = self.episode_length

        return tuple(self.agent_pos)

    def step(self, action):
        if action == 0:
            self.agent_pos[0] -= 1
        elif action == 1:
            self.agent_pos[0] += 1
        elif action == 2:
            self.agent_pos[1] -= 1
        elif action == 3:
            self.agent_pos[1] += 1

        self.agent_pos = np.clip(self.agent_pos, 0, self.size - 1).tolist()
        self.steps_left -= 1

        if self.agent_pos == self.goal_pos:
            reward = 1
        else:
            reward = 0

        done = (self.steps_left == 0)

        obs = tuple(self.agent_pos)

        return obs, reward, done, {}

    def render(self, mode='human', trajectory=[], log_name=None):
        assert mode == 'human', f'Unsupported rendering mode: {mode}'

        # Create a grid representing the dark room
        grid = np.zeros((self.size, self.size))
        grid[self.goal_pos[0], self.goal_pos[1]] = 1

        # Create a plot showing the grid and the agent's trajectory
        fig, ax = plt.subplots()
        ax.imshow(grid, cmap='gray_r')

        # only get states from the trajectory
        trajectory = [t[0] for t in trajectory]
        trajectory = np.array(trajectory)
        if len(trajectory) > 0:
            ax.plot(trajectory[:, 1], trajectory[:, 0], 'r-', linewidth=2)
        else:
            warnings.warn('Renderer got an empty trajectory.')

        ax.set_xticks(range(self.size))
        ax.set_yticks(range(self.size))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True)

        # if wandb is running, log the figure
        if wandb.run is not None and log_name:
            wandb.log({log_name: wandb.Image(fig)})
        else:
            plt.show()

        plt.close(fig)


class DarkRoomCornerGoals(DarkRoom):
    """
    A simple variant of the DarkRoom environment where the goal is always in one of the corners.
    """
    def generate_goal_pos(self):
        # the goal position is one of the corners, chosen randomly
        goal_pos_cands = [
            [0, 0],
            [0, self.size - 1],
            [self.size - 1, 0],
            [self.size - 1, self.size - 1],
        ]
        return goal_pos_cands[np.random.randint(len(goal_pos_cands))]


class DarkRoomEdgeGoals(DarkRoom):
    """
    A simple variant of the DarkRoom environment where the goal is always in the middle on one of the edges.
    Rationale: In this domain, the optimal policy is always deterministic (go all the way to any of the directions),
    which should be easy to imitate for a transformer model.
    For DarRoomCornerGoals, the agent can zigzag to the corner (left or up to reach the top-left corner), so a stochastic policy can be optimal.
    """
    def generate_goal_pos(self):
        # the goal position is in the middle of one of the edges, chosen randomly
        goal_pos_cands = [
            [0, self.size // 2],
            [self.size - 1, self.size // 2],
            [self.size // 2, 0],
            [self.size // 2, self.size - 1],
        ]
        return goal_pos_cands[np.random.randint(len(goal_pos_cands))]


class DarkRoomHard(DarkRoom):
    def __init__(self, size=17, episode_length=20):
        super().__init__(size=size, episode_length=episode_length)
        self.goal_reached = False

    def reset(self):
        self.goal_reached = False
        return super().reset()

    def step(self, action):
        obs, reward, done, _ = super().step(action)

        if not self.goal_reached and reward == 1:
            # only give reward once, when the agent reaches the goal for the first time
            self.goal_reached = True
            reward = 1
        else:
            reward = 0

        return obs, reward, done, {}
