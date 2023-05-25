import gym

# Register the environments with Gym
gym.envs.register(
    id='DarkRoom-v0',
    entry_point='envs.darkroom:DarkRoom',
)

gym.envs.register(
    id='DarkRoomCornerGoals-v0',
    entry_point='envs.darkroom:DarkRoomCornerGoals',
)

gym.envs.register(
    id='DarkRoomEdgeGoals-v0',
    entry_point='envs.darkroom:DarkRoomEdgeGoals',
)

gym.envs.register(
    id='DarkRoomHard-v0',
    entry_point='envs.darkroom:DarkRoomHard',
)
