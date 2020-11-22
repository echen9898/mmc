from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gym.envs.registration import register

from mazeworld.envs import MazeWorld, DeepmindMazeWorld_maze, DeepmindMazeWorld_5room, DeepmindMazeWorld_5room_mmc

register(
    id='Maze-v0',
    entry_point='mazeworld.envs:MazeWorld',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500})

register(
    id='DeepmindMaze-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_maze',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500})

register(
    id='Deepmind5Room-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5room',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500})

register(
    id='Deepmind5RoomMMC-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5room_mmc',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500})

register(
    id='Bottleneck-v0',
    entry_point='mazeworld.envs:Bottleneck',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500})

register(
    id='Obstacles-v0',
    entry_point='mazeworld.envs:Obstacles',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500})