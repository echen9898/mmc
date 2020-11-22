"""An example implementation of pycolab games as environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from gym import spaces

from pycolab.examples import better_scrolly_maze, deepmind_maze, deepmind_5room, deepmind_5room_mmc, bottleneck, obstacles
from pycolab import cropping
from mazeworld.envs import pycolab_env

class MazeWorld(pycolab_env.PyColabEnv):
    """Custom maze world game.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 default_reward=0.):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd', 'e', '@']
        self.state_layer_chars = ['P', '#'] + self.objects
        super(MazeWorld, self).__init__(
            max_iterations=max_iterations,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            act_null_value=4,
            resize_scale=8,
            render_mode='uncropped')

    def make_game(self):
        self._croppers = self.make_croppers()
        return better_scrolly_maze.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]

class DeepmindMazeWorld_maze(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models game.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 default_reward=0.):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd', 'e']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        super(DeepmindMazeWorld_maze, self).__init__(
            max_iterations=max_iterations,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            act_null_value=4,
            resize_scale=8,
            render_mode='uncropped')

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_maze.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]

class DeepmindMazeWorld_5room(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models game.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 default_reward=0.):
        self.level = level
        self.objects = ['a', 'b']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        super(DeepmindMazeWorld_5room, self).__init__(
            max_iterations=max_iterations,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=8)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]

class DeepmindMazeWorld_5room_mmc(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models game.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 default_reward=0.):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        super(DeepmindMazeWorld_5room_mmc, self).__init__(
            max_iterations=max_iterations,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=8)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room_mmc.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]

class Bottleneck(pycolab_env.PyColabEnv):
    """Bottleneck test environment.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 default_reward=0.):
        self.level = level
        self.objects = ['a', 'b']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        super(Bottleneck, self).__init__(
            max_iterations=max_iterations,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=8)

    def make_game(self):
        self._croppers = self.make_croppers()
        return bottleneck.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]

class Obstacles(pycolab_env.PyColabEnv):
    """Obstacles test environment.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 default_reward=0.):
        self.level = level
        self.objects = ['a', 'b', 'c']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        super(Obstacles, self).__init__(
            max_iterations=max_iterations,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=8)

    def make_game(self):
        self._croppers = self.make_croppers()
        return obstacles.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]









        