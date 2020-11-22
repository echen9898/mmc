
import os
import argparse
import numpy as np

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import gym
from gym import Wrapper
from gym.wrappers.time_limit import TimeLimit
from gym.wrappers import Monitor
from envs.general_wrappers import *
from envs.pycolab_wrappers import *
import mazeworld

from models.utils import to_onehot

import matplotlib.pyplot as plt
from models.mmc import *

"""
     ###################
     ##               ##
     # #             # #
     #  # a         #  #
     #   #         #   #
     #    #### ####    #
     #    #### ####    #
     #    ##     ##    #
     # d  ##     ##    #
     #        P        #
     #    ##     ##  c #
     #    ##     ##    #
     #    #### ####    #
     #    #### ####    #
     #   #         #   #
     #  #           #  #
     # # b           # #
     ##               ##
     ###################

P = player (can move up, down, left, right, and noop)
a = fixed object (changes position at the start of each episode)
b = fixed object (changes position at the start of each episode)
c = fixed object (changes position at the start of each episode)
d = fixed object (changes position at the start of each episode)

Observations are (5x5xc) local observations centered around the player,
with c = number_of_objects + 1. There is a channel for each object, plus
the walls to denote their presence with a binary mask.

A secondary observation is generated in PycolabAdditionalObsWrapper
(see pycolab_wrappers.py for details). Done manually for just this map
at the moment, but can always expand or generalize. Objects generate
unique vectors, while if the agent is not in contact with an object
(one cell away) a fixed randomly initialized signal is returned.
"""

def run(args):
    # model = MMC(image_shape=(5,5,3), action_size=5)
    env = gym.make('Deepmind5RoomMMC-v0')
    env = PycolabAdditionalObsWrapper(env, signal_length=50) # adds an additional position based observation signal

    # video recordings
    if args.record:
        env = Monitor(env, '/mmc/mmc_vids', video_callable=lambda episode_id: True, force=True)

    env.reset()
    for _ in range(args.batch_T):
        action = env.action_space.sample()
        observations, reward, done, info = env.step(action)
        obs_1, obs_2 = observations
    env.close()


if __name__ == '__main__':

    # CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-record', type=bool, default=False, help='Whether or not to record video')
    parser.add_argument('-batch_T', type=int, default=500, help='Number of timesteps to run (500 is 1 episode for pycolab)')
    args = parser.parse_args()

    run(args)














