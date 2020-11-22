
import gym
from gym import spaces
import numpy as np

class PycolabAdditionalObsWrapper(gym.Wrapper):
    """
    Add a position based secondary observation signal for the pycolab
    5 room environment.
    """
    def __init__(self, env, signal_length=50):
        gym.Wrapper.__init__(self, env)

        self.signal_length = signal_length

        # randomly initialize secondary signals
        self.empty_signal = np.random.rand(signal_length)
        self.a_signal = np.random.rand(signal_length)
        self.b_signal = np.random.rand(signal_length)
        self.c_signal = np.random.rand(signal_length)
        self.d_signal = np.random.rand(signal_length)

    def step(self, action):

        image_obs, rew, done, info = self.env.step(action)

        pos_player = self.env.env.current_game._sprites_and_drapes['P'].position
        pos_a = self.env.env.current_game._sprites_and_drapes['a'].position
        pos_b = self.env.env.current_game._sprites_and_drapes['b'].position
        pos_c = self.env.env.current_game._sprites_and_drapes['c'].position
        pos_d = self.env.env.current_game._sprites_and_drapes['d'].position

        second_obs = np.zeros(self.signal_length)
        if abs(pos_player[0]-pos_a[0]) <= 1 and abs(pos_player[1]-pos_a[1]) <= 1:
            second_obs += self.a_signal
        elif abs(pos_player[0]-pos_b[0]) <= 1 and abs(pos_player[1]-pos_b[1]) <= 1:
            second_obs += self.b_signal
        elif abs(pos_player[0]-pos_c[0]) <= 1 and abs(pos_player[1]-pos_c[1]) <= 1:
            second_obs += self.c_signal
        elif abs(pos_player[0]-pos_d[0]) <= 1 and abs(pos_player[1]-pos_d[1]) <= 1:
            second_obs += self.d_signal
        else:
            second_obs += self.empty_signal

        return (image_obs, second_obs), rew, done, info