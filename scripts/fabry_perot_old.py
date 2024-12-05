import gymnasium as gym
import numpy as np
import random

from experimental_quantities import *
from theoretical_noise import *
from calculate_output_powers import power_output_transmitted_fabry_perot

# data_series=signal_to_noise_conversion(df)

class Environment(gym.Env):
    metadata = {'render': ['human']}

    def __init__(self, history_length, noise_level, action_size):
        self.terminated = None
        self.truncated = None
        self.history_length = history_length
        self.history = [0] * self.history_length
        self.noise_level = noise_level
        self.action_size = action_size
        self.done = None
        self.position = None
        self.length_difference = 0

        actions = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        observations = gym.spaces.Box(low=-1, high=1, shape=(history_length,), dtype=np.float32)

        space_dict = {
            "observation_space": observations,
            "action_space": actions
        }

        self.action_space = space_dict["action_space"]
        self.observation_space = space_dict["observation_space"]
        self.delta_L = 0  # at start, the system is in equlibrium
        self.noise_ = 0
        self.real=0

        # properties of the interferometer

    def step(self, action):

        reward = 0
        self.timestep += 1
        self.delta_L = self.action_size * wavelength * action[
            0]  # 0.98 attention a difference of half wavelength corresponds to change of wavelength due to double roundtrip
#        self.noise_= noise_from_experiment(data_series)

        # noise_ = noise_from_experiment(signal_to_noise_conversion(df))

        noise_ = gaussian_noise(noise_factor_of_wavelength=self.noise_level)[0]

        power_equilibrium = power_output_transmitted_fabry_perot(0)
        power_noise = power_output_transmitted_fabry_perot(
            self.delta_L + noise_ + self.real) / power_equilibrium  # length difference as offset
        #self.length_difference += self.delta_L + noise_ ?
        self.real += self.delta_L + noise_
        self.length_difference += self.delta_L

        self.history.append(self.delta_L)
        self.history.append(np.float32(power_noise))
        self.history = self.history[-self.history_length:]
        observation = np.array(self.history, dtype=np.float32)

        if np.abs(power_noise) < 0.6:
            reward = -MaxTimesteps * 2
            self.terminated = True
        elif np.abs(power_noise) < 0.95:
            reward = -1
        else:
            reward = +1

        if self.timestep == MaxTimesteps:
            self.truncated = True

        info = {}
        return observation, reward, self.terminated, self.truncated, info

    def reset(
            self,
            *,
            seed: int | None = None,
            options: None = None,
    ):
        super().reset(seed=seed)
        self.delta_L = 0  # reset to initial position
        self.timestep = 0
        self.length_difference = 0
        self.real=0

        self.truncated = False
        self.terminated = False
        self.noise_ = 0
        observation = np.array([self.delta_L] * self.history_length, dtype=np.float32)
        info = {}

        return observation, info
