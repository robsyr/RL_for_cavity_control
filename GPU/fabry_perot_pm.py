import gymnasium as gym
import numpy as np
from typing import Optional
from calculate_output_powers import power_output_transmitted_fabry_perot
from experimental_quantities import * 




class Environment(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, history_length: int, noise_level: float, action_size: float):
        """
        Initialize the custom environment.
        """
        super(Environment, self).__init__()
        
        # Environment configuration
        self.history_length = history_length
        self.noise_level = noise_level
        self.action_size = action_size

        # Action and observation spaces
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(history_length,), dtype=np.float32)
        
        # Initialize state variables
        self.reset_variables()
    
    def reset_variables(self):
        """
        Reset internal state variables.
        power_equilibrium = power_outpu
        """
        self.delta_L = 0  # Initial system equilibrium
        self.timestep = 0
        self.real = 0
        self.noise_ = 0
        
        # Tracking episode status
        self.terminated = False
        self.truncated = False
        self.done = False
        
        # History buffer
        self.history = [0] * self.history_length

    def calculate_power_noise(self, delta_L: float, noise: float) -> float:
        """
        Calculate power noise based on current state.
        """
        power_equilibrium = power_output_transmitted_fabry_perot(0)
        power_noise = power_output_transmitted_fabry_perot(delta_L + noise + self.real) / power_equilibrium
        self.real += delta_L + noise  # Update real state with delta and noise
        return power_noise

    def compute_reward(self, power_noise: float) -> int:
        """
        Compute reward based on the power noise level.
        """
        
        reward = 40*power_noise -39


        if np.abs(power_noise) < 0.95:
            self.terminated = True
            return -MaxTimesteps

        return reward 

        
    
    def step(self, action: np.ndarray):
        """
        Perform one step in the environment given an action.
        """
        self.timestep += 1
        # Calculate length adjustment based on action
        self.delta_L = self.action_size * action[0] 
        # print("action", action[0])
        
        # Generate noise and calculate power noise
        noise_ = np.random.normal(0.0, self.noise_level)
        # print("noise", noise_)
        # print("deltaL", self.delta_L)
        power_noise = self.calculate_power_noise(self.delta_L, noise_)
        
        # Update history and prepare observation
        self.update_history(action[0], power_noise)
        observation = np.array(self.history, dtype=np.float32)
        
        # Compute reward
        reward = self.compute_reward(power_noise)
        
        # Check if max timesteps reached
        if self.timestep >= MaxTimesteps:
            self.truncated = True

        info = {}
        return observation, reward, self.terminated, self.truncated, info

    def update_history(self, action: float, power_noise: float):
        """
        Update the history buffer with current delta_L and power noise.
        """
        self.history.append(action)
        self.history.append(np.float32(power_noise))
        self.history = self.history[-self.history_length:]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the environment to initial state.
        """
        super().reset(seed=seed)
        self.reset_variables()
        
        # Initialize observation to be zero-filled history
        observation = np.array([self.delta_L] * self.history_length, dtype=np.float32)
        info = {}
        return observation, info
