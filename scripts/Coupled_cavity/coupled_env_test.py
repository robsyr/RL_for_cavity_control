import gymnasium as gym
import numpy as np
from typing import Optional
from calculate_output_powers import power_output_transmitted_coupled_cavity
from experimental_quantities import *


class CoupledEnvironment(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self, history_length: int, noise_level: float, action_size: float, tm: float
    ):
        super(CoupledEnvironment, self).__init__()

        # Environment configuration
        self.history_length = history_length
        self.noise_level = noise_level
        self.action_size = action_size
        self.tm = tm

        # Initialize action and observation space
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(history_length,), dtype=np.float32
        )

        # Initialize variables
        self.reset_variables()

    def reset_variables(self):
        """Reset all state variables."""
        self.real_L1 = 0
        self.real_L2 = 0
        self.timestep = 0

        # Episode status
        self.terminated = False
        self.truncated = False

        # History
        self.history = [0] * self.history_length

    def calculate_power_noise(
        self, delta_L1: float, delta_L2: float, noise1: float, noise2: float
    ) -> float:
        """Calculate output power with noise."""
        power_equilibrium = power_output_transmitted_coupled_cavity(
            0, 0, 0, wavelength / 4, self.tm
        )

        real_L1 = self.real_L1 + delta_L1 + noise1
        real_L2 = self.real_L2 + delta_L2 + noise2

        power_noise = (
            power_output_transmitted_coupled_cavity(
                real_L1, real_L2, noise1, wavelength / 4 + noise2, self.tm
            )
            / power_equilibrium
        )

        # Update real cavity lengths
        self.real_L1 = real_L1
        self.real_L2 = real_L2

        return power_noise

    def compute_reward(self, power_noise: float) -> int:
        """Compute reward based on power noise level."""
        reward = 40 * power_noise - 39

        if np.abs(power_noise) < 0.95:
            self.terminated = True
            return -MaxTimesteps

        return reward

    def step(self, action: np.ndarray):
        """Perform one step in the environment."""
        self.timestep += 1

        # Extract actions
        delta_L1 = self.action_size * action[0]
        delta_L2 = self.action_size * action[1]

        # Generate noise
        noise1 = (
            np.random.normal(0.0, self.noise_level) if np.random.rand() >= 0.5 else 0
        )
        noise2 = (
            np.random.normal(0.0, self.noise_level) if noise1 == 0 else 0
        )  # Ensuring one noise term is always zero

        # Compute power noise
        power_noise = self.calculate_power_noise(delta_L1, delta_L2, noise1, noise2)

        # Update history and prepare observation
        self.update_history(action[0], action[1], power_noise)
        observation = np.array(self.history, dtype=np.float32)

        # Compute reward
        reward = self.compute_reward(power_noise)

        # Check if max timesteps is reached
        if self.timestep >= MaxTimesteps:
            self.truncated = True

        return observation, reward, self.terminated, self.truncated, {}

    def update_history(self, action1: float, action2: float, power_noise: float):
        """Update the history buffer with current delta_L1, delta_L2, and power noise."""
        self.history.extend([action1, action2, np.float32(power_noise)])
        self.history = self.history[-self.history_length :]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to its initial state."""
        super().reset(seed=seed)
        self.reset_variables()

        # Initialize observation as zero-filled history
        observation = np.zeros(self.history_length, dtype=np.float32)
        return observation, {}
