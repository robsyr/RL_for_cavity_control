    def compute_reward(self, power_noise: float) -> int:
        """
        Compute reward based on the power noise level.
        """

        if np.abs(power_noise) < 0.8:
            self.terminated = True
            reward = -MaxTimesteps

        elif np.abs(power_noise) < 0.9:
            reward = -1

        else:
            reward = 20*np.abs(power_noise)-19

        return reward



MaxTimesteps = 000
