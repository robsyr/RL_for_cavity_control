import numpy as np
from fabry_perot_old import *
import os
from stable_baselines3 import SAC, DDPG, PPO, TD3
import pandas as pd
import matplotlib.pyplot as plt
import re

path = "C:\\Users\\robin\\Documents\\Uni\\MA\\models_15miosteps\\"

# define environment
env = Environment(history_length=3, noise_level=0.0001, action_size=0.001)
model_path = f'C:\\Users\\robin\\Documents\\Uni\\MA\\test\\models\\PPO_length_3_level_0.0001_action_size_0.001_try_0\\5000000.zip'
logdir = f'C:\\Users\\robin\\Documents\\Uni\\MA\\test\\logs\\PPO_length_3_level_0.0001_action_size_0.001_try_0'

model= PPO.load(model_path, tensorboard=logdir)
data_series = signal_to_noise_conversion(df)

# define pandas dataframe to store power outputs
max_episode_steps = MaxTimesteps
data = {"step": [i + 1 for i in range(max_episode_steps)]}
df_agent = pd.DataFrame(data)
df_no_agent = pd.DataFrame(data)

sample_size = 50
# test the environment
for j in range(sample_size):
    observations = []
    power_outputs = []

    power = []
    length_difference = 0
    for i in range(max_episode_steps):
        noise = gaussian_noise(noise_factor_of_wavelength=0.0001)[0]
        # noise= noise_from_experiment(data_series)
        power_equilibrium = power_output_transmitted_fabry_perot(0)
        power_noise = power_output_transmitted_fabry_perot(noise + length_difference) / power_equilibrium
        length_difference = noise + length_difference
        print(noise, length_difference)

        power.append(power_noise)
    df_no_agent[f'power{j}'] = power

    observation, info = env.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        observations.append(observation)
        action, states_ = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        power_outputs.append(observation[-1])
    if len(power_outputs) < max_episode_steps:
        zeros_to_add = max_episode_steps - len(power_outputs)
        power_outputs.extend([0] * zeros_to_add)
    df_agent[f'power_{j}'] = power_outputs

melted_df_agent = df_agent.melt(id_vars='step', var_name='Powers', value_name='Value')
melted_df_no_agent = df_no_agent.melt(id_vars='step', var_name='Powers', value_name='Value')

plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(16, 9))
sns.set_palette(palette='hls')
sns.lineplot(data=melted_df_no_agent, x="step", y="Value", errorbar="sd", label='No agent')
sns.lineplot(data=melted_df_agent, x="step", y="Value", errorbar="sd", label='agent')

plt.axhline(y=power_output_transmitted_fabry_perot(0), color='r', linestyle='--')
plt.ylim(0, power_output_transmitted_fabry_perot(0) + 0.1)
plt.xlabel('Steps')
plt.ylabel('Power Output')
plt.title('System evolution due to noise')
plt.annotate(text=f' Noise: 0.0002\n ', xy=[80, 0.1])

plt.legend()

plt.tight_layout()

# plt.annotate(f' Noise: {level}\n Model: PPO \n {logdir}\n Hisory Length {length}', [1, 1])
# plt.savefig(f'C:\\Users\\robin\Documents\\Uni\MA\Plots\Powerpoint_level_0.0002.jpg')
plt.show()
