import numpy as np
from Fabry_perot.fabry_perot import *
import os
from stable_baselines3 import SAC
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from experimental_quantities import *

# Define environments
env1 = Environment(history_length=4, noise_level=4e-12, action_size=4e-11)
env2 = Environment(history_length=1, noise_level=4e-12, action_size=4e-11)

# Paths to models
logdir1 = "/home/syring/logs/SAC_length_4_level_4e-12_action_size_4e-11_try_1"
logdir2 = "/home/syring/logs/SAC_length_1_level_4e-12_action_size_4e-11_try_0"

# Choose best models
model_path1 = os.path.join(logdir1, "best_model.zip")
model_path2 = os.path.join(logdir2, "best_model.zip")
model1 = SAC.load(model_path1, tensorboard=logdir1)
model2 = SAC.load(model_path2, tensorboard=logdir2)

# Define pandas dataframe to store power outputs
max_episode_steps = MaxTimesteps * 10
data = {"step": [i + 1 for i in range(max_episode_steps)]}
df_no_agent = pd.DataFrame(data)
df_agent1 = pd.DataFrame(data)
df_agent2 = pd.DataFrame(data)

sample_size = 50

# Test no-agent baseline
for j in range(sample_size):
    power = []
    length_difference = 0
    for i in range(max_episode_steps):
        noise = np.random.normal(0.0, 4e-12)
        power_equilibrium = power_output_transmitted_fabry_perot(0)
        power_noise = (
            power_output_transmitted_fabry_perot(noise + length_difference)
            / power_equilibrium
        )
        length_difference = noise + length_difference
        power.append(power_noise)
    df_no_agent[f"power{j}"] = power

# Test the first agent
for j in range(sample_size):
    observation, info = env1.reset()
    power_outputs = []
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _ = model1.predict(observation)
        observation, reward, terminated, truncated, info = env1.step(action)
        power_outputs.append(observation[-1])
    if len(power_outputs) < max_episode_steps:
        power_outputs += [None] * (max_episode_steps - len(power_outputs))
    df_agent1[f"power{j}"] = power_outputs[:max_episode_steps]

# Test the second agent
for j in range(sample_size):
    observation, info = env2.reset()
    power_outputs = []
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _ = model2.predict(observation)
        observation, reward, terminated, truncated, info = env2.step(action)
        power_outputs.append(observation[-1])
    if len(power_outputs) < max_episode_steps:
        power_outputs += [None] * (max_episode_steps - len(power_outputs))
    df_agent2[f"power{j}"] = power_outputs[:max_episode_steps]

# Apply rolling average smoothing
window_size = 20  # tune the smoothing
for df in [df_no_agent, df_agent1, df_agent2]:
    power_cols = [col for col in df.columns if col.startswith("power")]
    df[power_cols] = df[power_cols].rolling(window=window_size, min_periods=1).mean()

# Prepare for plotting
melted_no_agent = df_no_agent.melt(
    id_vars="step", var_name="Powers", value_name="Value"
)
melted_agent1 = df_agent1.melt(id_vars="step", var_name="Powers", value_name="Value")
melted_agent2 = df_agent2.melt(id_vars="step", var_name="Powers", value_name="Value")

plt.rcParams.update({"font.size": 16})
plt.figure(figsize=(16, 9))
sns.set_palette(palette="hls")

sns.lineplot(data=melted_no_agent, x="step", y="Value", errorbar="sd", label="No agent")
sns.lineplot(
    data=melted_agent2, x="step", y="Value", errorbar="sd", label="agent with length: 1"
)
sns.lineplot(
    data=melted_agent1, x="step", y="Value", errorbar="sd", label="agent with length: 4"
)

plt.xlabel("Steps")
plt.ylabel("Power Output")
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig("comparison_plot.png")
plt.show()
