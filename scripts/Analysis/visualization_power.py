import numpy as np
from fabry_perot_new_env_adapted import *
from experimental_quantities import *
from import_noise import gaussian_noise
from calculate_output_powers import power_output_transmitted_fabry_perot
import seaborn as sns
import os
from stable_baselines3 import SAC, DDPG, PPO, TD3
import pandas as pd
import matplotlib.pyplot as plt
import re

""" 
Goal is to extract all information out of logfile like history length and noise_level
Question is what model to take, i.e what is the best model from reward perspective ?
"""

# path should link us to the directory where logs and models are located
path_model = "C:\\Users\\robin\\Documents\\Uni\\MA\\models_new_env"
path_logs = "C:\\Users\\robin\\Documents\\Uni\\MA\\logs_new_env\\"

data_series=signal_to_noise_conversion(df)

class PlotPowerCurves():
    def __init__(self, action_size, noise_level, history_length, try_, zip_):
        self.size = action_size
        self.noise = noise_level
        self.length = history_length
        self.try_ = try_
        self.zip_ = zip_

        self.data_agent = None
        self.data_no_agent = None

        self.color = 'hls'

    def preprocessing(self):
        #logdir = f'{path_logs}\\action_size_{self.size}\\Length_{self.length}_level_{self.noise}\\PPO_length_{self.length}_level_{self.noise}_try_{self.try_}'
        #model_path = f'{path_model}\\action_size_{self.size}\\Length_{self.length}_level_{self.noise}\\PPO_length_{self.length}_level_{self.noise}_try_{self.try_}\\{self.zip_}.zip'
        logdir= f'{path_logs}\\PPO_length_{self.length}_level_{self.noise}_action_size_{self.size}_try_{self.try_}'
        model_path= f'{path_model}\\PPO_length_{self.length}_level_{self.noise}_action_size_{self.size}_try_{self.try_}\\{self.zip_}.zip'

        env = Environment(history_length=self.length, noise_level=self.noise, action_size=self.size)
        model = PPO.load(model_path, tensorboard=logdir)

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
                noise = noise_from_experiment(data_series)

                #noise = gaussian_noise(noise_factor_of_wavelength=self.noise)[0]
                power_equilibrium = power_output_transmitted_fabry_perot(0)
                power_noise = power_output_transmitted_fabry_perot(noise + length_difference) / power_equilibrium
                length_difference = noise + length_difference
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

        # Melt the DataFrame to have 'Episode' as a categorical variable
        melted_df_agent = df_agent.melt(id_vars='step', var_name='Powers', value_name='Value')
        log_anno = logdir.replace('/', '_')

        melted_df_no_agent = df_no_agent.melt(id_vars='step', var_name='Powers', value_name='Value')

        self.data_agent = melted_df_agent
        self.data_no_agent = melted_df_no_agent

    def plotting(self):
        plt.rcParams.update({'font.size': 16})
        plt.figure(figsize=(16, 9))
        sns.set_palette(palette=self.color)
        sns.lineplot(data=self.data_agent, x="step", y="Value", errorbar='sd', label='agent')
        sns.lineplot(data=self.data_no_agent, x="step", y="Value", errorbar="sd", label='no agent')

        plt.axhline(y=power_output_transmitted_fabry_perot(0), color='r', linestyle='--')
        plt.ylim(0, power_output_transmitted_fabry_perot(0) + 0.1)
        plt.xlabel('Steps')
        plt.ylabel('Power Output')
        plt.title('Agent Behavior Over Steps')
        plt.annotate(text=f' Noise: {self.noise}\n length: {self.length}\n action_size: {self.size}', xy=[80, 0.1])
        plt.legend()
        #        plt.annotate(f' Noise: {level}\n Model: PPO \n {logdir}\n Hisory Length {length}', [1, 1])
        plt.tight_layout()
        plt.savefig(
            f'C:\\Users\\robin\Documents\\Uni\MA\Plots\Powerhold\\new_env\\PPO_length_action_size_{self.size}_{self.length}_level_{self.noise}_try_{self.try_}.jpg')
        plt.show()

    def output_plot(self):
        self.preprocessing()
        self.plotting()


PlotPowerCurves(action_size=0.002, noise_level=0.0002, history_length=3, try_=0, zip_=1970000).output_plot()
