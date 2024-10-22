import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np
import pandas as pd
import seaborn as sns

# path to logs
path = "C:\\Users\\robin\Documents\\Uni\MA\logs_15miosteps\\"
# path = "C:\\Users\\robin\Documents\\Uni\MA\\finally_right_env\logs\\"
np.set_printoptions(suppress=True)


# path= "C:\\Users\\robin\Documents\\Uni\MA\\finally_right_env\\logs\\"


class PlotLearningCurves():
    def __init__(self, action_size, noise_level, history_length, try_, plotting_parameter):
        self.size = action_size
        self.noise = noise_level
        self.length = history_length
        self.try_ = try_
        self.plot_param = plotting_parameter
        self.data = None

        # initializing values
        self.tries = [0, 1, 2, 3, 4]
        self.lengths = [1, 2, 3, 4, 5, 6, 7, 8]
        self.noise_list = [0.0002]
        self.sizes = [0.0009]

        # defining the colour scheme
        self.color = 'hls'
        # self.color= 'dark'

    # bring the data in the right format
    def preprocessing(self):
        if self.plot_param == 'length':
            logdir = [
                f"{path}\\action_size_{self.size}\\Length_{i}_level_{self.noise}\\PPO_length_{i}_level_{self.noise}_try_{self.try_}"
                for i in self.lengths]
        elif self.plot_param == 'noise':
            logdir = [
                f"{path}\\action_size_{self.size}\\Length_{self.length}_level_{i}\\PPO_length_{self.length}_level_{i}_try_{self.try_}"
                for i in self.noise_list]
        elif self.plot_param == 'try':
            logdir = [
                f"{path}\\action_size_{self.size}\\Length_{self.length}_level_{self.noise}\\PPO_length_{self.length}_level_{self.noise}_try_{i}"
                for i in self.tries]
        elif self.plot_param == 'action_size':
            logdir = [
                f"{path}\\action_size_{i}\\Length_{self.length}_level_{self.noise}\\PPO_length_{self.length}_level_{self.noise}_try_{i}"
                for i in self.sizes]

        df = pd.DataFrame()

        timesteps_saved = False
        counter = 1
        shortest_len = float('inf')
        # get the shortest len in df
        for log in logdir:
            results = load_results(log)
            x, y = ts2xy(results, 'timesteps')
            min_len = min(len(x), len(y))
            shortest_len = min(shortest_len, min_len)
            # shortest_len= 43000

        for log in logdir:
            results = load_results(log)
            x, y = ts2xy(results, 'timesteps')
            x = x[:shortest_len]
            y = y[:shortest_len]

            if not timesteps_saved:
                df['timesteps'] = x
                timesteps_saved = True

            df[f'log_{counter}'] = y
            counter += 1

        # get the rolling average adjust the window for smoothing
        df = df.rolling(window=300, min_periods=1).mean()
        self.data = df

    def plotting(self):
        if self.data is None:
            print("Data not preprocessed. Run preprocessing first.")
            return
        melted_df = self.data.melt(id_vars='timesteps', var_name=f'{self.plot_param}', value_name='Value')

        plt.rcParams.update({'font.size': 16})
        plt.figure(figsize=(16, 9))
        plt.ylabel('Reward')
        plt.xlabel('Steps')
        plt.ylim(-105, 51)

        if self.plot_param == 'noise':
            plt.title(f'Learning curves for action size of {self.size} and history length of {self.length} by {self.plot_param} ', fontsize=18)
            legend = [f'{self.plot_param}: {i}' for i in self.noise_list]
            sns.set_palette(f"{self.color}", n_colors=len(self.noise_list))
        elif self.plot_param == 'length':
            plt.title(f'Learning curves for action size of {self.size} and noise level of {self.noise} by {self.plot_param} ', fontsize=18)
            legend = [f'{self.plot_param}: {i + 1}' for i in self.lengths]
            sns.set_palette(f"{self.color}", n_colors=len(self.lengths))
        elif self.plot_param== 'try':
            plt.title(
                f"Learning curves for action size of {self.size} and noise level of {self.noise} and history length of {self.length} by {self.plot_param}'s ",
                fontsize=18)
            legend = [f'{self.plot_param}: {i + 1}' for i in self.tries]
            sns.set_palette(f"{self.color}", n_colors=len(self.tries))
        elif self.plot_param=='action_size':
            plt.title(f'Learning curve for noise_level of {self.noise} and history_length of {self.length} by {self.plot_param}', fontsize=18)
            legend = [f'{self.plot_param}: {i + 1}' for i in self.sizes]
            sns.set_palette(f"{self.color}", n_colors=len(self.sizes))

        sns.lineplot(data=melted_df, x="timesteps", hue=self.plot_param, y="Value", errorbar=None)
        plt.legend(legend)
        plt.tight_layout()

        """
        if self.plot_param=='noise':
            plt.savefig(f'C:\\Users\\robin\Documents\\Uni\MA\Learning_curves\\LC_by_{self.plot_param}_action_size_{self.size}_lenght_{self.length}_try_{self.try_}.png')
        elif self.plot_param=='length':
            plt.savefig(f'C:\\Users\\robin\Documents\\Uni\MA\Learning_curves\\LC_by_{self.plot_param_action_size_{self.size}}_noise_{self.noise}_try_{self.try_}.png')
        elif self.plot_param=='try':
            plt.savefig(f'C:\\Users\\robin\Documents\\Uni\MA\Learning_curves\\LC_by_{self.plot_param}_action_size_{self.size}_noise_{self.noise}_length_{self.length}.png')
        elif self.plot_param=='action_size':
            plt.savefig(f'C:\\Users\\robin\Documents\\Uni\MA\Learning_curves\\LC_by_{self.plot_param}_noise_{self.noise}_length_{self.length}_try_{self.try_}.png')
        """

        plt.show()

    def output_plot(self):
        self.preprocessing()
        self.plotting()


data_plotter = PlotLearningCurves(noise_level=0.0003, history_length=5, try_=0, plotting_parameter='length')
data_plotter.output_plot()
