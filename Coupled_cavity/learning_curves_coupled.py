import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np
import pandas as pd
import seaborn as sns
import os
import glob

base_path = os.path.expanduser("~/Plotting/logs_coupled/logs_tm_")

np.set_printoptions(suppress=True)


class PlotLearningCurves:
    def __init__(self, action_size, noise_level, history_length, try_):
        self.size = action_size
        self.noise = noise_level
        self.length = history_length
        self.try_ = try_

        self.tms = [0.01] + [round(0.1 * i, 1) for i in range(1, 10)] + [0.99]
        self.bin_size = 20000
        self.smoothing_weight = 0.98

    def smooth_curve(self, values, method="rolling", weight=0.9, window_size=100):
        if method == "exponential":
            return pd.Series(values).ewm(alpha=(1 - weight)).mean().values
        elif method == "rolling":
            return (
                pd.Series(values)
                .rolling(window=window_size, min_periods=1)
                .mean()
                .values
            )
        else:
            return values

    def process_run(self, x, y, bin_size=5000):
        df = pd.DataFrame({"timesteps": x, "value": y})
        df["binned_step"] = (df["timesteps"] // bin_size) * bin_size
        binned = df.groupby("binned_step")["value"].mean().reset_index()
        return binned

    def preprocessing_and_plotting(self):
        plt.rcParams.update({"font.size": 16})
        plt.figure(figsize=(16, 9))

        for tm in self.tms:
            logdir = []

            pattern = f"{base_path}{tm}/TQC_length_{self.length}_level_{self.noise}_action_size_{self.size}_tm_{tm}_try_{self.try_}_*"
            matches = glob.glob(os.path.expanduser(pattern))
            if not matches:
                print(f"No logs found for tm={tm}")
                continue
            logdir.extend(matches)

            if not logdir:
                continue

            df_merged = None
            counter = 1

            for log in logdir:
                try:
                    results = load_results(log)
                    x, y = ts2xy(results, "timesteps")
                    binned = self.process_run(x, y, bin_size=self.bin_size)
                    binned = binned.set_index("binned_step")
                    binned.rename(columns={"value": f"run_{counter}"}, inplace=True)
                    counter += 1

                    if df_merged is None:
                        df_merged = binned
                    else:
                        df_merged = df_merged.join(binned, how="outer")
                except Exception as e:
                    print(f"Error processing {log}: {e}")

            if df_merged is None or df_merged.empty:
                continue

            df_merged = df_merged.sort_index()
            reward_cols = [col for col in df_merged.columns if col.startswith("run_")]
            df_merged["mean_reward"] = df_merged[reward_cols].mean(axis=1)

            # Apply exponential smoothing on averaged curve
            df_merged["smoothed_mean"] = self.smooth_curve(
                df_merged["mean_reward"].values,
                method="exponential",
                weight=self.smoothing_weight,
            )

            df_merged = df_merged.reset_index()
            df_merged.rename(columns={"binned_step": "timesteps"}, inplace=True)

            sns.lineplot(
                x="timesteps", y="smoothed_mean", data=df_merged, label=f"tm {tm}"
            )

        plt.ylabel("Reward")
        plt.xlabel("Timesteps")
        plt.ylim(-210, 210)
        plt.title(f"Learning Curves")
        plt.legend(ncol=2)
        plt.grid()
        plt.tight_layout()

        output_dir = os.path.expanduser("~/Plotting/")
        os.makedirs(output_dir, exist_ok=True)
        filename = f"learning_curve_tm_comparison_len{self.length}_noise{self.noise}_size{self.size}.png"
        output_path = os.path.join(output_dir, filename)

        plt.savefig(output_path)
        print(f"Plot saved to: {output_path}")
        plt.show()


plotter = PlotLearningCurves(
    action_size=2e-11, noise_level=4e-12, history_length=4, try_=0
)

plotter.preprocessing_and_plotting()
