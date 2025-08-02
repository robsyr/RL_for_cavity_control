import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy
import re


def find_log_files(base_dir, filename="monitor.csv"):
    log_files = []
    for root, _, files in os.walk(base_dir):
        if filename in files:
            log_files.append(root)
    return log_files


def extract_metadata_from_filename(filename):
    pattern = (
        r"(?P<algorithm>\w+)_length_(?P<length>\d+)_level_(?P<level>[\de\.-]+)"
        r"_action_size_(?P<action_size>[\de\.-]+)_try_\d+_\d+"
    )
    match = re.search(pattern, filename)
    return match.groupdict() if match else {}


def process_run(x, y, bin_size=5000):
    df = pd.DataFrame({"Step": x, "Value": y})
    df["binned_step"] = (df["Step"] // bin_size) * bin_size
    grouped = df.groupby("binned_step")["Value"].mean().reset_index()
    grouped.rename(columns={"Value": "run_mean"}, inplace=True)
    return grouped


def plot_grouped_learning_curves(
    logdir,
    action_size_filter=None,
    algorithm_filter=None,
    smoothing_alpha=0.1,
    bin_size=5000,
):
    all_runs = []

    for log in logdir:
        try:
            results = load_results(log)
            x, y = ts2xy(results, "timesteps")

            metadata = extract_metadata_from_filename(os.path.basename(log))
            if metadata:
                if action_size_filter and metadata["action_size"] != str(
                    action_size_filter
                ):
                    continue
                if algorithm_filter and metadata["algorithm"] != algorithm_filter:
                    continue

                df_run = process_run(x, y, bin_size=bin_size)
                df_run["algorithm"] = metadata["algorithm"]
                df_run["length"] = metadata["length"]
                all_runs.append(df_run)

        except Exception as e:
            print(f"Error processing {log}: {e}")

    if not all_runs:
        print("No runs matched the filter criteria.")
        return

    df_all = pd.concat(all_runs, ignore_index=True)

    # Compute per-length mean over runs
    final_list = []
    for length in sorted(df_all["length"].unique(), key=lambda x: int(x)):
        subset = df_all[df_all["length"] == length]

        grouped = subset.groupby("binned_step")["run_mean"].mean().reset_index()
        grouped["smoothed_mean"] = grouped["run_mean"].ewm(alpha=smoothing_alpha).mean()
        grouped["length"] = length

        final_list.append(grouped)

    stats_df = pd.concat(final_list, ignore_index=True)

    # Plotting
    plt.figure(figsize=(16, 9))
    plt.rcParams.update(
        {
            "axes.titlesize": 18,  # title font size
            "axes.labelsize": 16,  # x/y label font size
            "xtick.labelsize": 14,  # x-axis tick font size
            "ytick.labelsize": 14,  # y-axis tick font size
            "legend.fontsize": 14,  # legend font size
            "font.size": 14,  # default text size
        }
    )
    for length in sorted(stats_df["length"].unique(), key=lambda x: int(x)):
        subset = stats_df[stats_df["length"] == length]
        plt.plot(
            subset["binned_step"],
            subset["smoothed_mean"],
            label=f"History Length {length}",
        )

    plt.ylim(140, 200)
    # plt.ylim(-210, 210)
    plt.xlabel("Timesteps")
    plt.ylabel("Rewards")
    plt.title("Learning Curves")
    plt.legend(loc="lower right")
    plt.grid()

    output_path = f"/home/robin/Dokumente/Masterarbeit/RL_for_cavity_control/result_images/Learning_curve_action_size_{action_size_filter}_algorithm_{algorithm_filter}_detailed.png"
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.close()


if __name__ == "__main__":
    log_directory = "/home/robin/Server/Xmas_run/History_length/logs/SAC"
    log_files = find_log_files(log_directory)

    plot_grouped_learning_curves(
        log_files,
        action_size_filter="4e-11",
        algorithm_filter="SAC",
        smoothing_alpha=0.01,  # adjust smoothing strength
        bin_size=5000,  # adjust binning
    )
