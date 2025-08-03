import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy
import seaborn as sns
import re
from collections import defaultdict


def find_log_files(base_dir, filename="monitor.csv"):
    """Recursively search for and return paths to directories containing specified log files."""
    log_files = []
    for root, _, files in os.walk(base_dir):
        if filename in files:
            log_files.append(root)
    return log_files


def smooth_curve(values, method="exponential", weight=0.9, window_size=100):
    """Smooth a sequence of values using selected smoothing techniques."""
    if method == "exponential":
        return pd.Series(values).ewm(alpha=(1 - weight)).mean().values
    elif method == "rolling":
        return (
            pd.Series(values).rolling(window=window_size, min_periods=1).mean().values
        )
    else:
        raise ValueError("Invalid smoothing method. Choose 'exponential' or 'rolling'.")


def extract_metadata_from_filename(filename):
    """Extract metadata from a filename using a regular expression pattern."""
    pattern = (
        r"(?P<algorithm>\w+)_length_(?P<length>\d+)_level_(?P<level>[\de\.-]+)"
        r"_action_size_(?P<action_size>[\de\.-]+)_try_\d+_\d+"
    )
    match = re.search(pattern, filename)
    return match.groupdict() if match else {}


def plot_grouped_learning_curves(
    logdir, length_filter=None, smoothing="exponential", weight=0.9, window_size=10
):
    """Plot learning curves grouped by extracted metadata."""

    grouped_data = defaultdict(lambda: defaultdict(list))
    shortest_len = float("inf")

    # Determine the shortest length of timesteps among all logs
    for log in logdir:
        try:
            results = load_results(log)
            x, _ = ts2xy(results, "timesteps")
            shortest_len = min(shortest_len, len(x))
        except Exception as e:
            print(f"Error processing {log}: {e}")

    # Process each log file, apply filters, and smooth the reward data
    for log in logdir:
        try:
            results = load_results(log)
            x, y = ts2xy(results, "timesteps")
            x, y = x[:shortest_len], y[:shortest_len]

            # Apply selected smoothing method
            if smoothing == "exponential":
                y = smooth_curve(y, method="exponential", weight=weight)
            elif smoothing == "rolling":
                if len(y) >= window_size:
                    y = smooth_curve(y, method="rolling", window_size=window_size)
                else:
                    print(f"Skipping {log}: not enough data points for rolling window.")
                    continue

            # Extract metadata and apply filters
            metadata = extract_metadata_from_filename(os.path.basename(log))
            if metadata:
                if length_filter and metadata["length"] != str(length_filter):
                    continue

                # Store data grouped by algorithm and action size
                key = f"{metadata['algorithm']}_length_{metadata['length']}"
                action_size_key = metadata["action_size"]
                grouped_data[key][action_size_key].append((x, y))

        except Exception as e:
            print(f"Error processing {log}: {e}")

    # Set up plotting parameters and figure
    plt.rcParams.update({"font.size": 16})
    plt.figure(figsize=(16, 9))

    # Plot the mean and standard deviation for each group
    for group, action_data in grouped_data.items():

        sorted_action_data = sorted(
            action_data.items(), key=lambda x: float(x[0])
        )  # necessary to interpret scientific notation as floats
        for action_size, data in sorted_action_data:
            df_group = pd.DataFrame()

            for i, (x, y) in enumerate(data):
                if "timesteps" not in df_group:
                    df_group["timesteps"] = x
                df_group[f"reward_{i}"] = y

            # Calculate mean and standard deviation
            df_group["mean_reward"] = df_group.iloc[:, 1:].mean(axis=1)
            df_group["std_reward"] = df_group.iloc[:, 1:].std(axis=1)

            # Generate plots
            sns.lineplot(
                x="timesteps",
                y="mean_reward",
                data=df_group,
                label=f"{group}_action_{action_size}",
                errorbar=None,
            )
            plt.fill_between(
                df_group["timesteps"],
                df_group["mean_reward"] - df_group["std_reward"],
                df_group["mean_reward"] + df_group["std_reward"],
                alpha=0.3,
            )

    # Add labels, title, and legend to the plot
    plt.xlabel("Timesteps")
    plt.ylabel("Rewards")
    plt.title(f"Learning Curves (Filtered by length={length_filter})")
    plt.legend()
    # plt.ylim(150,200)

    plt.grid()
    # plt.show()
    plt.savefig(
        f"/home/robin/Dokumente/Masterarbeit/RL_for_cavity_control/result_images/Learning_curve_algo_length_{length_filter}.png"
    )


# Example usage demonstrating the ability to filter by length and visualize multiple action sizes.
log_directory = "/home/robin/Server/Xmas_run/TQC/logs"
log_files = find_log_files(log_directory)

plot_grouped_learning_curves(
    log_files,
    length_filter="4",  # Specify the history length to filter by
    smoothing="rolling",
    window_size=300,
    weight=0.99,
)
