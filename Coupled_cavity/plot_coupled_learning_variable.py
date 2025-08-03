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
        r"_action_size_(?P<action_size>[\de\.-]+)_tm_(?P<tm>[\d\.]+)_try_\d+_\d+"
    )
    match = re.search(pattern, filename)
    return match.groupdict() if match else {}


def plot_learning_curves_averaged_over_tm(
    logdirs, length_filter=None, smoothing="exponential", weight=0.9, window_size=10
):
    """Plot learning curves averaged over the `tm` parameter (grouped by algorithm, length, action_size)."""

    grouped_data = defaultdict(list)
    shortest_len = float("inf")

    # First pass: determine shortest sequence length
    for log in logdirs:
        try:
            results = load_results(log)
            x, _ = ts2xy(results, "timesteps")
            shortest_len = min(shortest_len, len(x))
        except Exception as e:
            print(f"Error processing {log}: {e}")

    # Second pass: process and group data
    for log in logdirs:
        try:
            results = load_results(log)
            x, y = ts2xy(results, "timesteps")
            x, y = x[:shortest_len], y[:shortest_len]

            # Smooth rewards
            if smoothing == "exponential":
                y = smooth_curve(y, method="exponential", weight=weight)
            elif smoothing == "rolling":
                if len(y) >= window_size:
                    y = smooth_curve(y, method="rolling", window_size=window_size)
                else:
                    print(f"Skipping {log}: not enough data points for rolling window.")
                    continue

            # Extract metadata
            metadata = extract_metadata_from_filename(os.path.basename(log))
            if metadata:
                if length_filter and metadata["length"] != str(length_filter):
                    continue

                key = f"{metadata['algorithm']}_length_{metadata['length']}_action_{metadata['action_size']}"
                grouped_data[key].append((x, y, float(metadata["tm"])))

        except Exception as e:
            print(f"Error processing {log}: {e}")

    # Plotting
    plt.rcParams.update({"font.size": 16})
    plt.figure(figsize=(16, 9))

    for group_key, runs in grouped_data.items():
        df_all = pd.DataFrame()
        for i, (x, y, tm) in enumerate(runs):
            if "timesteps" not in df_all:
                df_all["timesteps"] = x
            df_all[f"reward_{i}"] = y

        df_all["mean_reward"] = df_all.iloc[:, 1:].mean(axis=1)
        df_all["std_reward"] = df_all.iloc[:, 1:].std(axis=1)

        label = f"{group_key}_tm_avg_{len(runs)}"
        sns.lineplot(
            x="timesteps", y="mean_reward", data=df_all, label=label, errorbar=None
        )
        plt.fill_between(
            df_all["timesteps"],
            df_all["mean_reward"] - df_all["std_reward"],
            df_all["mean_reward"] + df_all["std_reward"],
            alpha=0.3,
        )

    plt.xlabel("Timesteps")
    plt.ylabel("Rewards")
    plt.title(f"Learning Curves Averaged Over `tm` (length={length_filter})")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Save the plot
    out_file = f"Learning_curve_length_{length_filter}_tm_avg.png"
    plt.savefig(out_file)
    print(f"Saved plot to {out_file}")


log_directory = "/home/syring/scripts/logs_coupled_L"
log_files = find_log_files(log_directory)

plot_learning_curves_averaged_over_tm(
    log_files,
    length_filter="6",  # Filter by length
    smoothing="rolling",  # Smoothing type
    window_size=300,
    weight=0.99,  # Only used for exponential smoothing
)
