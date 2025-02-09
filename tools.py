import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_all_environments(csv_file: str, output_dir: str = "plots", window: int = 10):
    """
    Creates one figure per unique environment in 'csv_file'.
    Each figure shows raw + smoothed (if enough data) reward curves.
    """
    df = pd.read_csv(csv_file)
    if df.empty:
        print(f"CSV file '{csv_file}' is empty or not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    unique_envs = df["Environment"].unique()
    print("Found environments:", unique_envs)

    for env_id in unique_envs:
        env_data = df[df["Environment"] == env_id]
        if env_data.empty:
            print(f"No data for environment '{env_id}'. Skipping.")
            continue

        # Extract rewards
        rewards = env_data["AverageReward"].values
        print(f"\nEnvironment: {env_id}")
        print(f"  Number of episodes: {len(rewards)}")

        smoothed_rewards = None
        if len(rewards) >= window:
            smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode="valid")

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(rewards, label="Raw Rewards", linewidth=1)

        if smoothed_rewards is not None and len(smoothed_rewards) > 0:
            plt.plot(range(len(smoothed_rewards)),
                     smoothed_rewards,
                     label=f"Smoothed (window={window})",
                     linestyle='--', linewidth=2)

        plt.xlabel("Episodes")
        plt.ylabel("Total Reward")
        plt.title(f"Baseline on {env_id}")
        plt.legend()
        plt.grid(True)

        # Save
        output_path = os.path.join(output_dir, f"{env_id}_baseline.png")
        plt.savefig(output_path, dpi=200)
        print(f"  Plot saved to {output_path}")
        plt.close()


def plot_environments_side_by_side(
    csv_file: str,
    output_file: str = "plots/baseline_subplots.png",
    window: int = 10
):
    """
    Reads 'csv_file' (with columns [Environment, NoiseType, NoiseParams, Episode, AverageReward])
    and creates a single figure containing subplots arranged in 1 row × N columns,
    where N is the number of unique environments.

    Each subplot shows:
      - Raw reward curve
      - Smoothed reward curve (if enough data points for the chosen window)
    
    Saves the resulting figure to 'output_file'.

    :param csv_file: Path to your baseline.csv file.
    :param output_file: File path (including .png) where the final figure will be saved.
    :param window: Smoothing window size for the moving average.
    """
    df = pd.read_csv(csv_file)
    if df.empty:
        print(f"CSV file '{csv_file}' is empty or not found.")
        return

    # Make sure the directory for output_file exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Identify all unique environments
    unique_envs = df["Environment"].unique()
    num_envs = len(unique_envs)

    if num_envs == 0:
        print("No environments found in the CSV.")
        return

    # Create 1×N subplots
    fig, axes = plt.subplots(nrows=1, ncols=num_envs, figsize=(6 * num_envs, 5), sharey=True)

    # If there's only one environment, 'axes' will be a single Axes object, not a list
    if num_envs == 1:
        axes = [axes]

    for i, env_id in enumerate(unique_envs):
        ax = axes[i]
        env_data = df[df["Environment"] == env_id]

        # Extract rewards
        rewards = env_data["AverageReward"].values
        if len(rewards) == 0:
            ax.set_title(env_id)
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            continue

        # Compute smoothed rewards if enough data
        smoothed_rewards = None
        if len(rewards) >= window:
            smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode="valid")

        # Plot raw rewards
        ax.plot(
            rewards,
            label="Raw Rewards",
            linestyle='-',
            linewidth=1
        )

        # Plot smoothed if available
        if smoothed_rewards is not None and len(smoothed_rewards) > 0:
            ax.plot(
                range(len(smoothed_rewards)),
                smoothed_rewards,
                label=f"Smoothed (w={window})",
                linestyle='--',
                linewidth=2
            )

        ax.set_title(env_id)
        ax.set_xlabel("Episodes")
        if i == 0:
            ax.set_ylabel("Total Reward")
        ax.grid(True)
        ax.legend()

    # Adjust spacing
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    print(f"Saved 1×{num_envs} subplot figure to '{output_file}'")
    plt.close(fig)


# Example usage:
if __name__ == "__main__":
    # The default CSV is "results/baseline.csv"
    csv_path = os.path.join("results", "baseline.csv")
    
    # 1) Create separate plots for each environment
    plot_all_environments(csv_path, output_dir="plots", window=10)
    
    # 2) Create a single figure with all environment curves
    plot_environments_side_by_side(csv_file=csv_path, output_file="plots/baseline_subplots.png", window=10)