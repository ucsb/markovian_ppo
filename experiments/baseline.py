import logging
import os
import time

import numpy as np
import pandas as pd
import setproctitle
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from experiments.rppo import RPPO  # Make sure RPPO is in experiments/rppo.py

setproctitle.setproctitle('experiment_compare_noise_envs')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs.txt"),
        logging.StreamHandler()
    ]
)

class RewardTrackingCallback(BaseCallback):
    """
    Records total reward per episode in a list for retrieval after training.
    """
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_episode_reward = 0.0

    def _on_step(self) -> bool:
        # Using index [0] for simplicity in single-env cases.
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]
        self.current_episode_reward += reward

        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0

        return True

    def get_rewards(self):
        return self.episode_rewards

def train(env_id, total_timesteps, n_envs=4):
    """
    Train the RPPO model on a specified environment without noise.
    """
    logging.info(f"Training on {env_id} (timesteps={total_timesteps}, n_envs={n_envs})")

    # Create vectorized environment
    vec_env = make_vec_env(env_id, n_envs=n_envs)

    # Initialize RPPO with no observation noise
    model = RPPO("MlpPolicy", vec_env, verbose=1, noise=None, learning_rate=3e-4)

    reward_callback = RewardTrackingCallback()
    model.learn(total_timesteps=total_timesteps, callback=reward_callback)
    return reward_callback.get_rewards()

def run_experiments_and_save_results(config, results_path):
    """
    Runs baseline (no noise) experiments for each environment specified in 'config',
    and stores the aggregated results in baseline.csv within 'results_path'.
    """
    output_file = os.path.join(results_path, "baseline.csv")
    results = []

    # Read environment specs from config
    env_specs = config.get("environments", {})

    for env_id, env_info in env_specs.items():
        total_timesteps = env_info["time_steps"]
        num_trials = env_info["num_trials"]
        n_envs = env_info["n_envs"]

        logging.info(f"\n=== Baseline Run for {env_id} ===")
        all_rewards_baseline = []

        for trial_i in range(num_trials):
            logging.info(f"  Trial {trial_i + 1} of {num_trials}")
            start_time = time.perf_counter()
            rewards_per_episode = train(env_id, total_timesteps, n_envs=n_envs)
            end_time = time.perf_counter()
            logging.info(f"  Time taken: {end_time - start_time:.2f} seconds")
            all_rewards_baseline.append(rewards_per_episode)

        # Align results across trials
        max_length = max(len(r) for r in all_rewards_baseline)
        aligned_baseline = np.zeros((num_trials, max_length))

        for i, rew in enumerate(all_rewards_baseline):
            rew = np.array(rew)
            aligned_baseline[i, :len(rew)] = rew

        averaged_baseline = np.mean(aligned_baseline, axis=0)

        # Store results in a list
        for episode_idx, reward_val in enumerate(averaged_baseline):
            results.append({
                "Environment": env_id,
                "NoiseType": "baseline",
                "NoiseParams": "None",
                "Episode": episode_idx,
                "AverageReward": reward_val
            })

    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    logging.info(f"Results saved to {output_file}")

def run(config, results_path):
    """
    A high-level wrapper to orchestrate baseline runs.
    """
    start_time = time.perf_counter()
    run_experiments_and_save_results(config, results_path)
    end_time = time.perf_counter()
    logging.info(f"Total time taken: {end_time - start_time:.2f} seconds")