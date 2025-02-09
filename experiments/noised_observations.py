"""
File: experiments/noised_observations.py

This file mirrors baseline.py but applies noise to the observations.
It uses the new RPPO public methods:
- set_is_noise
- set_noise_strategy
- set_noise_to_observation_id

and (if desired) set_noise_level (for Gaussian) or similar methods for other noise types.
"""

import logging
import os
import time

import numpy as np
import pandas as pd
import setproctitle
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

# Import your custom RPPO that has noise support
from experiments.rppo import RPPO

setproctitle.setproctitle("experiment_compare_noise_envs_noised")


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


def train_with_noise(env_id, total_timesteps, n_envs, noise_strategy, noise_params, obs_dim=0):
    """
    Train the RPPO model on a specified environment with a given noise strategy/params
    applied only to one observation dimension (obs_dim).
    
    :param env_id: The Gym environment ID (string).
    :param total_timesteps: Number of timesteps to train on.
    :param n_envs: Number of parallel environments (int).
    :param noise_strategy: The noise strategy name (e.g., "gaussian", "uniform", "salt_and_pepper").
    :param noise_params: A dictionary of the noise parameters (e.g., {"mean": 0, "std": 0.01}).
    :param obs_dim: Observation dimension to which noise will be applied (default=0).
    :return: List of per-episode rewards collected.
    """
    logging.info(
        f"Training on {env_id} with noise_strategy={noise_strategy}, noise_params={noise_params}, obs_dim={obs_dim}"
    )

    # Create vectorized environment
    vec_env = make_vec_env(env_id, n_envs=n_envs)

    # Initialize RPPO
    model = RPPO("MlpPolicy", vec_env, verbose=1, learning_rate=3e-4)

    # Enable and configure noise
    model.set_is_noise(True)
    model.set_noise_strategy(noise_strategy)

    if noise_strategy.lower() == "gaussian":
        # Typically we set noise as (mean, variance). 
        # The user-provided config is (mean=..., std=...), so we must square the std to get variance.
        mean = noise_params["mean"]
        std_dev = noise_params["std"]
        variance = std_dev ** 2
        model.set_noise_level(mean, variance)

    elif noise_strategy.lower() == "uniform":
        # If your RPPO does not implement uniform, either skip or raise error
        # For demonstration, we'll just set noise_level to something symbolic.
        # In practice, your RPPO would need logic to handle uniform noise.
        low = noise_params["low"]
        high = noise_params["high"]
        # model.set_noise_level_for_uniform(low, high)  # You'd need a method like this in your RPPO
        pass

    elif noise_strategy.lower() == "salt_and_pepper":
        # Similarly, if your RPPO does not implement salt & pepper, skip or raise error
        prob = noise_params["prob"]
        # model.set_noise_level_for_salt_and_pepper(prob) # Hypothetical method in your RPPO
        pass

    else:
        raise NotImplementedError(
            f"Noise strategy '{noise_strategy}' is not recognized or supported in RPPO."
        )

    # Optionally apply noise to a specific observation dimension
    model.set_noise_to_observation_id(obs_dim)

    # Train
    reward_callback = RewardTrackingCallback()
    model.learn(total_timesteps=total_timesteps, callback=reward_callback)

    return reward_callback.get_rewards()


def run_experiments_and_save_results(config, results_path):
    """
    Runs experiments for each environment in 'config' but with noise:
    - Iterates over 'noise_strategies' from config, applying each strategy with each set of params.
    - Aggregates and saves results in noised_observations.csv.
    """
    output_file = os.path.join(results_path, "noised_observations.csv")
    results = []

    # Get environment specs
    env_specs = config.get("environments", {})
    # Get noise strategies
    noise_strategies = config.get("noise_strategies", {})

    for env_id, env_info in env_specs.items():
        total_timesteps = env_info["time_steps"]
        num_trials = env_info["num_trials"]
        n_envs = env_info["n_envs"]

        logging.info(f"\n=== Noise Experiments for {env_id} ===")

        # Loop over each noise strategy and param set
        for strategy_name, strategy_params_list in noise_strategies.items():
            for params in strategy_params_list:
                logging.info(f"Noise strategy: {strategy_name}, params={params}")

                # Here, we apply noise to only one dimension, e.g., obs_dim=0.
                # You can loop over each dimension if you prefer.
                # e.g., for obs_dim in range(num_obs_dims):
                obs_dim = 0

                all_rewards_noised = []

                for trial_i in range(num_trials):
                    logging.info(f"  Trial {trial_i + 1} of {num_trials}")
                    start_time = time.perf_counter()
                    rewards_per_episode = train_with_noise(
                        env_id,
                        total_timesteps,
                        n_envs=n_envs,
                        noise_strategy=strategy_name,
                        noise_params=params,
                        obs_dim=obs_dim
                    )
                    end_time = time.perf_counter()
                    logging.info(f"  Time taken: {end_time - start_time:.2f} seconds")
                    all_rewards_noised.append(rewards_per_episode)

                # Align results across trials (just like baseline.py)
                max_length = max(len(r) for r in all_rewards_noised)
                aligned_noised = np.zeros((num_trials, max_length))

                for i, rew in enumerate(all_rewards_noised):
                    rew = np.array(rew)
                    aligned_noised[i, :len(rew)] = rew

                averaged_noised = np.mean(aligned_noised, axis=0)

                # Store results
                for episode_idx, reward_val in enumerate(averaged_noised):
                    results.append({
                        "Environment": env_id,
                        "NoiseType": strategy_name,
                        "NoiseParams": str(params),
                        "ObsDim": obs_dim,
                        "Episode": episode_idx,
                        "AverageReward": reward_val
                    })

    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    logging.info(f"Noise results saved to {output_file}")


def run(config, results_path):
    """
    A high-level wrapper to orchestrate noised observation runs.
    """
    start_time = time.perf_counter()
    run_experiments_and_save_results(config, results_path)
    end_time = time.perf_counter()
    logging.info(f"Total time taken for noised experiments: {end_time - start_time:.2f} seconds")