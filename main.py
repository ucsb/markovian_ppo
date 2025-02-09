import argparse
import json
import os
import pathlib
from datetime import datetime

import experiments.baseline as baseline
import experiments.noised_observations as noised_observations
def setup():
    """
    - Creates a 'backup' folder if it doesn't exist.
    - Moves old files/directories (logs.txt, plots, results) into 'backup'
      with a timestamp to avoid overwriting.
    - Recreates clean 'plots' and 'results' folders for fresh output.
    """
    os.makedirs("backup", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if os.path.exists("logs.txt"):
        os.rename("logs.txt", f"backup/logs_{timestamp}.txt")

    if os.path.exists("plots"):
        os.rename("plots", f"backup/plots_{timestamp}")

    if os.path.exists("results"):
        os.rename("results", f"backup/results_{timestamp}")

    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

def main(config_path, results_path):
    """
    Main entry point:
      1) Prepare folders/logs by calling setup()
      2) Read config from config_path
      3) Run baseline experiments, saving results to results_path/baseline.csv
    """
    setup()

    root_path = pathlib.Path().absolute()
    config_path = os.path.join(root_path, config_path)
    results_path = os.path.join(root_path, results_path)

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Run baseline experiments
    #baseline.run(config, results_path)
    # Run noised observations experiments
    noised_observations.run(config, results_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to config.json")
    parser.add_argument("--results_path", type=str, default="results", 
                        help="Directory for storing results (baseline.csv)")

    args = parser.parse_args()
    main(args.config_path, args.results_path)