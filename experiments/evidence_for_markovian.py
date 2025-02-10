#!/usr/bin/env python

"""
evidence_for_markovian.py

A minimal script that:
1. Runs random-policy PCMCI across multiple trials per environment,
2. Calls PCMCI's built-in `print_significant_links` to get the significant-links text,
3. Calls PCMCI's `get_first_order_markovian_score` for the Markov score,
4. Averages the per-trial p_matrix & val_matrix across trials if you like (shown below),
   or simply logs the mean Markov score across trials,
5. Finally, generates a PDF with environment results side-by-side in two columns.

We assume you have:
   - A `get_first_order_markovian_score(...)` method in your PCMCI code,
   - The config JSON specifying environments, time_steps, num_trials, etc.
"""

import argparse
import json
import os
import sys
import numpy as np

import gymnasium as gym
from tigramite.data_processing import DataFrame
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

# For capturing stdout from pcmci.print_significant_links
import io
from contextlib import redirect_stdout

# ReportLab for PDF generation
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4


###############################################################################
# 1) Helper Classes/Functions
###############################################################################

class ConditionalIndependenceTest:
    """
    Runs PCMCI with ParCorr, captures PCMCI's standard printed output
    (significant links), and calls `get_first_order_markovian_score`.
    """
    def __init__(self):
        self.data = None
        self.dataframe = None
        self.results = None
        self.links_text = ""   # We'll store the string from PCMCI
        self.markov_score = 0.0

    def set_observations(self, observations: np.ndarray):
        """
        observations shape = (time, variables).
        """
        self.data = observations
        self.dataframe = DataFrame(data=self.data)

    def run_pcmci(self, tau_min=1, tau_max=2, alpha=0.05, pc_alpha=None):
        """
        Run PCMCI, capture the printed significant links to self.links_text,
        and store the Markov score from `get_first_order_markovian_score`.
        """
        cond_ind_test = ParCorr()
        pcmci = PCMCI(dataframe=self.dataframe, cond_ind_test=cond_ind_test)
        self.results = pcmci.run_pcmci(tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha)
        pcmci.print_significant_links(
            p_matrix=self.results['p_matrix'],
            val_matrix=self.results['val_matrix'],
            alpha_level=0.05,
        )

        # 1) Capture the printed significant links as a string
        buf = io.StringIO()
        with redirect_stdout(buf):
            print("Significant links:")
            pcmci.print_significant_links(
                p_matrix=self.results['p_matrix'],
                val_matrix=self.results['val_matrix'],
                alpha_level=alpha
            )
        self.links_text = buf.getvalue()

        # 2) Compute your custom Markov score (assuming you have a method like this)
        score = pcmci.get_first_order_markovian_score(
            p_matrix=self.results['p_matrix'],
            val_matrix=self.results['val_matrix'],
            alpha_level=alpha
        )
        self.markov_score = score

        return self.results


def gather_random_observations(env_id, num_steps=1000, seed=None):
    """
    Create a gym env, gather random observations for num_steps,
    returns shape (num_steps, obs_dim).
    """
    env = gym.make(env_id)
    if seed is not None:
        env.reset(seed=seed)

    obs_list = []
    obs, _ = env.reset()
    for _ in range(num_steps):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        obs_list.append(obs)
        if done or truncated:
            obs, _ = env.reset()

    env.close()
    return np.array(obs_list)


def generate_pdf_report(results_dict, output_pdf="markovian_evidence_report.pdf"):
    """
    Summarizes final results for each environment in a PDF with two columns per page.

    results_dict:
      {
         env_id: {
            "num_trials": int,
            "mean_trial_score": float,
            "std_trial_score": float,
            "final_significant_links": str,
         },
         ...
      }
    """
    c = canvas.Canvas(output_pdf, pagesize=A4)
    page_width, page_height = A4

    top_margin = 50
    left_margin = 40
    right_margin = 40
    bottom_margin = 50

    usable_width = page_width - left_margin - right_margin
    col_width = usable_width / 2
    current_y = page_height - top_margin - 30

    c.setFont("Helvetica-Bold", 14)
    c.drawString(left_margin, page_height - top_margin,
                 "Evidence for Markovian Experiments Report")

    env_ids = list(results_dict.keys())
    total_envs = len(env_ids)
    i = 0
    while i < total_envs:
        if i > 0:
            c.showPage()
            c.setFont("Helvetica-Bold", 14)
            c.drawString(left_margin, page_height - top_margin,
                         "Evidence for Markovian Experiments Report")
            current_y = page_height - top_margin - 30

        for col_idx in range(2):
            env_index = i + col_idx
            if env_index >= total_envs:
                break

            env_id = env_ids[env_index]
            info = results_dict[env_id]
            x_start = left_margin + col_idx * col_width
            y_start = current_y
            draw_environment_info(c, env_id, info, x_start, y_start, col_width)

        i += 2

    c.save()
    print(f"PDF report generated at: {output_pdf}")


def draw_environment_info(c, env_id, info, x_start, y_start, col_width):
    """
    Helper function for the PDF columns.
    """
    font_size_title = 10
    font_size_text = 7
    line_spacing = 10

    c.setFont("Helvetica-Bold", font_size_title)
    c.drawString(x_start, y_start, f"Environment: {env_id}")
    y_start -= (line_spacing + 2)

    c.setFont("Helvetica", font_size_text)
    details = [
        f"Number of trials: {info['num_trials']}",
        f"Mean trial Markov Score: {info['mean_trial_score']:.4f}",
        f"Std  Markov Score:      {info['std_trial_score']:.4f}"
        # 'final_significant_links': see below
    ]
    # If you want a "final" Markov score from e.g. an averaged matrix approach, you can add it:
    # details.append(f"Final Markov Score: {info['final_markov_score']:.4f}")

    for line in details:
        c.drawString(x_start + 10, y_start, line)
        y_start -= line_spacing

    # Now print the captured significant links
    c.drawString(x_start + 10, y_start, "PCMCI Link Output (example):")
    y_start -= line_spacing

    sig_lines = info['final_significant_links'].split("\n")
    for line in sig_lines:
        c.drawString(x_start + 20, y_start, line)
        y_start -= line_spacing


###############################################################################
# 2) Main routine
###############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.json")
    parser.add_argument("--alpha_level", type=float, default=0.05)
    parser.add_argument("--tau_min", type=int, default=1)
    parser.add_argument("--tau_max", type=int, default=2)
    parser.add_argument("--override_num_steps", type=int, default=None)
    parser.add_argument("--output_pdf", type=str, default="markovian_evidence_report.pdf")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config file '{args.config}' not found.")
        sys.exit(1)

    with open(args.config, 'r') as f:
        config = json.load(f)

    env_specs = config.get("environments", {})
    if not env_specs:
        print("No environments found in config. Exiting.")
        sys.exit(0)

    results_dict = {}
    for env_id, env_info in env_specs.items():
        print(f"\n--- Environment: {env_id} ---")
        num_trials = env_info.get("num_trials", 1)
        steps_for_this_env = env_info.get("time_steps", 1000)
        if args.override_num_steps is not None:
            steps_for_this_env = args.override_num_steps

        # We store per-trial scores
        trial_markov_scores = []

        for trial_idx in range(num_trials):
            print(f"  Trial {trial_idx+1} of {num_trials} | {steps_for_this_env} steps (random)")
            observations = gather_random_observations(env_id, num_steps=steps_for_this_env)

            ci_test = ConditionalIndependenceTest()
            ci_test.set_observations(observations)
            results = ci_test.run_pcmci(
                tau_min=args.tau_min,
                tau_max=args.tau_max,
                alpha=args.alpha_level,
                pc_alpha=None
            )
            # store the score from this single run
            trial_markov_scores.append(ci_test.markov_score)

        # If you want, you can do an average or final approach:
        trial_markov_scores = np.array(trial_markov_scores)
        mean_score = float(trial_markov_scores.mean())
        std_score  = float(trial_markov_scores.std())

        # For demonstration, we just store the last run's "links_text" 
        # or combine them if you prefer. We'll just use the last one for the PDF.
        last_links_text = ci_test.links_text

        # Fill in data for the PDF
        results_dict[env_id] = {
            "num_trials": num_trials,
            "mean_trial_score": mean_score,
            "std_trial_score": std_score,
            "final_significant_links": last_links_text
        }

    # Finally, generate PDF
    generate_pdf_report(results_dict, output_pdf=args.output_pdf)


if __name__ == "__main__":
    main()