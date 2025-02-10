#!/usr/bin/env python

"""
evidence_for_markovian.py

A script that:
1. Runs random-policy PCMCI across multiple trials per environment using the FirstOrderMarkovian class,
2. Gets the significant-links text and first-order Markovian score,
3. Logs the mean Markov score across trials,
4. Generates a PDF with environment results side-by-side in two columns.

We assume:
   - A `get_first_order_markovian_score(...)` method is available in PCMCI,
   - A config JSON specifying environments, time_steps, num_trials, etc.
"""

import argparse
import json
import os
import sys
import numpy as np

import gymnasium as gym

# Tigramite
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
# 1) FirstOrderMarkovian Class
###############################################################################

class FirstOrderMarkovian:
    """
    A class to run PCMCI (with ParCorr) on custom observations,
    retrieve a first-order Markovian score, optionally print
    significant links, and generate a PDF report.

    Attributes
    ----------
    alpha : float
        Significance level for PCMCI
    tau_min : int
        Minimum time lag for PCMCI
    tau_max : int
        Maximum time lag for PCMCI
    pc_alpha : float or None
        If not None, alpha for PC steps in PCMCI
    verbose : int
        Verbosity level (0 = silent, 1 = print significant links)
    data : np.ndarray or None
        Observations array (time x variables)
    dataframe : DataFrame or None
        Tigramite DataFrame storing `data`
    results : dict or None
        Results from running PCMCI
    links_text : str
        Captured string of significant links
    markov_score : float
        First-order Markovian score after running PCMCI
    """
    def __init__(self, alpha=0.05, tau_min=1, tau_max=2, pc_alpha=None, verbose=0):
        """
        Parameters
        ----------
        alpha : float
            Significance level used in `print_significant_links` and Markov scoring.
        tau_min : int
            Minimum time lag for PCMCI.
        tau_max : int
            Maximum time lag for PCMCI.
        pc_alpha : float or None
            If set, alpha for PC steps in PCMCI (defines significant links).
        verbose : int
            Verbosity level. If 1, prints significant links to console.
        """
        self.alpha = alpha
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.pc_alpha = pc_alpha
        self.verbose = verbose

        self.data = None
        self.dataframe = None
        self.results = None
        self.links_text = ""
        self.markov_score = 0.0

    def set_observations(self, observations: np.ndarray):
        """
        Sets the custom observations to be analyzed.

        Parameters
        ----------
        observations : np.ndarray
            Array of shape (time, variables).
        """
        self.data = observations
        self.dataframe = DataFrame(data=self.data)
        self.results = None
        self.links_text = ""
        self.markov_score = 0.0

    def run_pcmci(self):
        """
        Run PCMCI on the currently set observations, capture significant links,
        compute and store the first-order Markovian score.

        Returns
        -------
        results : dict
            The dictionary returned by PCMCI's `run_pcmci` method.
        """
        if self.dataframe is None:
            raise ValueError("No observations have been set. Call set_observations first.")

        cond_ind_test = ParCorr()
        pcmci = PCMCI(dataframe=self.dataframe,
                      cond_ind_test=cond_ind_test)

        # Run PCMCI
        self.results = pcmci.run_pcmci(
            tau_min=self.tau_min,
            tau_max=self.tau_max,
            pc_alpha=self.pc_alpha
        )

        # Capture printed output from print_significant_links
        buf = io.StringIO()
        with redirect_stdout(buf):
            pcmci.print_significant_links(
                p_matrix=self.results['p_matrix'],
                val_matrix=self.results['val_matrix'],
                alpha_level=self.alpha
            )
        self.links_text = buf.getvalue()

        # Optionally print significant links to console
        if self.verbose == 1:
            print(self.links_text)

        # Get the Markov score
        self.markov_score = pcmci.get_first_order_markovian_score(
            p_matrix=self.results['p_matrix'],
            val_matrix=self.results['val_matrix'],
            alpha_level=self.alpha
        )
        return self.results

    def get_markovian_score(self) -> float:
        """
        Retrieve the Markovian score from the last PCMCI run.
        """
        return self.markov_score

    @staticmethod
    def generate_pdf_report(results_dict, output_pdf="markovian_evidence_report.pdf"):
        """
        Summarizes final results for each environment in a PDF,
        showing them side-by-side in two columns.

        Parameters
        ----------
        results_dict : dict
            Dictionary of the form:
            {
                env_id: {
                    "num_trials": int,
                    "mean_trial_score": float,
                    "std_trial_score": float,
                    "final_significant_links": str
                },
                ...
            }
        output_pdf : str
            Filename/path to save the resulting PDF.
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

                _draw_environment_info(c, env_id, info, x_start, y_start, col_width)

            i += 2

        c.save()
        print(f"PDF report generated at: {output_pdf}")


###############################################################################
# Internal PDF utility function (used by generate_pdf_report)
###############################################################################
def _draw_environment_info(c, env_id, info, x_start, y_start, col_width):
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
    ]

    for line in details:
        c.drawString(x_start + 10, y_start, line)
        y_start -= line_spacing

    c.drawString(x_start + 10, y_start, "PCMCI Link Output (example):")
    y_start -= line_spacing

    sig_lines = info['final_significant_links'].split("\n")
    for line in sig_lines:
        c.drawString(x_start + 20, y_start, line)
        y_start -= line_spacing

###############################################################################
# 2) Helper function to gather random observations
###############################################################################

def gather_random_observations(env_id, num_steps=1000, seed=None):
    """
    Create a gym environment, gather random observations for num_steps,
    then return the stacked observations as a NumPy array of shape (num_steps, obs_dim).
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

###############################################################################
# 3) Main routine
###############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.json")
    parser.add_argument("--alpha_level", type=float, default=0.05)
    parser.add_argument("--tau_min", type=int, default=1)
    parser.add_argument("--tau_max", type=int, default=2)
    parser.add_argument("--override_num_steps", type=int, default=None)
    parser.add_argument("--output_pdf", type=str, default="markovian_evidence_report.pdf")
    parser.add_argument("--verbose", type=int, default=0,
                        help="Set 1 to print significant links to console.")
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

        # Instantiate FirstOrderMarkovian for this environment
        # (you can also re-instantiate for each trial if you prefer).
        fm = FirstOrderMarkovian(
            alpha=args.alpha_level,
            tau_min=args.tau_min,
            tau_max=args.tau_max,
            pc_alpha=None,
            verbose=args.verbose
        )

        for trial_idx in range(num_trials):
            print(f"  Trial {trial_idx+1} of {num_trials} | {steps_for_this_env} steps (random)")
            observations = gather_random_observations(env_id, num_steps=steps_for_this_env)

            fm.set_observations(observations)
            fm.run_pcmci()

            # Store the score for this single run
            trial_markov_scores.append(fm.get_markovian_score())

        # Compute mean and std of Markov scores across all trials
        trial_markov_scores = np.array(trial_markov_scores)
        mean_score = float(trial_markov_scores.mean())
        std_score  = float(trial_markov_scores.std())

        # The last run's significant links text (or combine them if you prefer)
        last_links_text = fm.links_text

        # Fill data for the PDF
        results_dict[env_id] = {
            "num_trials": num_trials,
            "mean_trial_score": mean_score,
            "std_trial_score": std_score,
            "final_significant_links": last_links_text
        }

    # Finally, generate PDF using the class's static method
    FirstOrderMarkovian.generate_pdf_report(
        results_dict,
        output_pdf=args.output_pdf
    )


if __name__ == "__main__":
    main()