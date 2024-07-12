import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib.figure
from typing import Any, Optional, Tuple

from utils.performance_analysis import \
    get_perf_scores_df

# Global Variables set up #############################################################################################
FIG_SIZE = (16, 10)
PLOT_SCORE = 'OAC_t0'           # 'OAC_t0', 'OAO'
AS_PERCENTAGE = False


# Functions ###########################################################################################################

def oac_histogram(
        draw_ax: matplotlib.axes.Axes,
        experiment_name: str,
        plot_score: str,
        categorization: Tuple[str, Any],
        as_percentage: Optional[bool] = False,
        n_bins: int = 20
):
    # categorization [0, 1] <--> [column in experiment_df, value by which to filter that column]
    experiment_df = get_perf_scores_df(experiment_name)
    mini_df = experiment_df[
        (experiment_df[categorization[0]] == categorization[1]) & (pd.notna(experiment_df[plot_score]))
        ]

    scores = mini_df[plot_score].to_numpy()
    weights = np.ones_like(scores) / (scores.shape[0]) if as_percentage else None

    draw_ax.hist(scores, bins=n_bins, weights=weights)

    # draw_ax.set_xlabel(f"{plot_score}", loc='left')
    # draw_ax.set_title(f"{plot_score} histogram: {experiment_name} (last observed timestep: {-categorization[1]})")


def oac_histograms_versus_lastobs(
        draw_ax: matplotlib.axes.Axes,
        experiment_name: str,
        plot_score: str,
):
    experiment_df = get_perf_scores_df(experiment_name)

    mini_df = experiment_df[pd.notna(experiment_df[plot_score])]

    scores1 = mini_df[plot_score].to_numpy()
    past_pred_lengths = mini_df['past_pred_length'].to_numpy()
    unique, counts = np.unique(past_pred_lengths, return_counts=True)
    weights = 1 / counts[past_pred_lengths - 1]
    # weights = 1 / (counts[occl_lengths-1] * unique.shape[0])

    hist = draw_ax.hist2d(
        scores1, past_pred_lengths,
        bins=[np.linspace(0, 1, 21), np.arange(7) + 0.5],
        weights=weights
    )

    draw_ax.get_figure().colorbar(hist[3], ax=draw_ax)
    draw_ax.set_xlabel(plot_score)
    draw_ax.set_ylabel("last observed timestep")
    # draw_ax.set_title(f"OAC histogram: {experiment_name}")


def make_oac_histograms_figure(
        fig: matplotlib.figure.Figure,
        experiment_name: str,
        plot_score: str,
        as_percentage: bool = False
):
    gs = fig.add_gridspec(6, 2)

    ax_list = []
    for i in range(6):
        ax_list.append(fig.add_subplot(gs[i, 0]))
    ax_twodee = fig.add_subplot(gs[:, 1])

    for i, ax in enumerate(reversed(ax_list)):
        oac_histogram(draw_ax=ax, experiment_name=experiment_name, plot_score=plot_score,
                      categorization=('past_pred_length', i + 1), as_percentage=as_percentage)
    oac_histograms_versus_lastobs(draw_ax=ax_twodee, experiment_name=experiment_name, plot_score=plot_score)
    ax_list[-1].set_xlabel(plot_score)
    fig.suptitle(f"{plot_score} histograms: {experiment_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', nargs='+', default=None)
    args = parser.parse_args()

    print(f"{PLOT_SCORE} HISTOGRAMS:\n\n")

    experiment_names = args.cfg

    for experiment_name in experiment_names:
        fig = plt.figure(figsize=FIG_SIZE)
        make_oac_histograms_figure(
            fig=fig, experiment_name=experiment_name, plot_score=PLOT_SCORE, as_percentage=AS_PERCENTAGE
        )

    plt.show()

    # if SAVE:
    #     print("OAC_histograms: no saving implementation (yet)!")
