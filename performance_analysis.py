import argparse
import os.path
import pickle

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import yaml
import pandas as pd
import torch

from typing import Dict, List, Optional, Tuple, Any

from data.sdd_dataloader import PresavedDatasetSDD
from utils.utils import prepare_seed
from utils.config import Config, REPO_ROOT
from data.sdd_dataloader import HDF5DatasetSDD
from utils.sdd_visualize import visualize_input_and_predictions, write_scores_per_mode
from utils.performance_metrics import compute_samples_ADE, compute_samples_FDE
from model.agentformer_loss import index_mapping_gt_seq_pred_seq


def get_perf_scores_df(experiment_name: str, model_name: Optional[str] = None) -> pd.DataFrame:

    target_path = os.path.join(REPO_ROOT, 'results', experiment_name, 'results')

    dataset_used = os.listdir(target_path)[0]
    target_path = os.path.join(target_path, dataset_used)
    assert os.path.exists(target_path)

    if model_name is None:
        model_name = os.listdir(target_path)[0]
    target_path = os.path.join(target_path, model_name, 'test', 'prediction_scores.csv')
    assert os.path.exists(target_path)

    df = pd.read_csv(target_path)

    df_indices = ['idx', 'filename', 'agent_id']
    df.set_index(keys=df_indices, inplace=True)

    return df


def get_perf_scores_dict(experiment_name: str, model_name: Optional[str] = None) -> Dict:

    target_path = os.path.join(REPO_ROOT, 'results', experiment_name, 'results')

    dataset_used = os.listdir(target_path)[0]
    target_path = os.path.join(target_path, dataset_used)
    assert os.path.exists(target_path)

    if model_name is None:
        model_name = os.listdir(target_path)[0]
    target_path = os.path.join(target_path, model_name, 'test', 'prediction_scores.yml')
    assert os.path.exists(target_path)

    with open(target_path, 'r') as f:
        scores_dict = yaml.safe_load(f)

    return scores_dict


def print_occlusion_length_counts():
    dataframe = get_perf_scores_df('occlusionformer_no_map')
    print("last observed timestep\t| case count")
    for i in range(0, 7):
        mini_df = dataframe[dataframe['past_pred_length'] == i]
        print(f"{-i}\t\t\t| {len(mini_df)}")
    print(f"total\t\t\t| {len(dataframe)}")


def generate_performance_summary_df(experiment_names: List, metric_names: List) -> pd.DataFrame:
    df_columns = ['experiment', 'dataset_used', 'model_name'] + metric_names
    performance_df = pd.DataFrame(columns=df_columns)

    for experiment_name in experiment_names:
        run_results_path = os.path.join(REPO_ROOT, 'results', experiment_name, 'results')
        dataset_used = os.listdir(run_results_path)[0]
        model_name = os.listdir(os.path.join(run_results_path, dataset_used))[0]

        scores_dict = get_perf_scores_dict(experiment_name=experiment_name)
        scores_dict['experiment'] = experiment_name
        scores_dict['dataset_used'] = dataset_used
        scores_dict['model_name'] = model_name

        performance_df.loc[len(performance_df)] = scores_dict

    return performance_df


def remove_k_sample_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_columns = [name for name in df.columns.tolist() if not name.startswith('K')]
    return df[keep_columns]


def difference_summary_df(
        summary_df: pd.DataFrame, base_experiment_name: str, mode: str = 'absolute'
) -> pd.DataFrame:
    assert mode in ['absolute', 'relative']
    assert base_experiment_name in summary_df['experiment'].to_list()
    assert (summary_df['experiment'] == base_experiment_name).sum() == 1

    diff_df = summary_df.copy()

    compare_columns = summary_df.columns[summary_df.dtypes == float]
    compare_row = diff_df.loc[diff_df['experiment'] == base_experiment_name, compare_columns].iloc[0]

    if mode == 'absolute':
        diff_df.loc[:, compare_columns] = diff_df.loc[:, compare_columns].subtract(compare_row)
    elif mode == 'relative':
        diff_df.loc[:, compare_columns] = diff_df.loc[:, compare_columns].div(compare_row)
        diff_df.loc[:, compare_columns] = diff_df.loc[:, compare_columns] - 1.0

    return diff_df


def pretty_print_difference_summary_df(
        summary_df: pd.DataFrame, base_experiment_name: str, mode: str = 'absolute'
) -> pd.DataFrame:
    diff_df = difference_summary_df(summary_df=summary_df, base_experiment_name=base_experiment_name, mode=mode)
    compare_columns = summary_df.columns[summary_df.dtypes == float]

    if mode == 'absolute':
        diff_df.loc[:, compare_columns] = diff_df.loc[:, compare_columns].applymap(
            lambda x: f" (+{x:.3f})" if x >= 0. else f" ({x:.3f})"
        )
    if mode == 'relative':
        diff_df.loc[:, compare_columns] = diff_df.loc[:, compare_columns] * 100
        diff_df.loc[:, compare_columns] = diff_df.loc[:, compare_columns].applymap(
            lambda x: f" (+{x:.3f}%)" if x >= 0. else f" ({x:.3f}%)"
        )

    out_df = summary_df.copy()
    out_df.loc[:, compare_columns] = out_df.loc[:, compare_columns].round(5).astype(str)
    out_df.loc[:, compare_columns] = out_df.loc[:, compare_columns] + diff_df.loc[:, compare_columns]

    return out_df


# def top_score_dataframe(df: pd.DataFrame, column_name: str, ascending: bool = True, n: int = 5) -> pd.DataFrame:
#     return df.sort_values(column_name, ascending=ascending).head(n)


def reduce_by_unique_column_values(
        df: pd.DataFrame, column_name: str, operation: str = 'mean'
) -> pd.DataFrame:
    assert column_name in df.columns
    assert operation in ['mean', 'std', 'median']

    scores = df.columns.tolist()
    scores.remove(column_name)
    summary_df = pd.DataFrame(columns=['count', column_name] + scores)

    operation = eval(f"pd.DataFrame.{operation}")
    for col_value in sorted(df[column_name].unique()):

        mini_df = df[df[column_name] == col_value]

        row_dict = operation(mini_df[scores]).to_dict()
        row_dict['count'] = len(mini_df)
        row_dict[column_name] = col_value

        summary_df.loc[len(summary_df)] = row_dict
    return summary_df


def draw_boxplots(
        draw_ax: matplotlib.axes.Axes, df: pd.DataFrame, column_data: str, column_boxes: str
) -> None:
    assert column_data in df.columns
    assert column_boxes in df.columns

    box_names = sorted(df[column_boxes].unique())
    box_data = []

    for box_category in box_names:

        mini_df = df[(df[column_boxes] == box_category) & (pd.notna(df[column_boxes]))]

        scores = mini_df[column_data].to_numpy()
        box_data.append(scores)

    draw_ax.boxplot(box_data)
    draw_ax.set_xticklabels(box_names)


def performance_dataframes_comparison(
        base_df: pd.DataFrame, compare_df: pd.DataFrame
) -> pd.DataFrame:
    # removing rows that are not shared across base_df and comp_df
    keep_indices = base_df.index.intersection(compare_df.index)
    out_df = compare_df.sub(base_df)

    return out_df.loc[keep_indices, :]


def scatter_perf_gain_vs_perf_base(
        base_df: pd.DataFrame, comp_df: pd.DataFrame, col_name: str, relative: bool = True
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    # maybe get rid of this function altogether...
    assert col_name in base_df.columns
    assert col_name in comp_df.columns

    keep_indices = base_df.index.intersection(comp_df.index)

    fig, ax = plt.subplots()
    diff = comp_df.loc[keep_indices, col_name].sub(base_df.loc[keep_indices, col_name])

    if relative:
        diff = diff.div(base_df.loc[keep_indices, col_name])

    ax.scatter(base_df.loc[keep_indices, col_name], diff, marker='x', alpha=0.5)

    return fig, ax


def scatter_performance_scores(
        draw_ax: matplotlib.axes.Axes,
        x_df: pd.DataFrame, x_score_name: str,
        y_df: pd.DataFrame, y_score_name: str,
        xlabel_prefix: Optional[str] = None,
        ylabel_prefix: Optional[str] = None
) -> None:
    assert x_score_name in x_df.columns
    assert y_score_name in y_df.columns

    keep_indices = x_df.index.intersection(y_df.index)
    xs = x_df.loc[keep_indices, x_score_name].to_numpy()
    ys = y_df.loc[keep_indices, y_score_name].to_numpy()

    draw_ax.scatter(xs, ys, marker='x', color='black', alpha=0.7)

    x_label = f"{xlabel_prefix} | {x_score_name}" if xlabel_prefix is not None else x_score_name
    y_label = f"{ylabel_prefix} | {y_score_name}" if ylabel_prefix is not None else y_score_name

    draw_ax.set_xlabel(x_label, loc='left')
    draw_ax.set_ylabel(y_label, loc='bottom')


def get_occluded_identities(df: pd.DataFrame, idx: int):
    print(df.loc[idx].index.get_level_values('agent_id').to_list())
    print(df.loc[idx])
    instance_df = df.loc[idx]
    if (instance_df['past_pred_length'] != 0).sum() != 0:   # if there are agents who we do have to predict over the past
        return instance_df[instance_df['past_pred_length'] != 0].index.get_level_values('agent_id').to_list()
        # return df[df['past_pred_length'] != 0].loc[idx].index.get_level_values('agent_id').to_list()
    else:
        return []


def get_comparable_rows(
        base_df: pd.DataFrame,
        compare_df: pd.DataFrame,
        n_max_agents: Optional[int] = None
):
    # identifying the rows the two dataframes have in common
    common_rows = base_df.index.intersection(compare_df.index)
    compare_rows = common_rows.copy()
    if n_max_agents is not None:
        # defining a filter to remove instances with agent count larger than <n_max_agents>
        row_indices_leq_n_max_agents = (common_rows.get_level_values('idx').value_counts() <= n_max_agents)
        row_indices_leq_n_max_agents = row_indices_leq_n_max_agents[row_indices_leq_n_max_agents].index
        # the multiindex of rows to use for comparison
        compare_rows = common_rows[common_rows.get_level_values('idx').isin(row_indices_leq_n_max_agents)]
    return compare_rows


def make_box_plot_occlusion_lengths(
        draw_ax: matplotlib.axes.Axes,
        experiments: List[str],
        plot_score: str,
        categorization: Tuple[str, List[int]] = ('past_pred_length', range(1, 7)),
) -> None:
    category_name, category_values = categorization
    colors = [plt.cm.Pastel1(i) for i in range(len(experiments))]

    box_plot_dict = {experiment_name: None for experiment_name in experiments}
    for i, experiment in enumerate(experiments):

        experiment_df = get_perf_scores_df(experiment_name=experiment)
        experiment_df = remove_k_sample_columns(df=experiment_df)

        assert plot_score in experiment_df.columns
        assert category_name in experiment_df.columns

        pred_lengths = sorted(experiment_df[category_name].unique())

        experiment_data_dict = {int(pred_length): None for pred_length in pred_lengths}
        for pred_length in pred_lengths:
            mini_df = experiment_df[
                (experiment_df[category_name] == pred_length) & (pd.notna(experiment_df[plot_score]))
            ]

            scores = mini_df[plot_score].to_numpy()
            experiment_data_dict[int(pred_length)] = scores

        box_plot_dict[experiment] = experiment_data_dict

    box_plot_xs = []
    box_plot_ys = []
    box_plot_colors = []
    for length in category_values:
        for i, experiment in enumerate(experiments):
            box_plot_xs.append(f"{length} - {experiment}")
            box_plot_ys.append(box_plot_dict[experiment][length])
            box_plot_colors.append(colors[i])

    bplot = draw_ax.boxplot(box_plot_ys, positions=range(len(box_plot_ys)), patch_artist=True)
    for box_patch, median_line, color in zip(bplot['boxes'], bplot['medians'], box_plot_colors):
        box_patch.set_facecolor(color)
        median_line.set_color('red')

    x_tick_gap = len(experiments)
    x_tick_start = (len(experiments) - 1) / 2
    x_tick_end = x_tick_start + x_tick_gap * len(category_values)
    draw_ax.set_xticks(np.arange(x_tick_start, x_tick_end, x_tick_gap), labels=-np.array(category_values))
    draw_ax.set_xticks(np.arange(x_tick_start, x_tick_end, x_tick_gap / 2), minor=True)
    draw_ax.grid(which='minor', axis='x')

    draw_ax.legend([bplot["boxes"][i] for i in range(len(experiments))], experiments, loc='upper left')
    draw_ax.set_ylabel(f'{plot_score}', loc='bottom')
    draw_ax.set_xlabel('last observation timestep', loc='left')

    # draw_ax.set_title(f"{plot_score} vs. last observed timestep")


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
        plot_score: str
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
    ax_list[-1].set_xlabel('OAC')
    fig.suptitle(f"OAC histograms: {experiment_name}")


if __name__ == '__main__':
    # Script Controls #################################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--perf_summary', action='store_true', default=False)
    parser.add_argument('--boxplots', action='store_true', default=False)
    parser.add_argument('--oac_histograms', action='store_true', default=False)
    parser.add_argument('--qual_compare', action='store_true', default=False)
    parser.add_argument('--qual_example', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--show', action='store_true', default=False)
    args = parser.parse_args()

    SHOW = args.show
    SAVE = args.save

    assert SAVE or SHOW

    # Global Variables set up #########################################################################################
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 200)

    PERFORMANCE_ANALYSIS_DIRECTORY = os.path.join(REPO_ROOT, 'results', 'performance_analysis')
    os.makedirs(PERFORMANCE_ANALYSIS_DIRECTORY, exist_ok=True)

    MEASURE = 'm'       # 'm' | 'px'
    EXPERIMENT_SEPARATOR = "\n\n\n\n" + "#" * 200 + "\n\n\n\n"

    SDD_BASELINE_OCCLUSIONFORMER = 'sdd_baseline_occlusionformer'
    BASELINE_NO_POS_CONCAT = 'baseline_no_pos_concat'
    OCCLUSIONFORMER_NO_MAP = 'occlusionformer_no_map'
    OCCLUSIONFORMER_CAUSAL_ATTENTION = 'occlusionformer_causal_attention'
    OCCLUSIONFORMER_IMPUTED = 'occlusionformer_imputed'
    OCCLUSIONFORMER_WITH_OCCL_MAP = 'occlusionformer_with_occl_map'
    OCCLUSIONFORMER_WITH_OCCL_MAP_IMPUTED = 'occlusionformer_with_occl_map_imputed'
    ORIGINAL_AGENTFORMER = 'original_agentformer'
    OCCLUSIONFORMER_MOMENTARY = 'occlusionformer_momentary'
    OCCLUSIONFORMER_WITH_BOTH_MAPS = 'occlusionformer_with_both_maps'
    CONST_VEL_FULLY_OBSERVED = 'const_vel_fully_observed'
    CONST_VEL_FULLY_OBSERVED_MOMENTARY_2 = 'const_vel_fully_observed_momentary_2'
    CONST_VEL_OCCLUSION_SIMULATION = 'const_vel_occlusion_simulation'
    CONST_VEL_OCCLUSION_SIMULATION_IMPUTED = 'const_vel_occlusion_simulation_imputed'
    OCCLUSIONFORMER_CAUSAL_ATTENTION_FULLY_OBSERVED = 'occlusionformer_causal_attention_fully_observed'
    OCCLUSIONFORMER_CAUSAL_ATTENTION_IMPUTED = 'occlusionformer_causal_attention_imputed'
    OCCLUSIONFORMER_CAUSAL_ATTENTION_OCCL_MAP = 'occlusionformer_causal_attention_occl_map'
    OCCLUSIONFORMER_OFFSET_TIMECODES = 'occlusionformer_offset_timecodes'
    OCCLUSIONFORMER_IMPUTED_WITH_MARKERS = 'occlusionformer_imputed_with_markers'

    EXPERIMENTS = [
        SDD_BASELINE_OCCLUSIONFORMER,
        BASELINE_NO_POS_CONCAT,
        OCCLUSIONFORMER_NO_MAP,
        OCCLUSIONFORMER_CAUSAL_ATTENTION,
        OCCLUSIONFORMER_IMPUTED,
        OCCLUSIONFORMER_WITH_OCCL_MAP,
        OCCLUSIONFORMER_WITH_OCCL_MAP_IMPUTED,
        ORIGINAL_AGENTFORMER,
        OCCLUSIONFORMER_MOMENTARY,
        # OCCLUSIONFORMER_WITH_BOTH_MAPS,
        CONST_VEL_FULLY_OBSERVED,
        CONST_VEL_FULLY_OBSERVED_MOMENTARY_2,
        CONST_VEL_OCCLUSION_SIMULATION,
        CONST_VEL_OCCLUSION_SIMULATION_IMPUTED,
        # OCCLUSIONFORMER_CAUSAL_ATTENTION_FULLY_OBSERVED,
        # OCCLUSIONFORMER_CAUSAL_ATTENTION_IMPUTED,
        # OCCLUSIONFORMER_CAUSAL_ATTENTION_OCCL_MAP,
        # OCCLUSIONFORMER_OFFSET_TIMECODES,
        # OCCLUSIONFORMER_IMPUTED_WITH_MARKERS
    ]

    FULLY_OBSERVED_EXPERIMENTS = [
        SDD_BASELINE_OCCLUSIONFORMER,
        BASELINE_NO_POS_CONCAT,
        ORIGINAL_AGENTFORMER,
        # OCCLUSIONFORMER_CAUSAL_ATTENTION_FULLY_OBSERVED,
        CONST_VEL_FULLY_OBSERVED
    ]
    OCCLUSION_EXPERIMENTS = [
        OCCLUSIONFORMER_NO_MAP,
        OCCLUSIONFORMER_CAUSAL_ATTENTION,
        OCCLUSIONFORMER_WITH_OCCL_MAP,
        # OCCLUSIONFORMER_CAUSAL_ATTENTION_OCCL_MAP,
        # OCCLUSIONFORMER_OFFSET_TIMECODES,
        CONST_VEL_OCCLUSION_SIMULATION
    ]
    IMPUTED_EXPERIMENTS = [
        OCCLUSIONFORMER_IMPUTED,
        OCCLUSIONFORMER_WITH_OCCL_MAP_IMPUTED,
        # OCCLUSIONFORMER_CAUSAL_ATTENTION_IMPUTED,
        # OCCLUSIONFORMER_IMPUTED_WITH_MARKERS,
        CONST_VEL_OCCLUSION_SIMULATION_IMPUTED
    ]

    ADE_SCORES = ['min_ADE', 'mean_ADE']
    PAST_ADE_SCORES = ['min_past_ADE', 'mean_past_ADE']
    ALL_ADE_SCORES = ['min_all_ADE', 'mean_all_ADE']
    FDE_SCORES = ['min_FDE', 'mean_FDE']
    PAST_FDE_SCORES = ['min_past_FDE', 'mean_past_FDE']
    PRED_LENGTHS = ['past_pred_length', 'pred_length']
    OCCLUSION_MAP_SCORES = ['OAO', 'OAC', 'OAC_t0']
    if MEASURE == 'px':
        for score_var in [ADE_SCORES, PAST_ADE_SCORES, ALL_ADE_SCORES, FDE_SCORES, PAST_FDE_SCORES]:
            score_var = [f'{key}_px' for key in score_var]
    DISTANCE_METRICS = ADE_SCORES + FDE_SCORES + PAST_ADE_SCORES + PAST_FDE_SCORES + ALL_ADE_SCORES

    # Performance Summary #############################################################################################
    if args.perf_summary:
        print("\n\nPERFORMANCE SUMMARY:\n\n")
        base_experiment_names = [BASELINE_NO_POS_CONCAT]

        all_perf_df = generate_performance_summary_df(
            experiment_names=EXPERIMENTS, metric_names=DISTANCE_METRICS+PRED_LENGTHS+OCCLUSION_MAP_SCORES
        )
        all_perf_df.sort_values(by='min_FDE', inplace=True)

        if SHOW:
            print("Experiments Performance Summary:")
            print(all_perf_df)

        if SAVE:
            filename = "experiments_performance_summary.csv"
            filepath = os.path.join(PERFORMANCE_ANALYSIS_DIRECTORY, filename)
            print(f"saving dataframe to:\n{filepath}\n")
            all_perf_df.to_csv(filepath)

            for name in base_experiment_names:
                pretty_df = pretty_print_difference_summary_df(
                    summary_df=all_perf_df, base_experiment_name=name, mode='relative'
                )
                filename = f"experiments_performance_summary_relative_to_{name}.csv"
                filepath = os.path.join(PERFORMANCE_ANALYSIS_DIRECTORY, filename)
                print(f"saving dataframe to:\n{filepath}\n")
                pretty_df.to_csv(filepath)

        print(EXPERIMENT_SEPARATOR)

    # score boxplots vs last observed timesteps #######################################################################
    if args.boxplots:
        print("\n\nBOXPLOTS:\n\n")
        experiment_sets = {
            'occlusion': OCCLUSION_EXPERIMENTS,
            'imputation': IMPUTED_EXPERIMENTS,
            'experiments': OCCLUSION_EXPERIMENTS+IMPUTED_EXPERIMENTS
        }
        boxplot_scores = ADE_SCORES+PAST_ADE_SCORES+FDE_SCORES+PAST_FDE_SCORES+ALL_ADE_SCORES+OCCLUSION_MAP_SCORES
        figsize = (14, 10)

        if SHOW:
            fig, ax = plt.subplots(figsize=figsize)
            make_box_plot_occlusion_lengths(
                draw_ax=ax,
                experiments=OCCLUSION_EXPERIMENTS,
                plot_score='min_FDE'
            )
            plt.show()

        if SAVE:
            boxplot_directory = os.path.join(PERFORMANCE_ANALYSIS_DIRECTORY, 'boxplots')
            os.makedirs(boxplot_directory, exist_ok=True)

            for experiment_set, experiments in experiment_sets.items():
                for score_name in boxplot_scores:
                    fig, ax = plt.subplots(figsize=figsize)
                    make_box_plot_occlusion_lengths(
                        draw_ax=ax,
                        experiments=experiments,
                        plot_score=score_name
                    )
                    ax.set_title(f"{score_name}: [{experiment_set}] experiments")

                    experiment_dir = os.path.join(boxplot_directory, experiment_set)
                    os.makedirs(experiment_dir, exist_ok=True)

                    filename = f"{score_name}.png"
                    filepath = os.path.join(experiment_dir, filename)
                    print(f"saving boxplot to:\n{filepath}\n")
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    plt.close()

            for experiment in OCCLUSION_EXPERIMENTS+IMPUTED_EXPERIMENTS:
                for score_name in boxplot_scores:
                    fig, ax = plt.subplots(figsize=figsize)
                    make_box_plot_occlusion_lengths(draw_ax=ax, experiments=[experiment], plot_score=score_name)
                    ax.set_title(f"{score_name}: {experiment}")

                    experiment_dir = os.path.join(boxplot_directory, experiment)
                    os.makedirs(experiment_dir, exist_ok=True)

                    filename = f"{score_name}.png"
                    filepath = os.path.join(experiment_dir, filename)
                    print(f"saving boxplot to:\n{filepath}\n")
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    plt.close()

        print(EXPERIMENT_SEPARATOR)

    # OAC histograms ###################################################
    if args.oac_histograms:
        figsize = (16, 10)
        plot_score = 'OAC'
        as_percentage = False

        print(f"\n\n{plot_score} HISTOGRAMS:\n\n")

        if SHOW:
            experiment_name = OCCLUSIONFORMER_NO_MAP

            fig = plt.figure(figsize=figsize)
            make_oac_histograms_figure(fig=fig, experiment_name=experiment_name, plot_score=plot_score)
            plt.show()

        if SAVE:

            histograms_dir = os.path.join(PERFORMANCE_ANALYSIS_DIRECTORY, f'{plot_score}_histograms')
            os.makedirs(histograms_dir, exist_ok=True)

            for experiment_name in [
                OCCLUSIONFORMER_NO_MAP, OCCLUSIONFORMER_WITH_OCCL_MAP, OCCLUSIONFORMER_CAUSAL_ATTENTION
            ]:
                experiment_dir = os.path.join(histograms_dir, experiment_name)
                os.makedirs(experiment_dir, exist_ok=True)

                for occl_len in range(1, 7):
                    fig, ax = plt.subplots(figsize=figsize)

                    oac_histogram(
                        draw_ax=ax, experiment_name=experiment_name, plot_score=plot_score,
                        categorization=('past_pred_length', occl_len), as_percentage=as_percentage
                    )

                    filename = f"histogram_{occl_len}.png"
                    filepath = os.path.join(experiment_dir, filename)
                    print(f"saving {plot_score} histogram to:\n{filepath}\n")
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    plt.close()

                fig, ax = plt.subplots(figsize=figsize)
                oac_histograms_versus_lastobs(draw_ax=ax, experiment_name=experiment_name, plot_score=plot_score)
                filename = f"histograms_vs_last_obs.png"
                filepath = os.path.join(experiment_dir, filename)
                print(f"saving {plot_score} 2D-histogram to:\n{filepath}\n")
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()

                fig = plt.figure(figsize=figsize)
                make_oac_histograms_figure(fig=fig, experiment_name=experiment_name, plot_score=plot_score)
                filename = f"histograms_summary.png"
                filepath = os.path.join(experiment_dir, filename)
                print(f"saving {plot_score} histograms summary figure to:\n{filepath}\n")
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()

        print(EXPERIMENT_SEPARATOR)

    # qualitative display of predictions: comparison of experiments ###################################################
    if args.qual_compare:
        print("\n\nQUALITATIVE COMPARISON:\n\n")

        if SHOW:
            print("\n\n!!!!! SHOW IMPLEMENTATION OF QUALITATIVE COMPARISON NOT IMPLEMENTED !!!!!\n\n")

        if SAVE:
            comparisons_to_make = [
                {
                    'base': SDD_BASELINE_OCCLUSIONFORMER,
                    'comparisons': [BASELINE_NO_POS_CONCAT],
                    'scores': ['min_FDE', 'min_ADE'],
                    'only_occluded': False
                },
                {
                    'base': BASELINE_NO_POS_CONCAT,
                    'comparisons': [ORIGINAL_AGENTFORMER],
                    'scores': ['min_FDE', 'min_ADE'],
                    'only_occluded': False
                },
                {
                    'base': OCCLUSIONFORMER_NO_MAP,
                    'comparisons': [BASELINE_NO_POS_CONCAT],
                    'scores': ['min_FDE', 'min_ADE'],
                    'only_occluded': True
                },
                {
                    'base': OCCLUSIONFORMER_NO_MAP,
                    'comparisons': [OCCLUSIONFORMER_WITH_OCCL_MAP, OCCLUSIONFORMER_CAUSAL_ATTENTION],
                    'scores': ['min_FDE', 'min_ADE', 'min_past_FDE', 'OAO', 'OAC'],
                    'only_occluded': True
                },
                {
                    'base': OCCLUSIONFORMER_IMPUTED,
                    'comparisons': [BASELINE_NO_POS_CONCAT],
                    'scores': ['min_FDE', 'min_ADE'],
                    'only_occluded': True
                },
                {
                    'base': OCCLUSIONFORMER_IMPUTED,
                    'comparisons': [OCCLUSIONFORMER_WITH_OCCL_MAP_IMPUTED],
                    'scores': ['min_FDE', 'min_ADE', 'min_past_FDE', 'OAO', 'OAC'],
                    'only_occluded': True
                },
                {
                    'base': OCCLUSIONFORMER_MOMENTARY,
                    'comparisons': [BASELINE_NO_POS_CONCAT],
                    'scores': ['min_FDE', 'min_ADE'],
                    'only_occluded': False
                }
            ]
            n_max_agents = 16
            n_displays = 10
            figsize = (14, 10)
            qualitative_directory = os.path.join(PERFORMANCE_ANALYSIS_DIRECTORY, 'qualitative')
            os.makedirs(qualitative_directory, exist_ok=True)

            # for base_experiment, compare_list in experiments_to_compare.items():
            for comparison_dict in comparisons_to_make:
                base_experiment = comparison_dict['base']                   # str
                compare_list = comparison_dict['comparisons']               # list
                compare_scores = comparison_dict['scores']                  # list
                filter_only_occluded = comparison_dict['only_occluded']     # bool

                # performance dataframe, configuration, and dataloader of the base experiment
                base_df = get_perf_scores_df(base_experiment)
                config_base = Config(base_experiment)
                dataloader_base = HDF5DatasetSDD(config_base, log=None, split='test')

                # defining the directory of the base experiment
                base_directory = os.path.join(qualitative_directory, base_experiment)
                os.makedirs(base_directory, exist_ok=True)

                for experiment in compare_list:

                    # performance dataframe, configuration, and dataloader of the comparison experiment
                    compare_df = get_perf_scores_df(experiment)
                    config_compare = Config(experiment)
                    dataloader_compare = HDF5DatasetSDD(config_compare, log=None, split='test')

                    # the multiindex of rows to use for comparison
                    compare_rows = get_comparable_rows(
                        base_df=base_df,
                        compare_df=compare_df,
                        n_max_agents=n_max_agents
                    )

                    # defining the directory of the comparison experiment
                    compare_directory = os.path.join(base_directory, experiment)
                    os.makedirs(compare_directory, exist_ok=True)

                    # comparing the performance tables
                    diff_df = compare_df.loc[compare_rows].sub(base_df.loc[compare_rows])

                    # filtering to keep only occluded agents if desired
                    occluded_indices = None
                    if filter_only_occluded:
                        if 'past_pred_length' in base_df.columns:
                            occluded_indices = (base_df.loc[compare_rows]['past_pred_length'] != 0)
                        elif 'past_pred_length' in compare_df.columns:
                            occluded_indices = (compare_df.loc[compare_rows]['past_pred_length'] != 0)
                    if occluded_indices is not None:
                        diff_df = diff_df.loc[occluded_indices]

                    for compare_score in compare_scores:
                        assert compare_score in base_df.columns
                        assert compare_score in compare_df.columns
                        assert compare_score in diff_df.columns

                        sort_rows = diff_df.sort_values(
                            compare_score, ascending=True if compare_score not in OCCLUSION_MAP_SCORES else False
                        ).index
                        # summary_df = pd.DataFrame(columns=[base_experiment, experiment, 'difference'])
                        # summary_df[base_experiment] = base_df.loc[compare_rows, compare_score]
                        # summary_df[experiment] = compare_df.loc[compare_rows, compare_score]
                        # summary_df['difference'] = diff_df.loc[compare_rows, compare_score]
                        # print(f"Greatest {compare_score} difference: {experiment} vs. {base_experiment}")
                        # print(f"{summary_df.loc[sort_rows].head(n_displays)=}")

                        # defining the directory of the comparison experiment
                        score_directory = os.path.join(compare_directory, compare_score)
                        os.makedirs(score_directory, exist_ok=True)

                        for multi_index in sort_rows[:n_displays]:

                            idx, filename, agent_id = multi_index

                            # retrieve the corresponding entry name
                            instance_name = f"{filename}".rjust(8, '0')

                            # preparing the figure
                            fig, ax = plt.subplots(1, 2, figsize=figsize)
                            fig.canvas.manager.set_window_title(
                                f"{experiment} vs. {base_experiment}: (instance nr {instance_name})"
                            )

                            show_prediction_identities = [agent_id]

                            for i, (experiment_name, config, dataloader, perf_df) in enumerate([
                                (base_experiment, config_base, dataloader_base, base_df),
                                (experiment, config_compare, dataloader_compare, compare_df)
                            ]):
                                # defining path of the saved predictions
                                checkpoint_name = config.get_best_val_checkpoint_name()
                                saved_preds_dir = os.path.join(
                                    config.result_dir, dataloader.dataset_name, checkpoint_name, 'test'
                                )

                                # retrieve the input data dict
                                input_dict = dataloader.__getitem__(idx)
                                if 'map_homography' not in input_dict.keys():
                                    input_dict['map_homography'] = dataloader.map_homography

                                # retrieve the prediction data dict
                                pred_file = os.path.join(saved_preds_dir, instance_name)
                                assert os.path.exists(pred_file)
                                with open(pred_file, 'rb') as f:
                                    pred_dict = pickle.load(f)
                                pred_dict['map_homography'] = input_dict['map_homography']

                                visualize_input_and_predictions(
                                    draw_ax=ax[i],
                                    data_dict=input_dict,
                                    pred_dict=pred_dict,
                                    show_rgb_map=True,
                                    show_pred_agent_ids=show_prediction_identities,
                                    past_pred_alpha=0.5,
                                    future_pred_alpha=0.1 if compare_score in OCCLUSION_MAP_SCORES else 0.5
                                )

                                ax[i].set_title(experiment_name)

                            fig.subplots_adjust(wspace=0.10, hspace=0.0)

                            filename = f'{instance_name}.png'
                            filepath = os.path.join(score_directory, filename)
                            print(f"saving comparison figure to:\n{filepath}\n")
                            plt.savefig(filepath, dpi=300, bbox_inches='tight')
                            plt.close()

        print(EXPERIMENT_SEPARATOR)

    # qualitative display of predictions: comparison of experiments ###################################################
    if args.qual_example:
        print("\n\nQUALITATIVE EXAMPLE:\n\n")

        experiment_name = OCCLUSIONFORMER_WITH_OCCL_MAP
        instance_number = 7698
        show_pred_ids = [117]
        highlight_only_past_pred = True
        figsize = (14, 10)

        # preparing the dataloader for the experiment
        exp_df = get_perf_scores_df(experiment_name)
        config_exp = Config(experiment_name)
        dataloader_exp = HDF5DatasetSDD(config_exp, log=None, split='test')

        # retrieve the corresponding entry name
        instance_name = f"{instance_number}".rjust(8, '0')

        mini_df = exp_df.loc[instance_number, instance_number, :]
        print(f"Instance Dataframe:\n{mini_df}")
        show_agent_pred = []

        # preparing the figure
        fig, ax = plt.subplots(figsize=figsize)
        fig.canvas.manager.set_window_title(f"{experiment_name}: (instance nr {instance_name})")

        checkpoint_name = config_exp.get_best_val_checkpoint_name()
        saved_preds_dir = os.path.join(
            config_exp.result_dir, dataloader_exp.dataset_name, checkpoint_name, 'test'
        )

        # retrieve the input data dict
        input_dict = dataloader_exp.__getitem__(instance_number)
        if 'map_homography' not in input_dict.keys():
            input_dict['map_homography'] = dataloader_exp.map_homography

        # retrieve the prediction data dict
        pred_file = os.path.join(saved_preds_dir, instance_name)
        assert os.path.exists(pred_file)
        with open(pred_file, 'rb') as f:
            pred_dict = pickle.load(f)
        pred_dict['map_homography'] = input_dict['map_homography']

        visualize_input_and_predictions(
            draw_ax=ax,
            data_dict=input_dict,
            pred_dict=pred_dict,
            show_rgb_map=True,
            show_pred_agent_ids=show_pred_ids,
            past_pred_alpha=0.5,
            future_pred_alpha=0.1 if highlight_only_past_pred else 0.5
        )
        ax.legend()
        ax.set_title(experiment_name)
        fig.subplots_adjust(wspace=0.10, hspace=0.0)
        plt.show()

        print(EXPERIMENT_SEPARATOR)

    # map compliance vs generic performance ###########################################################################
    if False:       # uninteresting results, no matter the map score / trajectory score combination
        base_experiment = OCCLUSIONFORMER_NO_MAP
        compare_experiment = OCCLUSIONFORMER_WITH_OCCL_MAP
        x_score = 'OAO'
        y_score = 'mean_past_FDE'

        base_df = get_perf_scores_df(base_experiment)
        comp_df = get_perf_scores_df(compare_experiment)
        base_df = base_df[base_df['past_pred_length'] != 0]
        comp_df = comp_df[comp_df['past_pred_length'] != 0]

        compare_rows = get_comparable_rows(base_df=base_df, compare_df=comp_df)
        diff_df = comp_df.loc[compare_rows].sub(base_df.loc[compare_rows])

        xs = diff_df[x_score].to_numpy()
        ys = diff_df[y_score].to_numpy()

        fig, ax = plt.subplots()
        ax.scatter(xs, ys, marker='x', color='red', alpha=0.3)
        ax.set_xlabel(x_score)
        ax.set_ylabel(y_score)
        plt.show()

    # if False:
    #
    #     # plotting instances against one another:
    #     base_experiment = 'baseline_no_pos_concat'
    #     compare_experiment = 'occlusionformer_no_map'
    #     column_to_sort_by = 'min_FDE'
    #     n = 5
    #
    #     base_df = get_perf_scores_df(base_experiment)
    #     compare_df = get_perf_scores_df(compare_experiment)
    #     # filtering the comparison dataframe to only keep agents whose trajectories have been occluded
    #     compare_df = compare_df[compare_df['past_pred_length'] != 0]
    #
    #     diff_df = performance_dataframes_comparison(base_df, compare_df)
    #
    #     sort_indices = diff_df.sort_values(column_to_sort_by, ascending=True).index
    #     summary_df = pd.DataFrame(columns=[base_experiment, compare_experiment, 'difference'])
    #     summary_df[base_experiment] = base_df[[column_to_sort_by]]
    #     summary_df[compare_experiment] = compare_df[[column_to_sort_by]]
    #     summary_df['difference'] = diff_df[[column_to_sort_by]]
    #     # print(base_df.loc[sort_indices][[column_to_sort_by]].head(n))
    #     # print(compare_df.loc[sort_indices][[column_to_sort_by]].head(n))
    #     # print(diff_df.loc[sort_indices][[column_to_sort_by]].head(n))
    #     print(f"Greatest {column_to_sort_by} difference: {compare_experiment} vs. {base_experiment}")
    #     print(summary_df.loc[sort_indices].head(n))
    #
    #     # defining dataloader objects to retrieve input data
    #     config_base = Config(base_experiment)
    #     dataloader_base = HDF5DatasetSDD(config_base, log=None, split='test')
    #     config_compare = Config(compare_experiment)
    #     dataloader_compare = HDF5DatasetSDD(config_compare, log=None, split='test')
    #
    #     # for filename in sort_indices.get_level_values('filename').tolist()[:n]:
    #     for multi_index in sort_indices[:n]:
    #
    #         idx, filename, agent_id = multi_index
    #
    #         # retrieve the corresponding entry name
    #         # instance_name = filename.split('.')[0]
    #         instance_name = f"{filename}".rjust(8, '0')
    #
    #         # preparing the figure
    #         fig, ax = plt.subplots(1, 2)
    #         fig.canvas.manager.set_window_title(
    #             f"{compare_experiment} vs. {base_experiment}: (instance nr {instance_name})"
    #         )
    #
    #         # retrieving agent identities who are occluded, for which we are interested in displaying their predictions
    #         # show_prediction_agents = compare_df.loc[idx].index.get_level_values('agent_id').tolist()
    #         show_prediction_agents = [agent_id]
    #
    #         for i, (experiment_name, config, dataloader, perf_df) in enumerate([
    #             (base_experiment, config_base, dataloader_base, base_df),
    #             (compare_experiment, config_compare, dataloader_compare, compare_df)
    #         ]):
    #
    #             # defining path to saved predictions to retrieve prediction data
    #             checkpoint_name = config.get_best_val_checkpoint_name()
    #             saved_preds_dir = os.path.join(
    #                 config.result_dir, dataloader.dataset_name, checkpoint_name, 'test'
    #             )
    #
    #             # retrieve the input data dict
    #             input_dict = dataloader.__getitem__(idx)
    #             if 'map_homography' not in input_dict.keys():
    #                 input_dict['map_homography'] = dataloader.map_homography
    #
    #             # retrieve the prediction data dict
    #             pred_file = os.path.join(saved_preds_dir, instance_name)
    #             assert os.path.exists(pred_file)
    #             with open(pred_file, 'rb') as f:
    #                 pred_dict = pickle.load(f)
    #             pred_dict['map_homography'] = input_dict['map_homography']
    #
    #             visualize_input_and_predictions(
    #                 draw_ax=ax[i],
    #                 data_dict=input_dict,
    #                 pred_dict=pred_dict,
    #                 show_rgb_map=True,
    #                 show_pred_agent_ids=show_prediction_agents
    #             )
    #             # write_scores_per_mode(
    #             #     draw_ax=ax[i],
    #             #     pred_dict=pred_dict,
    #             #     show_agent_ids=show_prediction_agents,
    #             #     write_mode_number=True, write_ade_score=True, write_fde_score=True
    #             # )
    #             ax[i].legend()
    #
    #         # show figure
    #         plt.show()

    print(f"\n\nGoodbye!")

