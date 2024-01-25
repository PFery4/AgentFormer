import argparse
import os.path

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import yaml
import pandas as pd
import torch

from typing import Dict, List, Optional, Tuple

from data.sdd_dataloader import PresavedDatasetSDD
from utils.utils import prepare_seed
from utils.config import Config, REPO_ROOT


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


def generate_performance_summary_df(runs_of_interest: List, scores_of_interest: List) -> pd.DataFrame:
    df_columns = ['experiment', 'dataset_used', 'model_name'] + scores_of_interest
    performance_df = pd.DataFrame(columns=df_columns)

    for run in runs_of_interest:
        run_results_path = os.path.join(REPO_ROOT, 'results', run, 'results')
        dataset_used = os.listdir(run_results_path)[0]
        model_name = os.listdir(os.path.join(run_results_path, dataset_used))[0]

        scores_dict = get_perf_scores_dict(experiment_name=run)
        scores_dict['experiment'] = run
        scores_dict['dataset_used'] = dataset_used
        scores_dict['model_name'] = model_name

        performance_df.loc[len(performance_df)] = scores_dict

    return performance_df


def relative_improvement_df(summary_df: pd.DataFrame, base_experiment: str) -> pd.DataFrame:
    # each row contains summary score information for one experiment
    assert base_experiment in summary_df['experiment'].to_list()
    assert (summary_df['experiment'] == base_experiment).sum() == 1

    rel_df = summary_df.copy()

    compare_columns = summary_df.columns[summary_df.dtypes == float]

    rel_df.loc[:, compare_columns] = rel_df.loc[:, compare_columns].div(rel_df.loc[rel_df['experiment'] == base_experiment, compare_columns].iloc[0])
    rel_df.loc[:, compare_columns] = rel_df.loc[:, compare_columns] - 1.0

    return rel_df


def pretty_print_relative_improvement_df(summary_df: pd.DataFrame, base_experiment: str) -> pd.DataFrame:
    compare_columns = summary_df.columns[summary_df.dtypes == float]

    rel_df = relative_improvement_df(summary_df=summary_df, base_experiment=base_experiment)

    rel_df.loc[:, compare_columns] = rel_df.loc[:, compare_columns] * 100
    rel_df.loc[:, compare_columns] = rel_df.loc[:, compare_columns].applymap(
        lambda x: f" (+{x:.3f}%)" if x>=0. else f" ({x:.3f}%)"
    )

    out_df = summary_df.copy()
    out_df.loc[:, compare_columns] = out_df.loc[:, compare_columns].round(5).astype(str)
    out_df.loc[:, compare_columns] = out_df.loc[:, compare_columns] + rel_df.loc[:, compare_columns]

    return out_df


def top_score_dataframe(df: pd.DataFrame, column_name: str, ascending: bool = True, n: int = 5) -> pd.DataFrame:
    return df.sort_values(column_name, ascending=ascending).head(n)


def remove_sample_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_columns = [name for name in df.columns.tolist() if not name.startswith('K')]
    return df[keep_columns]


def summarize_per_occlusion_length(df: pd.DataFrame, operation: str = 'mean') -> pd.DataFrame:
    assert 'past_pred_length' in df.columns
    assert 'pred_length' in df.columns

    assert operation in ['mean', 'std']

    scores = df.columns.tolist()
    summary_df = pd.DataFrame(columns=['count'] + scores)
    scores.remove('past_pred_length')
    scores.remove('pred_length')

    operation = eval(f"pd.DataFrame.{operation}")

    for past_pred_length in sorted(df['past_pred_length'].unique()):

        mini_df = df[df['past_pred_length'] == past_pred_length]
        pred_length = mini_df['pred_length'].iloc[0]

        row_dict = operation(mini_df[scores]).to_dict()
        row_dict['count'] = len(mini_df)
        row_dict['past_pred_length'] = past_pred_length
        row_dict['pred_length'] = pred_length

        summary_df.loc[len(summary_df)] = row_dict
    return summary_df


def per_occlusion_length_boxplot(df: pd.DataFrame, column_name: str) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    assert 'past_pred_length' in df.columns
    assert 'pred_length' in df.columns

    assert column_name in df.columns

    fig, ax = plt.subplots()

    scores_dict = dict()
    past_pred_lengths = sorted(df['past_pred_length'].unique())

    for past_pred_length in past_pred_lengths:

        mini_df = df[(df['past_pred_length'] == past_pred_length) & (pd.notna(df[column_name]))]

        scores = mini_df[column_name].to_numpy()
        scores_dict[past_pred_length] = scores

    ax.boxplot(scores_dict.values())
    ax.set_xticklabels(scores_dict.keys())

    return fig, ax


def performance_dataframe_comparison(
        base_df: pd.DataFrame, comp_df: pd.DataFrame, relative: bool = True
) -> pd.DataFrame:
    # checking that the two dataframes have the same indices / cover the same cases
    # if len(base_df.index.difference(comp_df.index)):
    #     print(f"Comparing only rows that are shared between the two tables:\n"
    #           f"{len(base_df.index.difference(comp_df.index))} rows will be ignored")

    # if len(base_df.columns.difference(comp_df.columns)):
    #     print(f"Comparing only columns that are shared between the two tables:\n"
    #           f"{len(base_df.columns.difference(comp_df.columns))} columns will be ignored")

    # removing rows that are not shared across base_df and comp_df
    keep_indices = base_df.index.intersection(comp_df.index)
    base_df = base_df.loc[keep_indices]
    comp_df = comp_df.loc[keep_indices]

    out_df = comp_df.sub(base_df)

    if relative:
        out_df = out_df.div(base_df)

    return out_df


def scatter_perf_gain_vs_perf_base(
        base_df: pd.DataFrame, comp_df: pd.DataFrame, col_name: str, relative: bool = True
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    assert col_name in base_df.columns
    assert col_name in comp_df.columns

    keep_indices = base_df.index.intersection(comp_df.index)

    fig, ax = plt.subplots()
    diff = comp_df.loc[keep_indices, col_name].sub(base_df.loc[keep_indices, col_name])

    if relative:
        diff = diff.div(base_df.loc[keep_indices, col_name])

    ax.scatter(base_df.loc[keep_indices, col_name], diff, marker='x', alpha=0.5)

    return fig, ax


def retrieve_interesting_performance_diff_cases(
        base_df: pd.DataFrame, comp_df: pd.DataFrame, column_name: str, n: int
) -> pd.DataFrame:
    diff_df = performance_dataframe_comparison(base_df=base_df, comp_df=comp_df, relative=True)

    diff_df = diff_df.sort_values(column_name, ascending=True)

    min_diff = diff_df.head(n)
    max_diff = diff_df.tail(n)

    return pd.concat([min_diff, max_diff])


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 200)

    MEASURE = 'm'       # 'm' | 'px'
    SORT_BY_SCORE = 'min_ADE'
    EXPERIMENT_SEPARATOR = "\n\n\n\n" + "#" * 200 + "\n\n\n\n"

    FULL_OBS_RUNS = [
        'sdd_baseline_occlusionformer',
        'baseline_no_pos_concat',
        'const_vel_fully_observed',
        'const_vel_fully_observed_momentary_2'
    ]
    FULL_OBS_SCORES = ['min_ADE', 'min_FDE', 'mean_ADE', 'mean_FDE']
    OCCL_SIM_RUNS = [
        'occlusionformer_no_map',
        'occlusionformer_causal_attention',
        'occlusionformer_with_occl_map'
    ]
    OCCL_SIM_SCORES = [
        'min_ADE',
        'mean_ADE',
        'min_FDE',
        'mean_FDE',
        'mean_past_ADE',
        'mean_past_FDE',
        'rF'
    ]
    if MEASURE == 'px':
        SORT_BY_SCORE += '_px'
        FULL_OBS_SCORES = [f'{key}_px' for key in FULL_OBS_SCORES]
        OCCL_SIM_SCORES = [f'{key}_px' for key in OCCL_SIM_SCORES]
    EXTRA_OCCL_SIM_SCORES = [
        'past_pred_length',
        'pred_length',
        'OAC',
        'OAO'
    ]

    ###################################################################################################################

    all_perf_df = generate_performance_summary_df(
        runs_of_interest=FULL_OBS_RUNS+OCCL_SIM_RUNS,
        scores_of_interest=OCCL_SIM_SCORES+EXTRA_OCCL_SIM_SCORES
    )
    print(f"All experiments:")
    # print(all_perf_df)
    print(pretty_print_relative_improvement_df(all_perf_df, 'baseline_no_pos_concat'))
    print(EXPERIMENT_SEPARATOR)

    # full_obs_perf_df = generate_performance_summary_df(
    #     runs_of_interest=FULL_OBS_RUNS,
    #     scores_of_interest=FULL_OBS_SCORES
    # )
    # print(f"Experiments on fully observed dataset:")
    # # print(full_obs_perf_df.sort_values('min_ADE'))
    # print(pretty_print_relative_improvement_df(full_obs_perf_df, 'sdd_baseline_occlusionformer'))
    # print(EXPERIMENT_SEPARATOR)

    # occluded_perf_df = generate_performance_summary_df(
    #     runs_of_interest=OCCL_SIM_RUNS,
    #     scores_of_interest=OCCL_SIM_SCORES+EXTRA_OCCL_SIM_SCORES
    # )
    # print(f"Experiments on occluded dataset:")
    # # print(occluded_perf_df.sort_values('min_ADE'))
    # print(pretty_print_relative_improvement_df(occluded_perf_df, 'occlusionformer_no_map'))
    # print(EXPERIMENT_SEPARATOR)

    for experiment in ['occlusionformer_no_map', 'occlusionformer_with_occl_map']:
        for operation in ['mean']:
            exp_df = get_perf_scores_df(experiment_name=experiment)
            exp_df = remove_sample_columns(exp_df)
            exp_df = summarize_per_occlusion_length(df=exp_df, operation=operation)
            print(f'\n\n\n\nScores summary separated by occlusion lengths for {experiment} ({operation}):')
            print(exp_df[['count']+OCCL_SIM_SCORES+EXTRA_OCCL_SIM_SCORES])
    print(EXPERIMENT_SEPARATOR)

    box_plot_scores = OCCL_SIM_SCORES + EXTRA_OCCL_SIM_SCORES
    [box_plot_scores.remove(score) for score in ['rF', 'past_pred_length', 'pred_length']]
    for experiment in ['occlusionformer_no_map']:
        for score in box_plot_scores:
            exp_df = get_perf_scores_df(experiment_name=experiment)
            exp_df = remove_sample_columns(exp_df)
            fig, ax = per_occlusion_length_boxplot(df=exp_df, column_name=score)
            ax.set_title(f"{experiment}: {score} / occlusion duration")
            plt.show()
    print(EXPERIMENT_SEPARATOR)

    # base_run = 'baseline_no_pos_concat'
    # compare_run = 'occlusionformer_no_map'
    # compare_score = 'min_ADE'
    # n_top = 5
    # base_df = get_perf_scores_df(experiment_name=base_run)
    # base_df = base_df[FULL_OBS_SCORES]
    # comp_df = get_perf_scores_df(experiment_name=compare_run)
    # comp_df = comp_df[FULL_OBS_SCORES]
    # # out_df = performance_dataframe_comparison(base_df, comp_df, relative=True)
    # out_df = retrieve_interesting_performance_diff_cases(base_df, comp_df, column_name=compare_score, n=n_top)
    # print(f"Per agent performance comparison: {compare_run} vs. {base_run} (Best/Worst {n_top} agents in {compare_score}):")
    # print(out_df)
    # # print(out_df.sort_values(compare_score, ascending=True).head(5))
    # print(EXPERIMENT_SEPARATOR)

    # # qualitative display of predictions
    # comparison_runs = [
    #     ['baseline_no_pos_concat', 'occlusionformer_no_map', 'min_ADE'],
    #     # ['baseline_no_pos_concat', 'occlusionformer_no_map', 'min_FDE'],
    #     ['occlusionformer_no_map', 'occlusionformer_with_occl_map', 'min_ADE'],
    #     # ['occlusionformer_no_map', 'occlusionformer_with_occl_map', 'min_FDE'],
    #     # ['occlusionformer_no_map', 'occlusionformer_with_occl_map', 'OAO'],
    #     # ['occlusionformer_no_map', 'occlusionformer_with_occl_map', 'OAC'],
    #     # ['baseline_no_pos_concat', 'occlusionformer_causal_attention', 'min_ADE'],
    #     # ['baseline_no_pos_concat', 'occlusionformer_causal_attention', 'min_FDE'],
    # ]
    # for comp_run in comparison_runs:
    #     base_df = get_perf_scores_df(experiment_name=comp_run[0])
    #     comp_df = get_perf_scores_df(experiment_name=comp_run[1])
    #     compare_score = comp_run[2]
    #     out_df = retrieve_interesting_performance_diff_cases(
    #         base_df=base_df, comp_df=comp_df, column_name=compare_score, n=5
    #     )
    #     print(f"Pickle files to retrieve for: {comp_run}")
    #     # print(" ".join(out_df.index.get_level_values('filename').tolist()))
    #     print()
    # print(EXPERIMENT_SEPARATOR)


    # base_df = get_perf_scores_df(experiment_name=base_run)
    # base_df = base_df[FULL_OBS_SCORES]
    # comp_df = get_perf_scores_df(experiment_name=compare_run)
    # comp_df = comp_df[FULL_OBS_SCORES]
    # fig, ax = scatter_perf_gain_vs_perf_base(base_df=base_df, comp_df=comp_df, col_name='min_ADE', relative=True)
    # plt.show()

    print(f"\n\nGoodbye!")

