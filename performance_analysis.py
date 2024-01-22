import argparse
import os.path
import yaml
import pandas as pd
import torch

from typing import List

from data.sdd_dataloader import PresavedDatasetSDD
from utils.utils import prepare_seed
from utils.config import Config, REPO_ROOT


def generate_performance_summary_df(runs_of_interest: List, scores_of_interest: List) -> pd.DataFrame:
    results_path = os.path.join(REPO_ROOT, 'results')
    df_columns = ['experiment', 'dataset_used', 'model_name'] + scores_of_interest
    performance_df = pd.DataFrame(columns=df_columns)

    for run in runs_of_interest:
        run_results_path = os.path.join(results_path, run, 'results')
        dataset_used = os.listdir(run_results_path)[0]

        for model_name in os.listdir(os.path.join(run_results_path, dataset_used)):
            target_path = os.path.join(run_results_path, dataset_used, model_name, 'test', 'prediction_scores.yml')
            score_values = [None] * len(scores_of_interest)

            if os.path.exists(target_path):
                with open(target_path, 'r') as f:
                    scores_dict = yaml.safe_load(f)
                    score_values = [scores_dict.get(key, 'N/A') for key in scores_of_interest]

            performance_df.loc[len(performance_df)] = [run, dataset_used, model_name] + score_values

    return performance_df


def get_perf_scores_df(experiment_name: str) -> pd.DataFrame:

    target_path = os.path.join(REPO_ROOT, 'results', experiment_name, 'results')

    dataset_used = os.listdir(target_path)[0]
    target_path = os.path.join(target_path, dataset_used)
    assert os.path.exists(target_path)

    default_model_name = os.listdir(target_path)[0]
    target_path = os.path.join(target_path, default_model_name, 'test', 'prediction_scores.csv')
    assert os.path.exists(target_path)

    df = pd.read_csv(target_path)

    df_indices = ['idx', 'filename', 'agent_id']
    df.set_index(keys=df_indices, inplace=True)

    return df


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
    summary_df = pd.DataFrame(columns=scores)
    scores.remove('past_pred_length')
    scores.remove('pred_length')

    operation = eval(f"pd.DataFrame.{operation}")

    for past_pred_length in sorted(df['past_pred_length'].unique()):

        mini_df = df[df['past_pred_length'] == past_pred_length]
        pred_length = mini_df['pred_length'].iloc[0]

        row_dict = operation(mini_df[scores]).to_dict()
        row_dict['past_pred_length'] = past_pred_length
        row_dict['pred_length'] = pred_length

        summary_df.loc[len(summary_df)] = row_dict
    return summary_df


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    scores_full_obs = [
        'min_ADE',
        'mean_ADE',
        'min_FDE',
        'mean_FDE',
        'rF'
    ]
    runs_full_obs = [
        'sdd_baseline_occlusionformer',
        'baseline_no_pos_concat'
    ]
    full_obs_perf_df = generate_performance_summary_df(
        runs_of_interest=runs_full_obs,
        scores_of_interest=scores_full_obs
    )
    print(f"\nExperiments on fully observed dataset:")
    print(full_obs_perf_df.sort_values('min_ADE'))

    runs_occl = [
        'occlusionformer_no_map',
        'occlusionformer_causal_attention',
        'occlusionformer_with_occl_map'
    ]
    scores_occl = [
        'min_ADE',
        'mean_ADE',
        'min_FDE',
        'mean_FDE',
        'mean_past_ADE',
        'mean_past_FDE',
        'past_pred_length',
        'pred_length',
        'rF',
        'OAC',
        'OAO'
    ]
    occluded_perf_df = generate_performance_summary_df(
        runs_of_interest=runs_occl,
        scores_of_interest=scores_occl
    )
    print(f"\nExperiments on occluded dataset:")
    print(occluded_perf_df.sort_values('min_ADE'))

    experiment = 'occlusionformer_no_map'
    score = 'OAO'
    exp_df = get_perf_scores_df(experiment_name=experiment)
    exp_df = remove_sample_columns(exp_df)
    exp_df = top_score_dataframe(df=exp_df, column_name=score, ascending=False, n=5)
    print(f'\nTop_scores of {experiment} ({score}):')
    # print(exp_df)
    print(exp_df[scores_occl])

    operation = 'mean'
    exp_df = get_perf_scores_df(experiment_name=experiment)
    exp_df = remove_sample_columns(exp_df)
    exp_df = summarize_per_occlusion_length(exp_df, operation=operation)
    print(f'\nScores summary separated by occlusion lengths for {experiment} ({operation}):')
    print(exp_df[scores_occl])

    print(f"\nGoodbye!")
