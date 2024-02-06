import argparse
import os.path
import pickle

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


def per_occlusion_length_boxplot(df: pd.DataFrame, column_name: str) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    # TODO: REWORK THIS FUNCTION:
    #       - being able to plot multiple experiments in one single plot (each one with separate color)
    #       - don't hard-specify 'past_pred_length', let the user choose
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


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 200)

    MEASURE = 'm'       # 'm' | 'px'
    EXPERIMENT_SEPARATOR = "\n\n\n\n" + "#" * 200 + "\n\n\n\n"

    EXPERIMENT_DICT = {
        1: 'sdd_baseline_occlusionformer',
        2: 'baseline_no_pos_concat',
        3: 'occlusionformer_no_map',
        4: 'occlusionformer_causal_attention',
        5: 'occlusionformer_imputed',
        6: 'occlusionformer_with_occl_map',
        7: 'occlusionformer_with_occl_map_imputed',
        8: 'original_agentformer',
        9: 'occlusionformer_momentary',
        10: 'occlusionformer_with_both_maps',
        11: 'const_vel_fully_observed',
        12: 'const_vel_fully_observed_momentary_2',
        13: 'const_vel_occlusion_simulation',
        14: 'const_vel_occlusion_simulation_imputed'
    }

    DISTANCE_METRICS = [
        'min_ADE', 'min_FDE',
        'mean_ADE', 'mean_FDE',
        'min_past_ADE', 'min_past_FDE',
        'mean_past_ADE', 'mean_past_FDE',
        # 'min_all_ADE', 'mean_all_ADE',
    ]
    if MEASURE == 'px':
        DISTANCE_METRICS = [f'{key}_px' for key in DISTANCE_METRICS]
    OTHER_METRICS = [
        'past_pred_length', 'pred_length',
        'OAC', 'OAO'
    ]

    ###################################################################################################################

    if False:
        all_perf_df = generate_performance_summary_df(
            experiment_names=[EXPERIMENT_DICT[i] for i in [1, 2]],
            metric_names=DISTANCE_METRICS+OTHER_METRICS
        )
        print(f"All experiments:")
        # print(all_perf_df)
        print(pretty_print_difference_summary_df(
            summary_df=all_perf_df,
            base_experiment_name='baseline_no_pos_concat',
            mode='relative'
        ))
        print(EXPERIMENT_SEPARATOR)

    if False:
        for experiment_name in ['occlusionformer_no_map', 'occlusionformer_with_occl_map']:
            experiment_df = get_perf_scores_df(experiment_name=experiment_name)
            experiment_df = remove_k_sample_columns(df=experiment_df)
            operation = 'mean'
            experiment_df = reduce_by_unique_column_values(
                df=experiment_df,
                column_name='past_pred_length',
                operation=operation)
            print(f'\n\n\n\nScores summary separated by occlusion lengths for {experiment_name} ({operation}):')
            print(experiment_df[['count'] + DISTANCE_METRICS + OTHER_METRICS])

    # box_plot_scores = OCCL_SIM_SCORES + EXTRA_OCCL_SIM_SCORES
    # [box_plot_scores.remove(score) for score in ['rF', 'past_pred_length', 'pred_length']]
    # for experiment in ['occlusionformer_no_map']:
    #     for score in box_plot_scores:
    #         exp_df = get_perf_scores_df(experiment_name=experiment)
    #         exp_df = remove_sample_columns(exp_df)
    #         fig, ax = per_occlusion_length_boxplot(df=exp_df, column_name=score)
    #         ax.set_title(f"{experiment}: {score} / occlusion duration")
    #         plt.show()
    # print(EXPERIMENT_SEPARATOR)

    # base_experiment = 'baseline_no_pos_concat'
    # compare_experiment = 'occlusionformer_no_map'
    # base_df = get_perf_scores_df(base_experiment)
    # base_df = remove_k_sample_columns(base_df)
    # compare_df = get_perf_scores_df(compare_experiment)
    # compare_df = remove_k_sample_columns(compare_df)
    # print(performance_dataframes_comparison(base_df, compare_df))

    if True:
        from data.sdd_dataloader import HDF5DatasetSDD
        from utils.sdd_visualize import visualize_input_and_predictions, write_scores_per_mode
        from utils.performance_metrics import compute_samples_ADE, compute_samples_FDE
        from model.agentformer_loss import index_mapping_gt_seq_pred_seq

        # plotting instances against one another:
        base_experiment = 'baseline_no_pos_concat'
        compare_experiment = 'occlusionformer_no_map'
        column_to_sort_by = 'min_FDE'
        n = 10

        base_df = get_perf_scores_df(base_experiment)
        compare_df = get_perf_scores_df(compare_experiment)
        # filtering the comparison dataframe to only keep agents whose trajectories have been occluded
        compare_df = compare_df[compare_df['past_pred_length'] != 0]

        diff_df = performance_dataframes_comparison(base_df, compare_df)

        sort_indices = diff_df.sort_values(column_to_sort_by, ascending=True).index
        summary_df = pd.DataFrame(columns=[base_experiment, compare_experiment, 'difference'])
        summary_df[base_experiment] = base_df[[column_to_sort_by]]
        summary_df[compare_experiment] = compare_df[[column_to_sort_by]]
        summary_df['difference'] = diff_df[[column_to_sort_by]]
        # print(base_df.loc[sort_indices][[column_to_sort_by]].head(n))
        # print(compare_df.loc[sort_indices][[column_to_sort_by]].head(n))
        # print(diff_df.loc[sort_indices][[column_to_sort_by]].head(n))
        print(f"Greatest {column_to_sort_by} difference: {compare_experiment} vs. {base_experiment}")
        print(summary_df.loc[sort_indices].head(n))

        # defining dataloader objects to retrieve input data
        config_base = Config(base_experiment)
        dataloader_base = HDF5DatasetSDD(config_base, log=None, split='test')
        config_compare = Config(compare_experiment)
        dataloader_compare = HDF5DatasetSDD(config_compare, log=None, split='test')

        # for filename in sort_indices.get_level_values('filename').tolist()[:n]:
        for multi_index in sort_indices[:n]:

            idx, filename, agent_id = multi_index

            # retrieve the corresponding entry name
            # instance_name = filename.split('.')[0]
            instance_name = f"{filename}".rjust(8, '0')

            # preparing the figure
            fig, ax = plt.subplots(1, 2)
            fig.canvas.manager.set_window_title(
                f"{compare_experiment} vs. {base_experiment}: (instance nr {instance_name})"
            )

            # retrieving agent identities who are occluded, for which we are interested in displaying their predictions
            show_prediction_agents = compare_df.loc[idx].index.get_level_values('agent_id').tolist()

            for i, (experiment_name, config, dataloader, perf_df) in enumerate([
                (base_experiment, config_base, dataloader_base, base_df),
                (compare_experiment, config_compare, dataloader_compare, compare_df)
            ]):

                # defining path to saved predictions to retrieve prediction data
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
                    show_pred_agent_ids=show_prediction_agents
                )
                write_scores_per_mode(
                    draw_ax=ax[i],
                    pred_dict=pred_dict,
                    show_agent_ids=show_prediction_agents,
                    write_mode_number=True, write_ade_score=True, write_fde_score=True
                )
                ax[i].legend()

            # show figure
            plt.show()

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

    if False:
        for comp_run in comparison_runs:
            base_df = get_perf_scores_df(experiment_name=comp_run[0])
            comp_df = get_perf_scores_df(experiment_name=comp_run[1])
            compare_score = comp_run[2]
            out_df = retrieve_interesting_performance_diff_cases(
                base_df=base_df, comp_df=comp_df, column_name=compare_score, n=5
            )
            print(f"Pickle files to retrieve for: {comp_run}")
            # print(" ".join(out_df.index.get_level_values('filename').tolist()))
            print()
        print(EXPERIMENT_SEPARATOR)


    # base_df = get_perf_scores_df(experiment_name=base_run)
    # base_df = base_df[FULL_OBS_SCORES]
    # comp_df = get_perf_scores_df(experiment_name=compare_run)
    # comp_df = comp_df[FULL_OBS_SCORES]
    # fig, ax = scatter_perf_gain_vs_perf_base(base_df=base_df, comp_df=comp_df, col_name='min_ADE', relative=True)
    # plt.show()

    print(f"\n\nGoodbye!")

