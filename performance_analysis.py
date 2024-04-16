import argparse
import os.path
import pickle

import matplotlib.axes
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
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
from utils.performance_analysis import \
    generate_performance_summary_df, \
    pretty_print_difference_summary_df, \
    make_box_plot_occlusion_lengths, \
    make_oac_histograms_figure, \
    oac_histogram, \
    oac_histograms_versus_lastobs, \
    get_perf_scores_df, \
    get_comparable_rows


if __name__ == '__main__':
    # Script Controls #################################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--perf_summary', action='store_true', default=False)
    parser.add_argument('--boxplots', action='store_true', default=False)
    parser.add_argument('--oac_histograms', action='store_true', default=False)
    parser.add_argument('--qual_compare', action='store_true', default=False)
    parser.add_argument('--qual_example', action='store_true', default=False)
    parser.add_argument('--comp_phase_1_2', action='store_true', default=False)
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
    pd.set_option('display.max_colwidth', 75)

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
    OCCLUSIONFORMER_CAUSAL_ATTENTION_FULLY_OBSERVED_162 = 'occlusionformer_causal_attention_fully_observed_162'
    OCCLUSIONFORMER_CAUSAL_ATTENTION_FULLY_OBSERVED_314 = 'occlusionformer_causal_attention_fully_observed_314'

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
        OCCLUSIONFORMER_CAUSAL_ATTENTION_FULLY_OBSERVED,
        OCCLUSIONFORMER_CAUSAL_ATTENTION_IMPUTED,
        OCCLUSIONFORMER_CAUSAL_ATTENTION_OCCL_MAP,
        OCCLUSIONFORMER_OFFSET_TIMECODES,
        OCCLUSIONFORMER_IMPUTED_WITH_MARKERS,
        OCCLUSIONFORMER_CAUSAL_ATTENTION_FULLY_OBSERVED_162,
        OCCLUSIONFORMER_CAUSAL_ATTENTION_FULLY_OBSERVED_314
    ]

    FULLY_OBSERVED_EXPERIMENTS = [
        SDD_BASELINE_OCCLUSIONFORMER,
        BASELINE_NO_POS_CONCAT,
        ORIGINAL_AGENTFORMER,
        OCCLUSIONFORMER_CAUSAL_ATTENTION_FULLY_OBSERVED,
        # CONST_VEL_FULLY_OBSERVED
    ]
    OCCLUSION_EXPERIMENTS = [
        OCCLUSIONFORMER_NO_MAP,
        OCCLUSIONFORMER_CAUSAL_ATTENTION,
        OCCLUSIONFORMER_WITH_OCCL_MAP,
        OCCLUSIONFORMER_CAUSAL_ATTENTION_OCCL_MAP,
        # OCCLUSIONFORMER_OFFSET_TIMECODES,
        # CONST_VEL_OCCLUSION_SIMULATION
    ]
    IMPUTED_EXPERIMENTS = [
        OCCLUSIONFORMER_IMPUTED,
        OCCLUSIONFORMER_WITH_OCCL_MAP_IMPUTED,
        OCCLUSIONFORMER_CAUSAL_ATTENTION_IMPUTED,
        # OCCLUSIONFORMER_IMPUTED_WITH_MARKERS,
        # CONST_VEL_OCCLUSION_SIMULATION_IMPUTED
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

        # experiment_names = EXPERIMENTS
        experiment_names = [
            # 'original_100_pre',
            # 'original_101_pre',
            # 'original_102_pre',
            # 'original_103_pre',
            # 'original_104_pre',
            # 'original_100',
            # 'original_101',
            # 'original_102',
            # 'original_103',
            # 'original_104',
            # 'occlusionformer_basis_bias_1_pre',
            # 'occlusionformer_basis_bias_2_pre',
            # 'occlusionformer_basis_bias_3_pre',
            # 'occlusionformer_basis_bias_4_pre',
            # 'occlusionformer_basis_bias_5_pre',
            # 'occlusionformer_basis_bias_1',
            # 'occlusionformer_basis_bias_2',
            # 'occlusionformer_basis_bias_3',
            # 'occlusionformer_basis_bias_4',
            # 'occlusionformer_basis_bias_5',
            'v2_difficult_occlusions_pre',
            'v2_difficult_occlusions_with_map_w5_pre',
            'v2_difficult_occlusions_with_map_w10_pre',
            'v2_difficult_occlusions_with_map_w15_pre'
        ]

        # metric_names = DISTANCE_METRICS+PRED_LENGTHS+OCCLUSION_MAP_SCORES
        metric_names = ADE_SCORES + FDE_SCORES + OCCLUSION_MAP_SCORES

        all_perf_df = generate_performance_summary_df(
            experiment_names=experiment_names, metric_names=metric_names
        )
        all_perf_df.sort_values(by='min_FDE', inplace=True)

        if SHOW:
            print("Experiments Performance Summary:")
            print(all_perf_df)

        if SAVE:
            # base_experiment_names = [BASELINE_NO_POS_CONCAT]
            base_experiment_names = []

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
        plot_score = 'OAC'      # 'OAC', 'OAC_t0', 'OAO'
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

        experiment_name = ORIGINAL_AGENTFORMER
        experiment_name = BASELINE_NO_POS_CONCAT
        experiment_name = OCCLUSIONFORMER_CAUSAL_ATTENTION
        experiment_name = OCCLUSIONFORMER_CAUSAL_ATTENTION_OCCL_MAP
        experiment_name = OCCLUSIONFORMER_MOMENTARY
        experiment_name = OCCLUSIONFORMER_CAUSAL_ATTENTION_IMPUTED
        experiment_name = SDD_BASELINE_OCCLUSIONFORMER
        experiment_name = OCCLUSIONFORMER_WITH_OCCL_MAP
        experiment_name = OCCLUSIONFORMER_NO_MAP
        experiment_name = OCCLUSIONFORMER_WITH_OCCL_MAP_IMPUTED
        experiment_name = OCCLUSIONFORMER_IMPUTED
        experiment_name = OCCLUSIONFORMER_CAUSAL_ATTENTION_FULLY_OBSERVED_162
        experiment_name = OCCLUSIONFORMER_CAUSAL_ATTENTION_FULLY_OBSERVED_314
        experiment_name = OCCLUSIONFORMER_CAUSAL_ATTENTION_FULLY_OBSERVED

        experiment_list = [
            # 'occlusionformer_basis_bias_1',
            # 'occlusionformer_basis_bias_2',
            # 'occlusionformer_basis_bias_3',
            # 'occlusionformer_basis_bias_4',
            # 'occlusionformer_basis_bias_5',
            CONST_VEL_OCCLUSION_SIMULATION,
            'v2_difficult_occlusions_pre',
            'v2_difficult_occlusions_with_map_w5_pre',
            'v2_difficult_occlusions_with_map_w10_pre',
            'v2_difficult_occlusions_with_map_w15_pre'
        ]

        # experiment_name = experiment_name + '_pre'
        print(f"{experiment_name=}")
        instance_number, show_pred_ids = 1000, [452]
        instance_number, show_pred_ids = 3000, [14]
        instance_number, show_pred_ids = 6000, [2]          # single agent, moving forward
        instance_number, show_pred_ids = 8000, [12]         # fast agent (cyclist) (no occlusion)
        instance_number, show_pred_ids = 9000, [119, 217]
        instance_number, show_pred_ids = 9998, [13]         # productive use of occlusion map
        instance_number, show_pred_ids = 2000, [116, 481]
        instance_number, show_pred_ids = 8881, [181]        # large turning maneuver under occlusion
        instance_number, show_pred_ids = 5000, [194, 222, 314]
        instance_number, show_pred_ids = 0, [66, 68, 69]
        instance_number, show_pred_ids = 7698, [117]
        instance_number, show_pred_ids = 500, [20, 41, 98, 111]
        instance_number, show_pred_ids = 4000, [4, 10]          # idles
        # instance_number, show_pred_ids = 11025, [110]          # what?
        #
        # # using occlusion map, OAO / OAC_t0 = inf (OAC_t0 is 0.)
        # instance_number, show_pred_ids = 4041, [38]          # "needle", poor use of occlusion map
        # instance_number, show_pred_ids = 7930, [37]          # "needle", poor use of occlusion map
        # instance_number, show_pred_ids = 8074, [49]          # "needle", poor use of occlusion map
        # instance_number, show_pred_ids = 46, [38]          # "needle", poor use of occlusion map
        # instance_number, show_pred_ids = 9973, [0]          # "needle", poor use of occlusion map
        #
        # # using occlusion map, OAO / OAC_t0 is very high (filtering away OAC_t0 = 0.)
        # instance_number, show_pred_ids = 446, [51]          # poor use of occlusion map
        # instance_number, show_pred_ids = 5769, [91]          # poor use of occlusion map, no_map impressively better
        # instance_number, show_pred_ids = 4757, [115]          # poor use of occlusion map, no_map impressively better

        # # Constant Velocity fails to place occluded agent inside the occlusion zone at t=0
        instance_number, show_pred_ids = 7231, [82]     # a needle
        instance_number, show_pred_ids = 57, [39]       # following the edge of the occlusion zone
        instance_number, show_pred_ids = 1665, [628]       # CV just too slow before entering the occlusion zone
        instance_number, show_pred_ids = 4061, [85]       # CV just too slow before entering the occlusion zone
        instance_number, show_pred_ids = 9779, [247]    # probably an inaccuracy in the coordinate frame (just inside the zone, but still registered as outside)
        instance_number, show_pred_ids = 8872, [209]    # CV just too slow before entering the occlusion zone
        instance_number, show_pred_ids = 6435, [18]     # probably an inaccuracy in the coordinate frame (just inside the zone, but still registered as outside)
        instance_number, show_pred_ids = 5465, [496]    # probably an inaccuracy in the coordinate frame (just inside the zone, but still registered as outside)
        instance_number, show_pred_ids = 1734, [486]    # overextension of the prediction
        # instance_number, show_pred_ids = 8966, [210]    # following the edge of the occlusion zone
        # instance_number, show_pred_ids = 859, [115]    # probably an inaccuracy in the coordinate frame (just inside the zone, but still registered as outside)
        # instance_number, show_pred_ids = 2366, [17]    # overextension of the prediction
        # instance_number, show_pred_ids = 3313, [28]    # CV just too slow before entering the occlusion zone
        # instance_number, show_pred_ids = 69, [39]    # overextension of the prediction
        # instance_number, show_pred_ids = 8011, [39]    # overextension of the prediction
        # instance_number, show_pred_ids = 10973, [107]    # overextension of the prediction
        # instance_number, show_pred_ids = 6185, [24]    # overextension of the prediction
        # instance_number, show_pred_ids = 7589, [64]     # following the edge of the occlusion zone
        # instance_number, show_pred_ids = 4947, [434]    # following the edge of the occlusion zone
        # instance_number, show_pred_ids = 3030, [93]    # following the edge of the occlusion zone
        # instance_number, show_pred_ids = 5886, [410]    # overextension of the prediction
        # instance_number, show_pred_ids = 11703, [26]    # CV just too slow before entering the occlusion zone
        # instance_number, show_pred_ids = 9973, [0]    # a needle
        # instance_number, show_pred_ids = 716, [23]    # probably an inaccuracy in the coordinate frame (just inside the zone, but still registered as outside)
        # instance_number, show_pred_ids = 1898, [246]    # probably an inaccuracy in the coordinate frame (just inside the zone, but still registered as outside)

        # # Constant Velocity fails to place occluded agent inside the occlusion zone at t=0, AND past_pred_length > 1
        # instance_number, show_pred_ids = 5369, [125]    # following the edge of the occlusion zone
        # instance_number, show_pred_ids = 11049, [118]      # undershot
        # instance_number, show_pred_ids = 5643, [693]      # following the edge of the occlusion zone
        # instance_number, show_pred_ids = 6305, [6]      # overshot
        # instance_number, show_pred_ids = 175, [39]      # following the edge of the occlusion zone
        # instance_number, show_pred_ids = 2898, [1]      # idle
        # instance_number, show_pred_ids = 7446, [24]      # overshot
        # instance_number, show_pred_ids = 2366, [17]      # overshot
        # instance_number, show_pred_ids = 8096, [49]      # idle
        # instance_number, show_pred_ids = 4957, [163]      # overshot

        highlight_only_past_pred = True
        figsize = (14, 10)

        for exp_name in experiment_list:
            experiment_name = exp_name
            # preparing the dataloader for the experiment
            # exp_df = get_perf_scores_df(experiment_name)
            config_exp = Config(experiment_name)
            dataloader_exp = HDF5DatasetSDD(config_exp, log=None, split='test')

            # # investigating high OAO / OAC_t0 ratios
            # print(f"{exp_df['OAO']=}")
            # print(f"{exp_df['OAC_t0']=}")
            # mask = exp_df['OAO'] > exp_df['OAC_t0']
            # exp_df = exp_df[mask & exp_df['OAC_t0'] != 0.]
            # exp_df['OAO_by_OAC_t0'] = exp_df['OAO'] / exp_df['OAC_t0']
            # print(exp_df.sort_values('OAO_by_OAC_t0', ascending=False)[['OAO', 'OAC_t0', 'OAO_by_OAC_t0']])

            # retrieve the corresponding entry name and dataset index
            instance_name = f"{instance_number}".rjust(8, '0')
            instance_index = dataloader_exp.get_instance_idx(instance_num=instance_number)

            # mini_df = exp_df.loc[instance_number, instance_number, :]
            # mini_df = remove_k_sample_columns(mini_df)
            # print(f"Instance Dataframe:\n{mini_df}")
            show_agent_pred = []

            # preparing the figure
            fig, ax = plt.subplots(figsize=figsize)
            fig.canvas.manager.set_window_title(f"{experiment_name}: (instance nr {instance_name})")

            checkpoint_name = config_exp.get_best_val_checkpoint_name()
            saved_preds_dir = os.path.join(
                config_exp.result_dir, dataloader_exp.dataset_name, checkpoint_name, 'test'
            )

            # retrieve the input data dict
            input_dict = dataloader_exp.__getitem__(instance_index)
            if 'map_homography' not in input_dict.keys():
                input_dict['map_homography'] = dataloader_exp.map_homography

            # retrieve the prediction data dict
            pred_file = os.path.join(saved_preds_dir, instance_name)
            print(f"{pred_file=}")
            assert os.path.exists(pred_file)
            with open(pred_file, 'rb') as f:
                pred_dict = pickle.load(f)
            pred_dict['map_homography'] = input_dict['map_homography']

            visualize_input_and_predictions(
                draw_ax=ax,
                data_dict=input_dict,
                pred_dict=pred_dict,
                show_rgb_map=True,
                show_gt_agent_ids=show_pred_ids,
                show_obs_agent_ids=show_pred_ids,
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
    if False:
        # uninteresting results, no matter the map score / trajectory score combination
        # actually, maybe a (very slight trend)
        base_experiment = OCCLUSIONFORMER_NO_MAP
        compare_experiment = OCCLUSIONFORMER_WITH_OCCL_MAP
        x_score = 'OAC_t0'
        y_score = 'min_FDE'

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
        ax.set_xlabel(f"Delta {x_score}")
        ax.set_ylabel(f"Delta {y_score}")
        ax.set_title(f"{compare_experiment} vs {base_experiment}")
        plt.show()

        fig, ax = plt.subplots(4, 3)
        for i, y_score in enumerate(['min_FDE', 'min_ADE', 'mean_FDE', 'mean_ADE']):
            for j, x_score in enumerate(['OAO', 'OAC', 'OAC_t0']):
                xs = diff_df[x_score].to_numpy()
                ys = diff_df[y_score].to_numpy()
                ax[i, j].scatter(xs, ys, marker='x', color='red', alpha=0.3)
                if j == 0:
                    ax[i, j].set_ylabel(f"Delta {y_score}")
                if i == 3:
                    ax[i, j].set_xlabel(f"Delta {x_score}")
                # ax[i, j].set_title(f"{compare_experiment} vs {base_experiment}")
                ax[i, j].grid(axis='y')
        fig.suptitle(f"{compare_experiment} vs {base_experiment}")
        plt.show()

    if False:
        experiment_name = OCCLUSIONFORMER_NO_MAP
        experiment_name = OCCLUSIONFORMER_WITH_OCCL_MAP
        x_score = 'OAC_t0'
        y_score = 'min_FDE'

        exp_df = get_perf_scores_df(experiment_name)
        exp_df = exp_df[exp_df['past_pred_length'] != 0]
        exp_df = remove_k_sample_columns(exp_df)

        # WIP WIP WIP
        # exp_df = exp_df.sort_values('occlusion_area', ascending=True)
        # print(f"{exp_df['occlusion_area']=}")
        # print(exp_df.keys())
        # print(zblu)

        xs = exp_df[x_score].to_numpy()
        ys = exp_df[y_score].to_numpy()

        fig, ax = plt.subplots()
        ax.scatter(xs, ys, marker='x', color='red', alpha=0.3)
        ax.set_xlabel(f"{x_score}")
        ax.set_ylabel(f"{y_score}")
        ax.set_title(f"{experiment_name}")
        plt.show()

        fig, ax = plt.subplots(4, 3)
        for i, (y_score, vmax_value) in enumerate(zip(
                ['min_FDE', 'min_ADE', 'mean_FDE', 'mean_ADE'],
                [3200, 3200, 2200, 1850]
        )):
            for j, x_score in enumerate(['OAO', 'OAC', 'OAC_t0']):

                xs = exp_df[x_score].to_numpy()
                ys = exp_df[y_score].to_numpy()

                isnan_mask = np.isnan(xs)

                xs = xs[~isnan_mask]
                ys = ys[~isnan_mask]

                if False:
                    # scatter plot
                    ax[i, j].scatter(xs, ys, marker='x', color='red', alpha=0.3)
                else:
                    H, yedges, xedges = np.histogram2d(ys, xs, bins=[50, 20])
                    im = ax[i, j].pcolormesh(xedges, yedges, H, cmap='rainbow', vmin=0.0, vmax=vmax_value)

                    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax[i, j])
                    cax = divider.append_axes('right', size='2%', pad=0.02)

                    fig.colorbar(im, cax=cax, orientation='vertical')

                if j == 0:
                    ax[i, j].set_ylabel(f"{y_score}")
                if i == 3:
                    ax[i, j].set_xlabel(f"{x_score}")
                # ax[i, j].set_title(f"{compare_experiment} vs {base_experiment}")
                ax[i, j].grid(axis='y')
        fig.suptitle(f"{experiment_name}")
        plt.show()

    if False:
        import copy
        # OAO / OAC / OAC_t0 correlation
        experiment_name = OCCLUSIONFORMER_WITH_OCCL_MAP
        experiment_name = OCCLUSIONFORMER_NO_MAP
        mode = 'scatter'        # 'scatter' | 'heatmap'

        my_cmap = copy.copy(matplotlib.colormaps['rainbow'])  # copy the default cmap
        my_cmap.set_bad((0, 0, 0))

        hist_bins = 20
        d_bin = 1/(hist_bins-1)
        hist_range = np.array([0-d_bin/2, 1+d_bin/2])
        hist_range = [hist_range, hist_range]

        scores = ['OAO', 'OAC', 'OAC_t0']

        exp_df = get_perf_scores_df(experiment_name)
        exp_df = exp_df[exp_df['past_pred_length'] != 0]
        exp_df = remove_k_sample_columns(exp_df)

        fig, ax = plt.subplots(len(scores), len(scores))

        for i, x_score in enumerate(scores):
            for j, y_score in enumerate(scores):
                xs = exp_df[x_score].to_numpy()
                ys = exp_df[y_score].to_numpy()

                isnan_mask = np.logical_or(np.isnan(xs), np.isnan(ys))
                is_perfect_mask = np.logical_and(xs == 1.0, ys == 1.0)
                mask = np.logical_or(isnan_mask, is_perfect_mask)

                xs = xs[~isnan_mask]
                ys = ys[~isnan_mask]

                if mode == 'scatter':
                    ax[i, j].scatter(xs, ys, marker='x', color='red', alpha=0.1)
                elif mode == 'heatmap':
                    h, xedges, yedges, im = ax[i, j].hist2d(
                        xs, ys,
                        bins=hist_bins, range=hist_range, cmap=my_cmap, norm='log'
                    )

                    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax[i, j])
                    cax = divider.append_axes('right', size='2%', pad=0.02)
                    fig.colorbar(im, cax=cax, orientation='vertical')

                ax[i, j].set_aspect('equal', 'box')
                ax[i, j].set_xlabel(f"{x_score}")
                ax[i, j].set_ylabel(f"{y_score}")
        fig.suptitle(experiment_name)
        plt.show()

    if True:
        # evaluating insightfulness of the occlusion map between
        #   - occlusionformer_no_map
        #   - occlusionformer_with_occl_map
        # we are particularly interested in instances where a CV model performs poorly

        COMPARE_SCORES = OCCLUSION_MAP_SCORES
        COMPARE_SCORES = PAST_ADE_SCORES + PAST_FDE_SCORES
        COMPARE_SCORES = ADE_SCORES + FDE_SCORES
        cv_model_name = CONST_VEL_OCCLUSION_SIMULATION
        no_map_name = OCCLUSIONFORMER_NO_MAP
        yes_map_name = OCCLUSIONFORMER_WITH_OCCL_MAP
        experiment_names = [yes_map_name, no_map_name, cv_model_name]

        perf_summary = generate_performance_summary_df(
            experiment_names=experiment_names, metric_names=DISTANCE_METRICS+PRED_LENGTHS+OCCLUSION_MAP_SCORES
        )
        perf_summary.sort_values(by='min_FDE', inplace=True)

        print(f"Performance summary:")
        print(perf_summary)

        cv_perf_df = get_perf_scores_df(experiment_name=cv_model_name)
        cv_perf_df = cv_perf_df[cv_perf_df['past_pred_length'] > 0]
        no_map_perf_df = get_perf_scores_df(experiment_name=no_map_name)
        no_map_perf_df = no_map_perf_df[no_map_perf_df['past_pred_length'] > 0]
        yes_map_perf_df = get_perf_scores_df(experiment_name=yes_map_name)
        yes_map_perf_df = yes_map_perf_df[yes_map_perf_df['past_pred_length'] > 0]

        assert all(cv_perf_df.index == no_map_perf_df.index)
        assert all(cv_perf_df.index == yes_map_perf_df.index)

        failed_cv_oac_t0 = (cv_perf_df['OAC_t0'] == 0.)
        print(f"Out of all {len(failed_cv_oac_t0)} occluded cases,\n"
              f"the constant velocity model misplaced the current position of the agent as being "
              f"outside the occluded zone "
              f"{sum(failed_cv_oac_t0)} times ({sum(failed_cv_oac_t0)/len(failed_cv_oac_t0)*100:.2f}%)")

        cv_perf_df = cv_perf_df[failed_cv_oac_t0]
        no_map_perf_df = no_map_perf_df[failed_cv_oac_t0]
        yes_map_perf_df = yes_map_perf_df[failed_cv_oac_t0]

        # sample_rows = np.random.choice(cv_perf_df.index, 5, replace=False)
        # [print(sample) for sample in sample_rows]

        fig, ax = plt.subplots()
        boxplot_dict = {score_name: None for score_name in COMPARE_SCORES}
        for score_name in COMPARE_SCORES:
            diff = (yes_map_perf_df[score_name] - no_map_perf_df[score_name]).to_numpy()
            boxplot_dict[score_name] = diff

        ax.axhline(0, linestyle='--', color='k', alpha=0.1)  # horizontal line through y=0

        box_plot_xs = []
        box_plot_ys = []
        for k, v in boxplot_dict.items():
            box_plot_xs.append(k)
            box_plot_ys.append(v)

        bplot = ax.boxplot(box_plot_ys, positions=(range(len(box_plot_ys))))
        ax.set_xticklabels(box_plot_xs)

        plt.show()

    if args.comp_phase_1_2:
        # checking the performance difference between phase 1 (model) and phase 2 (model + Dlow for diversity sampling)
        experiment_list = [
            OCCLUSIONFORMER_CAUSAL_ATTENTION_FULLY_OBSERVED_314,
            BASELINE_NO_POS_CONCAT,
            OCCLUSIONFORMER_CAUSAL_ATTENTION,
            OCCLUSIONFORMER_MOMENTARY,
            SDD_BASELINE_OCCLUSIONFORMER,
            OCCLUSIONFORMER_WITH_OCCL_MAP,
            OCCLUSIONFORMER_NO_MAP,
            OCCLUSIONFORMER_WITH_OCCL_MAP_IMPUTED,
            OCCLUSIONFORMER_IMPUTED,
            OCCLUSIONFORMER_CAUSAL_ATTENTION_FULLY_OBSERVED_162,
            OCCLUSIONFORMER_CAUSAL_ATTENTION_FULLY_OBSERVED,
        ]

        metric_names = ADE_SCORES + FDE_SCORES

        # list of experiment names (phase 1 and phase 2)
        phase_1_list = [f"{name}_pre" for name in experiment_list]
        phase_2_list = experiment_list

        # extracting the performance summaries
        perf_df_1 = generate_performance_summary_df(experiment_names=phase_1_list, metric_names=metric_names)
        perf_df_2 = generate_performance_summary_df(experiment_names=phase_2_list, metric_names=metric_names)

        experiment_id_columns = ['experiment', 'dataset_used']

        diff_df = perf_df_2[experiment_id_columns].copy()
        for metric_name in metric_names:
            diff_df[f"P1:{metric_name}"] = perf_df_1[metric_name]
            diff_df[f"P2:{metric_name}"] = perf_df_2[metric_name]
            diff_df[f"Delta:{metric_name}"] = perf_df_2[metric_name] - perf_df_1[metric_name]

        if SHOW:
            print(f"{diff_df}")
        if SAVE:
            print(f"Sorry, saving functionality not implemented for this performance evaluation...")
            raise NotImplementedError


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

