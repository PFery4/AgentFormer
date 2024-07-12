import argparse
import os.path
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import pandas as pd
from utils.config import Config, REPO_ROOT
from data.sdd_dataloader import HDF5DatasetSDD
from utils.sdd_visualize import visualize_input_and_predictions
from utils.performance_analysis import \
    generate_performance_summary_df, \
    pretty_print_difference_summary_df, \
    make_oac_histograms_figure, \
    get_perf_scores_df, \
    get_reference_indices, \
    get_all_results_directories, \
    get_df_filter, \
    remove_k_sample_columns, \
    scores_stats_df_per_occlusion_lengths


if __name__ == '__main__':
    # Script Controls #################################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', nargs='+', default=None)
    parser.add_argument('--perf_summary', action='store_true', default=False)
    parser.add_argument('--unit', type=str, default='m')        # 'm' | 'px'
    parser.add_argument('--sort_by', type=str, default='experiment')
    parser.add_argument('--filter', nargs='+', default=None)
    parser.add_argument('--boxplots', action='store_true', default=False)
    parser.add_argument('--stats_per_last_obs', action='store_true', default=False)
    parser.add_argument('--ttest', nargs='*', help='specify 0 or 2 arguments', type=int, default=None)
    parser.add_argument('--oac_histograms', action='store_true', default=False)
    parser.add_argument('--qual_compare', action='store_true', default=False)
    parser.add_argument('--instance_num', type=int, default=None)
    parser.add_argument('--identities', nargs='+', type=int, default=[])
    parser.add_argument('--qual_example', action='store_true', default=False)
    parser.add_argument('--comp_phase_1_2', action='store_true', default=False)
    parser.add_argument('--cv_wrong_map_rate', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--show', action='store_true', default=False)
    args = parser.parse_args()

    SHOW = args.show
    SAVE = args.save

    assert SAVE or SHOW
    assert args.unit in ['m', 'px']

    # Global Variables set up #########################################################################################
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 75)

    PERFORMANCE_ANALYSIS_DIRECTORY = os.path.join(REPO_ROOT, 'performance_analysis')
    os.makedirs(PERFORMANCE_ANALYSIS_DIRECTORY, exist_ok=True)

    UNIT = args.unit       # 'm' | 'px'
    EXPERIMENT_SEPARATOR = "\n\n\n\n" + "#" * 200 + "\n\n\n\n"

    DEFAULT_CFG = [
        'original_100',
        'original_101',
        'original_102',
        'original_103',
        'original_104',
        'occlusionformer_basis_bias_1',
        'occlusionformer_basis_bias_2',
        'occlusionformer_basis_bias_3',
        'occlusionformer_basis_bias_4',
        'occlusionformer_basis_bias_5',
    ]

    MIN_SCORES = ['min_ADE', 'min_FDE']
    MEAN_SCORES = ['mean_ADE', 'mean_FDE']
    PAST_MIN_SCORES = ['min_past_ADE', 'min_past_FDE']
    PAST_MEAN_SCORES = ['mean_past_ADE', 'mean_past_FDE']

    PRED_LENGTHS = ['past_pred_length', 'pred_length']
    OCCLUSION_MAP_SCORES = ['OAO', 'OAC', 'OAC_t0']

    if UNIT == 'px':

        px_name = lambda names_list: [f'{score_name}_px' for score_name in names_list]
        MIN_SCORES = px_name(MIN_SCORES)
        MEAN_SCORES = px_name(MEAN_SCORES)
        PAST_MIN_SCORES = px_name(PAST_MIN_SCORES)
        PAST_MEAN_SCORES = px_name(PAST_MEAN_SCORES)

    # Performance Summary #############################################################################################
    if args.perf_summary:
        raise NotImplementedError
        print("\n\nPERFORMANCE SUMMARY:\n\n")

        experiment_names = args.cfg if args.cfg is not None else DEFAULT_CFG
        operation = 'mean'          # 'mean' | 'median' | 'IQR'
        sort_by = args.sort_by

        metric_names = (MIN_SCORES + MEAN_SCORES +
                        PAST_MIN_SCORES + PAST_MEAN_SCORES +
                        PRED_LENGTHS + OCCLUSION_MAP_SCORES)

        ref_index = get_reference_indices()

        exp_dicts = get_all_results_directories()
        exp_dicts = [exp_dict for exp_dict in exp_dicts if exp_dict['split'] in 'test']
        exp_dicts = [exp_dict for exp_dict in exp_dicts if exp_dict['dataset_used'] not in ['occlusion_simulation_difficult']]
        exp_dicts = [exp_dict for exp_dict in exp_dicts if exp_dict['experiment_name'] in experiment_names]

        df_filter = get_df_filter(ref_index=ref_index, filters=args.filter)

        all_perf_df = generate_performance_summary_df(
            experiments=exp_dicts, metric_names=metric_names, operation=operation, df_filter=df_filter
        )
        all_perf_df.sort_values(by=sort_by, inplace=True)

        if SHOW:

            if True:  # specify some dataset characteristics
                example_df = get_perf_scores_df(
                    experiment_name='const_vel_occlusion_simulation',
                    dataset_used='occlusion_simulation',
                    model_name='untrained',
                    split='test'
                )
                example_df = df_filter(example_df)

                print(f"Dataset Characteristics:")
                print(f"# instances\t\t: {len(example_df.index.unique(level='filename'))}")
                print(f"# trajectories\t\t: {len(example_df)}")
                print(f"# occlusion cases\t: {(example_df['past_pred_length'] != 0).sum()}")
                print("\n")

            print(f"Experiments Performance Summary ({operation}):")
            print(all_perf_df)

        if SAVE:
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

    # qualitative display of predictions ##############################################################################
    if args.qual_example:
        raise NotImplementedError
        print("\n\nQUALITATIVE EXAMPLE:\n\n")

        experiment_names = args.cfg if args.cfg is not None else DEFAULT_CFG
        assert args.instance_num is not None

        instance_number, show_pred_ids = args.instance_num, args.identities     # int, List[int]

        imputed = False
        highlight_only_past_pred = False
        figsize = (14, 10)
        figdpi = 300

        # TODO: SWITCH TO EXPERIMENT DICTS INSTEAD
        for experiment_name in experiment_names:
            # preparing the dataloader for the experiment
            # exp_df = get_perf_scores_df(experiment_name)
            config_exp = Config(experiment_name)
            dataloader_exp = HDF5DatasetSDD(config_exp, log=None, split='test') if not imputed else \
                HDF5DatasetSDD(Config('const_vel_occlusion_simulation_imputed'), log=None, split='test')

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
            fig.set_dpi(figdpi)
            fig.canvas.manager.set_window_title(f"{experiment_name}_{instance_name}")

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
                show_obs_agent_ids=None,
                show_pred_agent_ids=show_pred_ids,
                past_pred_alpha=0.5,
                future_pred_alpha=0.1 if highlight_only_past_pred else 0.5
            )
            # ax.legend()
            ax.set_title(experiment_name)
            fig.subplots_adjust(wspace=0.10, hspace=0.0)
        plt.show()

        print(EXPERIMENT_SEPARATOR)

    # t-tests of score diff dependent on last observed timestep categories ############################################
    if args.ttest is not None:
        raise NotImplementedError
        print("\n\nT-TESTS:\n\n")
        assert len(args.ttest) in {0, 2}
        assert args.cfg is not None
        experiment_names = args.cfg

        exp_dicts = get_all_results_directories()
        exp_dicts = [exp_dict for exp_dict in exp_dicts if exp_dict['split'] in ['test']]
        exp_dicts = [exp_dict for exp_dict in exp_dicts if exp_dict['experiment_name'] in experiment_names]
        # exp_dicts = [exp_dict for exp_dict in exp_dicts if exp_dict['dataset_used'] in ['fully_observed']]

        test_scores = MIN_SCORES + MEAN_SCORES

        ref_index = get_reference_indices()
        ref_past_pred_lengths = get_perf_scores_df(
            experiment_name='const_vel_occlusion_simulation',
            dataset_used='occlusion_simulation',
            model_name='untrained',
            split='test'
        )
        ref_past_pred_lengths = ref_past_pred_lengths.iloc[ref_past_pred_lengths.index.isin(ref_index)]
        ref_past_pred_lengths = ref_past_pred_lengths['past_pred_length']

        df_filter = get_df_filter(ref_index=ref_index, filters=args.filter)
        ref_past_pred_lengths = df_filter(ref_past_pred_lengths)

        if len(args.ttest) == 2:

            category_1, category_2 = args.ttest
            out_df_columns = [
                'experiment', 'dataset_used', 'test_score', 'n_1', 'mean_1', 's^2_1', 'n_2', 'mean_2', 's^2_2',
                't_stat', 'df', 'p_value', 't_critical', 'significant'
            ]

            category_1_index = ref_past_pred_lengths[ref_past_pred_lengths == category_1].index
            category_2_index = ref_past_pred_lengths[ref_past_pred_lengths == category_2].index

            out_df = pd.DataFrame(columns=out_df_columns)

            for exp_dict in exp_dicts:

                exp_name = exp_dict['experiment_name']
                dataset_used = exp_dict['dataset_used']
                model_name = exp_dict['model_name']
                split = exp_dict['split']

                experiment_df = get_perf_scores_df(
                    experiment_name=exp_name,
                    dataset_used=dataset_used,
                    model_name=model_name,
                    split=split
                )
                if df_filter is not None:
                    experiment_df = df_filter(experiment_df)

                for test_score in test_scores:

                    scores_1 = experiment_df.iloc[experiment_df.index.isin(category_1_index)][test_score]
                    scores_2 = experiment_df.iloc[experiment_df.index.isin(category_2_index)][test_score]
                    assert all(scores_1.index == category_1_index)
                    assert all(scores_2.index == category_2_index)

                    scores_1 = scores_1.values
                    scores_2 = scores_2.values

                    # import numpy as np
                    # example A:
                    # scores_1 = np.array([27.5, 21.0, 19.0, 23.6, 17.0, 17.9, 16.9, 20.1, 21.9, 22.6, 23.1, 19.6, 19.0, 21.7, 21.4])
                    # scores_2 = np.array([27.1, 22.0, 20.8, 23.4, 23.4, 23.5, 25.8, 22.0, 24.8, 20.2, 21.9, 22.1, 22.9, 20.5, 24.4])
                    # example B:
                    # scores_1 = np.array([17.2, 20.9, 22.6, 18.1, 21.7, 21.4, 23.5, 24.2, 14.7, 21.8])
                    # scores_2 = np.array([21.5, 22.8, 21.0, 23.0, 21.6, 23.6, 22.5, 20.7, 23.4, 21.8, 20.7, 21.7, 21.5, 22.5, 23.6, 21.5, 22.5, 23.5, 21.5, 21.8])
                    # example C:
                    # scores_1 = np.array([19.8, 20.4, 19.6, 17.8, 18.5, 18.9, 18.3, 18.9, 19.5, 22.0])
                    # scores_2 = np.array([28.2, 26.6, 20.1, 23.3, 25.2, 22.1, 17.7, 27.6, 20.6, 13.7, 23.2, 17.5, 20.6, 18.0, 23.9, 21.6, 24.3, 20.4, 24.0, 13.2])

                    is_significant, t_stat, p_val, df, t_crit = perform_ttest(array_a=scores_1, array_b=scores_2)

                    row_dict = {
                        'experiment': exp_name,
                        'dataset_used': dataset_used,
                        'n_1': len(scores_1),
                        'mean_1': scores_1.mean(),
                        's^2_1': scores_1.std(ddof=1)**2,
                        'n_2': len(scores_2),
                        'mean_2': scores_2.mean(),
                        's^2_2': scores_2.std(ddof=1)**2,
                        'test_score': test_score,
                        't_stat': t_stat,
                        'p_value': p_val,
                        'df': df,
                        't_critical': t_crit,
                        'significant': is_significant
                    }

                    out_df.loc[len(out_df)] = row_dict

            print(f"The goal of the t-tests is to evaluate whether performance metric scores differ significantly between "
                  f"different last-observation timestep categories.\nThe chosen last observation timestep categories for "
                  f"the following set of t-tests are ({category_1}, {category_2}):\n")

            for test_score in test_scores:
                print(f"{test_score}:")
                print(out_df[out_df['test_score'] == test_score])
                print("")

        else:

            # comparisons = [(1, cat_2) for cat_2 in range(2, 6)]
            comparisons = [(0, cat_2) for cat_2 in range(1, 7)]
            def comp_name(comp): return f"-{comp[0]}/-{comp[1]}"

            row_index_tuples = [(exp_dict['experiment_name'], exp_dict['dataset_used']) for exp_dict in exp_dicts]
            row_df_index = pd.MultiIndex.from_tuples(row_index_tuples, names=['experiment', 'dataset_used'])
            col_index_tuples = [(test_score, comp_name(comp)) for test_score in test_scores for comp in comparisons]
            col_df_index = pd.MultiIndex.from_tuples(col_index_tuples, names=['test_score', 'categories'])

            out_df = pd.DataFrame(index=row_df_index, columns=col_df_index)

            for exp_dict in exp_dicts:
                exp_name = exp_dict['experiment_name']
                dataset_used = exp_dict['dataset_used']
                model_name = exp_dict['model_name']
                split = exp_dict['split']

                experiment_df = get_perf_scores_df(
                    experiment_name=exp_name,
                    dataset_used=dataset_used,
                    model_name=model_name,
                    split=split
                )
                if df_filter is not None:
                    experiment_df = df_filter(experiment_df)

                for test_score in test_scores:

                    for comp in comparisons:
                        category_1, category_2 = comp

                        category_1_index = ref_past_pred_lengths[ref_past_pred_lengths == category_1].index
                        category_2_index = ref_past_pred_lengths[ref_past_pred_lengths == category_2].index

                        scores_1 = experiment_df.iloc[experiment_df.index.isin(category_1_index)][test_score]
                        scores_2 = experiment_df.iloc[experiment_df.index.isin(category_2_index)][test_score]
                        assert all(scores_1.index == category_1_index)
                        assert all(scores_2.index == category_2_index)

                        scores_1 = scores_1.values
                        scores_2 = scores_2.values

                        is_significant, t_stat, p_val, df, t_crit = perform_ttest(array_a=scores_1, array_b=scores_2)

                        out_df.loc[(exp_name, dataset_used), (test_score, comp_name(comp))] = is_significant

            print(out_df)

        print(EXPERIMENT_SEPARATOR)

    # score boxplots vs last observed timesteps #######################################################################
    if args.boxplots:
        raise NotImplementedError
        print("\n\nBOXPLOTS:\n\n")
        assert args.cfg is not None
        experiment_names = args.cfg

        exp_dicts = get_all_results_directories()
        exp_dicts = [exp_dict for exp_dict in exp_dicts if exp_dict['split'] in ['test']]
        exp_dicts = [exp_dict for exp_dict in exp_dicts if exp_dict['experiment_name'] in experiment_names]
        # exp_dicts = [exp_dict for exp_dict in exp_dicts if exp_dict['dataset_used'] not in ['fully_observed']]
        # exp_dicts = [exp_dict for exp_dict in exp_dicts if exp_dict['dataset_used'] in ['fully_observed']]

        boxplot_scores = [
            'min_ADE', 'min_FDE',
            'mean_ADE', 'mean_FDE',
            'min_past_ADE', 'min_past_FDE',
            'mean_past_ADE', 'mean_past_FDE'
        ]
        ylims = [
            (0.0, 11), (0.0, 11),
            (0.0, 37), (0.0, 37),
            (0.0, 5), (0.0, 5),
            (0.0, 9), (0.0, 9),
        ]

        figsize = (9, 6)

        boxplot_experiments_together = True
        boxplot_experiments_individually = False

        ref_index = get_reference_indices()
        ref_past_pred_lengths = get_perf_scores_df(
            experiment_name='const_vel_occlusion_simulation',
            dataset_used='occlusion_simulation',
            model_name='untrained',
            split='test'
        )
        ref_past_pred_lengths = ref_past_pred_lengths.iloc[ref_past_pred_lengths.index.isin(ref_index)]
        ref_past_pred_lengths = ref_past_pred_lengths['past_pred_length']

        df_filter = get_df_filter(ref_index=ref_index, filters=args.filter)
        ref_past_pred_lengths = df_filter(ref_past_pred_lengths)

        # print(f"{ref_past_pred_lengths[ref_past_pred_lengths==6]=}")
        if boxplot_experiments_together:
            for plot_score, ylim in zip(boxplot_scores, ylims):
                fig, ax = plt.subplots(figsize=figsize)
                make_box_plot_occlusion_lengths(
                    draw_ax=ax,
                    experiments=exp_dicts,
                    plot_score=plot_score,
                    categorization=ref_past_pred_lengths,
                    df_filter=df_filter,
                    ylim=ylim,
                    legend=False
                )
                ax.set_title(f"{plot_score} vs. Last Observed timestep")

                if SAVE:
                    boxplot_directory = os.path.join(PERFORMANCE_ANALYSIS_DIRECTORY, 'boxplots')
                    os.makedirs(boxplot_directory, exist_ok=True)

                    filename = f"{plot_score}.png"
                    filepath = os.path.join(boxplot_directory, filename)
                    print(f"saving boxplot to:\n{filepath}\n")
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')

        if boxplot_experiments_individually:
            for exp_dict in exp_dicts:
                for plot_score in boxplot_scores:
                    fig, ax = plt.subplots(figsize=figsize)
                    make_box_plot_occlusion_lengths(
                        draw_ax=ax,
                        experiments=[exp_dict],
                        plot_score=plot_score,
                        categorization=ref_past_pred_lengths
                    )
                    ax.set_title(f"{plot_score} vs. Last Observed timestep")

        if SHOW:
            plt.show()

        if SAVE:
            print("BOXPLOTS: no saving implementation (yet)!")

            # boxplot_directory = os.path.join(PERFORMANCE_ANALYSIS_DIRECTORY, 'boxplots')
            # os.makedirs(boxplot_directory, exist_ok=True)
            #
            # experiment_dir = os.path.join(boxplot_directory, <EXPERIMENT_NAME>)
            # os.makedirs(experiment_dir, exist_ok=True)
            #
            # filename = f"{<PLOT_SCORE>}.png"
            # filepath = os.path.join(experiment_dir, filename)
            # print(f"saving boxplot to:\n{filepath}\n")
            # plt.savefig(filepath, dpi=300, bbox_inches='tight')
            # plt.close()

        print(EXPERIMENT_SEPARATOR)

    if args.stats_per_last_obs:
        raise NotImplementedError
        print("\n\nPERFORMANCE STATISTICS BY LAST OBSERVED TIMESTEP GROUPS\n\n")
        assert args.cfg is not None
        experiment_names = args.cfg

        exp_dicts = get_all_results_directories()
        exp_dicts = [exp_dict for exp_dict in exp_dicts if exp_dict['split'] in ['test']]
        exp_dicts = [exp_dict for exp_dict in exp_dicts if exp_dict['experiment_name'] in experiment_names]

        scores = [
            'min_ADE', 'min_FDE',
            'mean_ADE', 'mean_FDE',
            'min_past_ADE', 'min_past_FDE',
            'mean_past_ADE', 'mean_past_FDE',
        ]
        operations = [
            'mean',
            # 'median',
            'IQR',
        ]

        ref_index = get_reference_indices()
        ref_past_pred_lengths = get_perf_scores_df(
            experiment_name='const_vel_occlusion_simulation',
            dataset_used='occlusion_simulation',
            model_name='untrained',
            split='test'
        )
        ref_past_pred_lengths = ref_past_pred_lengths.iloc[ref_past_pred_lengths.index.isin(ref_index)]
        ref_past_pred_lengths = ref_past_pred_lengths['past_pred_length']

        df_filter = get_df_filter(ref_index=ref_index, filters=args.filter)
        ref_past_pred_lengths = df_filter(ref_past_pred_lengths)

        for exp_dict in exp_dicts:

            print(f"{exp_dict['experiment_name']}:\n")

            summary_df = scores_stats_df_per_occlusion_lengths(
                exp_dict=exp_dict,
                scores=scores,
                operations=operations,
                categorization=ref_past_pred_lengths,
                df_filter=df_filter
            )
            summary_df.drop([0, 6], axis=1, inplace=True)
            print(summary_df)
            print("\n\n")

            fig, ax = plt.subplots()
            keep_op = 'mean'
            mini_df = summary_df.loc[([keep_op], slice(None)), :]

            lin_df = pd.DataFrame(index=mini_df.index, columns=['a', 'b', 'R^2', '@0', '@6', '@12'])

            for index, row in mini_df.iterrows():
                xs, ys = row.index.values.astype(float), row.values.astype(float)

                a, b, r_value, p_value, std_err = scipy.stats.linregress(xs, ys)

                lin_func = np.poly1d([a, b])
                lin_xs = np.array([0, 12])

                ax.plot(xs, ys, 'x', label=' '.join(index))
                ax.plot(lin_xs, lin_func(lin_xs), 'k--', alpha=0.5)

                lin_dict = {
                    'a': a,
                    'b': b,
                    'R^2': r_value**2,
                    '@0': lin_func(0),
                    '@6': lin_func(6),
                    '@12': lin_func(12)
                }
                lin_df.loc[index] = lin_dict

            print(lin_df)

            ax.legend()
            plt.show()

    # OAC histograms ##################################################################################################
    if args.oac_histograms:
        figsize = (16, 10)
        plot_score = 'OAC_t0'      # 'OAC', 'OAC_t0', 'OAO'
        as_percentage = False

        print(f"\n\n{plot_score} HISTOGRAMS:\n\n")

        experiment_names = args.cfg

        for experiment_name in experiment_names:
            fig = plt.figure(figsize=figsize)
            make_oac_histograms_figure(fig=fig, experiment_name=experiment_name, plot_score=plot_score)

        if SHOW:
            plt.show()

        if SAVE:
            print("OAC_histograms: no saving implementation (yet)!")

        print(EXPERIMENT_SEPARATOR)
