import argparse
import numpy as np
import pandas as pd
import scipy.stats

from utils.performance_analysis import \
    get_reference_indices, \
    get_all_results_directories, \
    get_df_filter, \
    get_perf_scores_df


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 75)

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


def perform_ttest(array_a: np.array, array_b: np.array, alpha=0.05):
    # https://en.wikipedia.org/wiki/Welch%27s_t-test
    # corrected sample stdev
    def c_std(a): return a.std(ddof=1)

    # standard error
    def s_x(a): return c_std(a) / np.sqrt(len(a))

    # degrees of freedom
    def dof(a, b): return (s_x(a) ** 2 + s_x(b) ** 2) ** 2 / \
                          (s_x(a) ** 4 / (len(a) - 1) + s_x(b) ** 4 / (len(b) - 1))

    # https://www.geeksforgeeks.org/how-to-find-the-t-critical-value-in-python/
    # two-tailed T-critical value
    def twotailed_t_critical(alpha, df): return scipy.stats.t.ppf(q=1 - alpha / 2, df=df)

    t_test = scipy.stats.ttest_ind(a=array_a, b=array_b, equal_var=False)
    df = dof(array_a, array_b)
    t_crit = twotailed_t_critical(alpha=alpha, df=df)

    is_significant = np.abs(t_test.statistic) > np.abs(t_crit)

    return is_significant, t_test.statistic, t_test.pvalue, df, t_crit


def main(args: argparse.Namespace):
    if args.unit == 'm':
        min_scores, mean_scores = MIN_SCORES, MEAN_SCORES
        past_min_scores, past_mean_scores = PAST_MIN_SCORES, PAST_MEAN_SCORES
    elif args.unit == 'px':
        px_name = lambda names_list: [f'{score_name}_px' for score_name in names_list]
        min_scores, mean_scores = px_name(MIN_SCORES), px_name(MEAN_SCORES)
        past_min_scores, past_mean_scores = px_name(PAST_MIN_SCORES), px_name(PAST_MEAN_SCORES)
    else:
        raise NotImplementedError

    print("T-TESTS:\n\n")
    assert args.cfg is not None
    experiment_names = args.cfg

    exp_dicts = get_all_results_directories()
    exp_dicts = [exp_dict for exp_dict in exp_dicts if exp_dict['split'] in ['test']]
    exp_dicts = [exp_dict for exp_dict in exp_dicts if exp_dict['experiment_name'] in experiment_names]
    # exp_dicts = [exp_dict for exp_dict in exp_dicts if exp_dict['dataset_used'] in ['fully_observed']]

    test_scores = min_scores + mean_scores

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

    if args.comp_tlast is not None:

        tlast_1, tlast_2 = args.comp_tlast
        out_df_columns = [
            'experiment', 'dataset_used', 'test_score', 'n_1', 'mean_1', 's^2_1', 'n_2', 'mean_2', 's^2_2',
            't_stat', 'df', 'p_value', 't_critical', 'significant'
        ]

        category_1_index = ref_past_pred_lengths[ref_past_pred_lengths == tlast_1].index
        category_2_index = ref_past_pred_lengths[ref_past_pred_lengths == tlast_2].index

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
              f"the following set of t-tests are ({tlast_1}, {tlast_2}):\n")

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


if __name__ == '__main__':
    print("Hello!")
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', nargs='+', default=None)
    parser.add_argument('--filter', nargs='+', default=None)
    parser.add_argument('--comp_tlast', nargs=2,
                        help='specify 2 last observed timestep categories to compare', type=int, default=None)
    parser.add_argument('--unit', type=str, default='m')        # 'm' | 'px'
    args = parser.parse_args()

    main(args=args)
    print("Goodbye!")
