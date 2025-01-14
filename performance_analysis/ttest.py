import argparse
import numpy as np
import os.path
import pandas as pd
import scipy.stats

from utils.performance_analysis import \
    SCORES_CSV_FILENAME, \
    MIN_SCORES, MEAN_SCORES, PAST_MIN_SCORES, PAST_MEAN_SCORES, \
    get_reference_indices, \
    get_df_filter, \
    get_perf_scores_df, \
    process_analysis_subjects_txt, \
    get_experiment_dict, \
    get_df_from_csv


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 75)

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
    assert args.unit in ['m', 'px']
    print("T-TESTS:\n\n")

    if args.unit == 'm':
        min_scores, mean_scores = MIN_SCORES, MEAN_SCORES
        past_min_scores, past_mean_scores = PAST_MIN_SCORES, PAST_MEAN_SCORES
    elif args.unit == 'px':
        px_name = lambda names_list: [f'{score_name}_px' for score_name in names_list]
        min_scores, mean_scores = px_name(MIN_SCORES), px_name(MEAN_SCORES)
        past_min_scores, past_mean_scores = px_name(PAST_MIN_SCORES), px_name(PAST_MEAN_SCORES)
    else:
        raise NotImplementedError

    if len(args.score_files) == 1 and args.score_files[0].endswith('.txt'):
        subjects = process_analysis_subjects_txt(txt_file=args.score_files[0])
    else:
        subjects = [file for file in args.score_files if (not file.startswith('#') and file.endswith(SCORES_CSV_FILENAME))]

    for file in subjects:
        assert os.path.exists(file), f"Error, file does not exist:\n{file}"

    test_scores = min_scores + mean_scores

    ref_index = get_reference_indices()
    ref_past_pred_lengths = get_perf_scores_df(
        experiment_name='CV_predictor',
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

        for exp_csv_path in subjects:

            exp_dict = get_experiment_dict(exp_csv_path)
            experiment_df = get_df_from_csv(exp_csv_path)

            if df_filter is not None:
                experiment_df = df_filter(experiment_df)

            for test_score in test_scores:

                scores_1 = experiment_df.iloc[experiment_df.index.isin(category_1_index)][test_score]
                scores_2 = experiment_df.iloc[experiment_df.index.isin(category_2_index)][test_score]
                assert all(scores_1.index == category_1_index)
                assert all(scores_2.index == category_2_index)

                scores_1 = scores_1.values
                scores_2 = scores_2.values

                is_significant, t_stat, p_val, df, t_crit = perform_ttest(array_a=scores_1, array_b=scores_2)

                row_dict = {
                    'experiment': exp_dict['experiment_name'],
                    'dataset_used': exp_dict['dataset_used'],
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

        comparisons = [(0, cat_2) for cat_2 in range(1, 7)]
        def comp_name(comp): return f"-{comp[0]}/-{comp[1]}"

        row_index_tuples = []
        for exp_csv_path in subjects:
            exp_dict = get_experiment_dict(exp_csv_path)
            row_index_tuples.append((exp_dict['experiment_name'], exp_dict['dataset_used']))
        row_df_index = pd.MultiIndex.from_tuples(row_index_tuples, names=['experiment', 'dataset_used'])
        col_index_tuples = [(test_score, comp_name(comp)) for test_score in test_scores for comp in comparisons]
        col_df_index = pd.MultiIndex.from_tuples(col_index_tuples, names=['test_score', 'categories'])

        out_df = pd.DataFrame(index=row_df_index, columns=col_df_index)

        for exp_csv_path in subjects:
            exp_dict = get_experiment_dict(exp_csv_path)
            experiment_df = get_df_from_csv(exp_csv_path)

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

                    out_df.loc[
                        (exp_dict['experiment_name'], exp_dict['dataset_used']), (test_score, comp_name(comp))
                    ] = is_significant

        print(out_df)


if __name__ == '__main__':
    print("Hello!")
    parser = argparse.ArgumentParser()
    parser.add_argument('--score_files', nargs='+', type=os.path.abspath, required=True,
                        help="provide either multiple paths to 'prediction_scores.csv' files inside the 'results' "
                             "directory, or a path to a single .txt file, containing paths to those files "
                             "(relative to the repository's root directory).")
    parser.add_argument('--unit', type=str, default='m',
                        help="\'m\' | \'px\'")
    parser.add_argument('--filter', nargs='+', type=str, default=None,
                        help="select any number of options from:\n"
                             "\'occluded_ids\', \'fully_observed_ids\', \'difficult_dataset\', "
                             "\'difficult_occluded_ids\', \'moving\', \'idle\'")
    parser.add_argument('--comp_tlast', nargs=2, type=int, default=None,
                        help='specify 2 last observed timestep categories to compare')
    args = parser.parse_args()

    main(args=args)
    print("Goodbye!")
