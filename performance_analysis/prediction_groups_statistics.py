import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path
import scipy.stats

import pandas.core.series
from typing import List

from utils.performance_analysis import \
    SCORES_CSV_FILENAME, \
    get_all_pred_scores_csv_files, \
    process_analysis_subjects_txt, \
    get_df_filter, \
    get_perf_scores_df, \
    get_reference_indices, \
    get_scores_dict_by_categories, \
    remove_k_sample_columns, \
    get_experiment_dict, \
    get_df_from_csv, \
    STATISTICAL_OPERATIONS

DEFAULT_SCORES = [
    'min_ADE', 'min_FDE',
    'mean_ADE', 'mean_FDE',
    'min_past_ADE', 'min_past_FDE',
    'mean_past_ADE', 'mean_past_FDE',
]
OPERATIONS = [
    'mean',
    # 'median',
    'IQR',
]


def scores_stats_df_per_occlusion_lengths(
        exp_csv_path: str,
        scores: List[str],
        operations: List[str],
        categorization: pandas.core.series.Series,
        df_filter=None
) -> pd.DataFrame:
    assert all([operation in STATISTICAL_OPERATIONS.keys() for operation in operations])

    print(f"categorization counts (total: {len(categorization)}):\n{categorization.value_counts()}")
    category_name, category_values = categorization.name, sorted(categorization.unique())

    experiment_df = get_df_from_csv(file_path=exp_csv_path)
    if df_filter is not None:
        experiment_df = df_filter(experiment_df)

    experiment_df = remove_k_sample_columns(df=experiment_df)

    experiment_df = experiment_df.iloc[experiment_df.index.isin(categorization.index)]

    if category_name not in experiment_df.columns:
        experiment_df[category_name] = categorization

    assert all([score in experiment_df.columns for score in scores])
    assert category_name in experiment_df.columns

    df_index = pd.MultiIndex.from_product([operations, scores], names=['operation', 'score'])
    out_df = pd.DataFrame(index=df_index, columns=category_values)

    for score in scores:
        scores_dict = get_scores_dict_by_categories(
            exp_df=experiment_df,
            score=score,
            categorization=category_name
        )
        for operation in operations:
            row_dict = {key: STATISTICAL_OPERATIONS[operation](value) for key, value in scores_dict.items()}
            out_df.loc[(operation, score)] = row_dict

    return out_df


def main(args: argparse.Namespace):
    print("PERFORMANCE STATISTICS BY LAST OBSERVED TIMESTEP GROUPS\n\n")

    if args.score_files is None:
        # search for every single prediction_scores.csv file and return them all
        subjects = get_all_pred_scores_csv_files()
    elif len(args.score_files) == 1 and args.score_files[0].endswith('.txt'):
        subjects = process_analysis_subjects_txt(txt_file=args.score_files[0])
    else:
        subjects = [file for file in args.score_files if (not file.startswith('#') and file.endswith(SCORES_CSV_FILENAME))]

    for file in subjects:
        assert os.path.exists(file), f"Error, file does not exist:\n{file}"

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

    for exp_csv_path in subjects:

        exp_dict = get_experiment_dict(file_path=exp_csv_path)
        print(f"{exp_dict['experiment_name']}:\n")

        summary_df = scores_stats_df_per_occlusion_lengths(
            exp_csv_path=exp_csv_path,
            scores=DEFAULT_SCORES,
            operations=OPERATIONS,
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


if __name__ == '__main__':
    print("Hello!")
    parser = argparse.ArgumentParser()
    parser.add_argument('--score_files', nargs='+', type=os.path.abspath, default=None,
                        help="provide either multiple paths to 'prediction_scores.csv' files inside the 'results' "
                             "directory, or a path to a single .txt file, containing paths to those files "
                             "(relative to the repository's root directory). If nothing is passed, the script will "
                             "search for every single 'prediction_scores.csv' file inside the 'results' directory.")
    parser.add_argument('--filter', nargs='+', type=str, default=None,
                        help="select any number of options from:\n"
                             "\'occluded_ids\', \'fully_observed_ids\', \'difficult_dataset\', "
                             "\'difficult_occluded_ids\', \'moving\', \'idle\'")
    args = parser.parse_args()

    main(args=args)
    print("Goodbye!")
