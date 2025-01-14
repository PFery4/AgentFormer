import argparse
import os.path
import pandas as pd

from utils.performance_analysis import \
    SCORES_CSV_FILENAME, \
    MIN_SCORES, MEAN_SCORES, PAST_MIN_SCORES, PAST_MEAN_SCORES, \
    get_all_pred_scores_csv_files, \
    get_reference_indices, \
    get_df_filter, \
    generate_performance_summary_df, \
    process_analysis_subjects_txt, \
    get_df_from_csv, \
    get_experiment_dict


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 75)

PRED_LENGTHS = ['past_pred_length', 'pred_length']
OCCLUSION_MAP_SCORES = ['OAO', 'OAC', 'OAC_t0']

OPERATION = 'mean'  # 'mean' | 'median' | 'IQR'


def main(args: argparse.Namespace):
    assert args.unit in ['m', 'px']
    print("PERFORMANCE SUMMARY:\n\n")

    if args.unit == 'm':
        min_scores, mean_scores = MIN_SCORES, MEAN_SCORES
        past_min_scores, past_mean_scores = PAST_MIN_SCORES, PAST_MEAN_SCORES
    elif args.unit == 'px':
        px_name = lambda names_list: [f'{score_name}_px' for score_name in names_list]
        min_scores, mean_scores = px_name(MIN_SCORES), px_name(MEAN_SCORES)
        past_min_scores, past_mean_scores = px_name(PAST_MIN_SCORES), px_name(PAST_MEAN_SCORES)
    else:
        raise NotImplementedError

    metric_names = (min_scores + mean_scores +
                    past_min_scores + past_mean_scores +
                    PRED_LENGTHS + OCCLUSION_MAP_SCORES)

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

    df_filter = get_df_filter(ref_index=ref_index, filters=args.filter)

    all_perf_df = generate_performance_summary_df(
        experiments=subjects, metric_names=metric_names, operation=OPERATION, df_filter=df_filter
    )
    all_perf_df.sort_values(by=args.sort_by, inplace=True)

    if args.print_dataset_stats:
        example_exp_dict = get_experiment_dict(file_path=subjects[0])
        example_df = get_df_from_csv(file_path=subjects[0])
        example_df = df_filter(example_df)

        print(f"Dataset Statistics:")
        print(f"# instances\t\t: {len(example_df.index.unique(level='filename'))}")
        print(f"# trajectories\t\t: {len(example_df)}")
        if 'occlusion_simulation' in example_exp_dict['dataset_used']:
            print(f"# occlusion cases\t: {(example_df['past_pred_length'] != 0).sum()}")
        print("\n")

    print(f"Experiments Performance Summary ({OPERATION}):")
    print(all_perf_df)

    if args.save_file:
        assert os.path.exists(os.path.dirname(args.save_file))
        assert not os.path.isfile(args.save_file)

        print(f"saving dataframe to:\n{args.save_file}\n")
        all_perf_df.to_csv(args.save_file)


if __name__ == '__main__':
    print("Hello!")
    parser = argparse.ArgumentParser()
    parser.add_argument('--score_files', nargs='+', type=os.path.abspath, default=None,
                        help="provide either multiple paths to 'prediction_scores.csv' files inside the 'results' "
                             "directory, or a path to a single .txt file, containing paths to those files "
                             "(relative to the repository's root directory). If nothing is passed, the script will "
                             "search for every single 'prediction_scores.csv' file inside the 'results' directory.")
    parser.add_argument('--unit', type=str, default='m',
                        help="\'m\' | \'px\'")
    parser.add_argument('--sort_by', nargs='+', type=str, default='experiment_name',
                        help="sort the performance table by column names.")
    parser.add_argument('--filter', nargs='+', type=str, default=None,
                        help="select any number of options from:\n"
                             "\'occluded_ids\', \'fully_observed_ids\', \'difficult_dataset\', "
                             "\'difficult_occluded_ids\', \'moving\', \'idle\'")
    parser.add_argument('--save_file', type=os.path.abspath, default=None,
                        help="path of a \'.csv\' file to save the performance table.")
    parser.add_argument('--print_dataset_stats', action='store_true', default=False,
                        help="passing this argument will make the script specify some dataset characteristics "
                             "by printing them to the screen.")
    args = parser.parse_args()

    main(args=args)
    print("Goodbye!")
