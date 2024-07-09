import argparse
import os.path
import pandas as pd

from utils.performance_analysis import \
    get_reference_indices, \
    get_all_results_directories, \
    get_df_filter, \
    generate_performance_summary_df, \
    get_perf_scores_df

# Global Variables set up #########################################################################################
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

if __name__ == '__main__':
    # Script Controls #################################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', nargs='+', default=None)
    parser.add_argument('--unit', type=str, default='m')        # 'm' | 'px'
    parser.add_argument('--sort_by', type=str, default='experiment')
    parser.add_argument('--filter', nargs='+', default=None)
    parser.add_argument('--save_file', type=os.path.abspath, default=None)
    parser.add_argument('--print_dataset_stats', action='store_true', default=False)
    args = parser.parse_args()

    assert args.unit in ['m', 'px']

    UNIT = args.unit  # 'm' | 'px'
    if UNIT == 'px':
        px_name = lambda names_list: [f'{score_name}_px' for score_name in names_list]
        MIN_SCORES = px_name(MIN_SCORES)
        MEAN_SCORES = px_name(MEAN_SCORES)
        PAST_MIN_SCORES = px_name(PAST_MIN_SCORES)
        PAST_MEAN_SCORES = px_name(PAST_MEAN_SCORES)

    print("PERFORMANCE SUMMARY:\n\n")

    experiment_names = args.cfg if args.cfg is not None else DEFAULT_CFG
    operation = 'mean'  # 'mean' | 'median' | 'IQR'

    metric_names = (MIN_SCORES + MEAN_SCORES +
                    PAST_MIN_SCORES + PAST_MEAN_SCORES +
                    PRED_LENGTHS + OCCLUSION_MAP_SCORES)

    ref_index = get_reference_indices()

    exp_dicts = get_all_results_directories()
    exp_dicts = [exp_dict for exp_dict in exp_dicts if exp_dict['split'] in 'test']
    exp_dicts = [exp_dict for exp_dict in exp_dicts if
                 exp_dict['dataset_used'] not in ['occlusion_simulation_difficult']]
    exp_dicts = [exp_dict for exp_dict in exp_dicts if exp_dict['experiment_name'] in experiment_names]

    df_filter = get_df_filter(ref_index=ref_index, filters=args.filter)

    all_perf_df = generate_performance_summary_df(
        experiments=exp_dicts, metric_names=metric_names, operation=operation, df_filter=df_filter
    )
    all_perf_df.sort_values(by=args.sort_by, inplace=True)

    if args.print_dataset_stats:  # specify some dataset characteristics
        exp_dict = exp_dicts[0]
        example_df = get_perf_scores_df(
            experiment_name=exp_dict['experiment_name'],
            dataset_used=exp_dict['dataset_used'],
            model_name=exp_dict['model_name'],
            split=exp_dict['split']
        )
        example_df = df_filter(example_df)

        print(f"Dataset Statistics:")
        print(f"# instances\t\t: {len(example_df.index.unique(level='filename'))}")
        print(f"# trajectories\t\t: {len(example_df)}")
        print(f"# occlusion cases\t: {(example_df['past_pred_length'] != 0).sum()}")
        print("\n")

    print(f"Experiments Performance Summary ({operation}):")
    print(all_perf_df)

    if args.save_file:

        assert os.path.exists(os.path.dirname(args.save_file))
        assert not os.path.isfile(args.save_file)

        print(f"saving dataframe to:\n{args.save_file}\n")
        all_perf_df.to_csv(args.save_file)
