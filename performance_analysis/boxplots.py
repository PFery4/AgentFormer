import argparse
import os
import matplotlib.pyplot as plt

from utils.performance_analysis import \
    get_reference_indices, \
    get_all_results_directories, \
    get_df_filter, \
    get_perf_scores_df, \
    make_box_plot_occlusion_lengths

# Global Variables set up #############################################################################################

FIG_SIZE = (9, 6)
FIG_DPI = 300

DEFAULT_SCORES = [
    'min_ADE', 'min_FDE',
    'mean_ADE', 'mean_FDE',
    # 'min_past_ADE', 'min_past_FDE',
    # 'mean_past_ADE', 'mean_past_FDE',
    # 'min_ADE_px', 'min_FDE_px',
    # 'mean_ADE_px', 'mean_FDE_px',
    # 'min_past_ADE_px', 'min_past_FDE_px',
    # 'mean_past_ADE_px', 'mean_past_FDE_px',
]

YLIMS_DICT = {
    # key: value --> score_name : (min ylim, max ylim)
    'min_ADE': (0.0, 11),
    'min_FDE': (0.0, 11),
    'mean_ADE': (0.0, 37),
    'mean_FDE': (0.0, 37),
    'min_past_ADE': (0.0, 5),
    'min_past_FDE': (0.0, 5),
    'mean_past_ADE': (0.0, 9),
    'mean_past_FDE': (0.0, 9),
    'min_ADE_px': (None, None),
    'min_FDE_px': (None, None),
    'mean_ADE_px': (None, None),
    'mean_FDE_px': (None, None),
    'min_past_ADE_px': (None, None),
    'min_past_FDE_px': (None, None),
    'mean_past_ADE_px': (None, None),
    'mean_past_FDE_px': (None, None),
}

if __name__ == '__main__':
    # Script Controls #################################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', nargs='+', default=None)
    parser.add_argument('--filter', nargs='+', default=None)
    parser.add_argument('--scores', nargs='+', type=str, default=DEFAULT_SCORES)
    parser.add_argument('--save_dir', type=os.path.abspath, default=None)
    args = parser.parse_args()

    print(args.scores)
    assert args.cfg is not None
    for score in args.scores:
        assert score in YLIMS_DICT.keys(), f"incorrect score: {score}"
    if args.save_dir is not None:
        print(args.save_dir)
        assert not os.path.exists(args.save_dir)
        os.makedirs(args.save_dir)

    print("BOXPLOTS:\n\n")

    experiment_names = args.cfg

    exp_dicts = get_all_results_directories()
    exp_dicts = [exp_dict for exp_dict in exp_dicts if exp_dict['split'] in ['test']]
    exp_dicts = [exp_dict for exp_dict in exp_dicts if exp_dict['experiment_name'] in experiment_names]
    # exp_dicts = [exp_dict for exp_dict in exp_dicts if exp_dict['dataset_used'] not in ['fully_observed']]
    # exp_dicts = [exp_dict for exp_dict in exp_dicts if exp_dict['dataset_used'] in ['fully_observed']]

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

    for plot_score in args.scores:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        make_box_plot_occlusion_lengths(
            draw_ax=ax,
            experiments=exp_dicts,
            plot_score=plot_score,
            categorization=ref_past_pred_lengths,
            df_filter=df_filter,
            ylim=YLIMS_DICT.get(plot_score, (None, None)),
            legend=False
        )
        ax.set_title(f"{plot_score} vs. Last Observed timestep")

        if args.save_dir is not None:
            filename = f"{plot_score}.png"
            filepath = os.path.join(args.save_dir, filename)

            print(f"Saving Boxplot figure to:\n{filepath}\n")
            plt.savefig(filepath, dpi=FIG_DPI, bbox_inches='tight')

    plt.show()
