import argparse
import os
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas.core.series

from typing import List

from utils.performance_analysis import \
    SCORES_CSV_FILENAME, \
    MIN_SCORES, MEAN_SCORES, \
    get_experiment_dict, \
    get_reference_indices, \
    get_df_filter, \
    get_perf_scores_df, \
    get_scores_dict_by_categories, \
    remove_k_sample_columns, \
    get_all_pred_scores_csv_files, \
    process_analysis_subjects_txt, \
    get_df_from_csv


FIG_SIZE = (9, 6)
FIG_DPI = 300

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


def make_box_plot_occlusion_lengths(
        draw_ax: matplotlib.axes.Axes,
        experiments: List[str],
        plot_score: str,
        categorization: pandas.core.series.Series,
        df_filter=None,
        ylim=(None, None),
        legend=True
) -> None:
    for exp_csv_path in experiments:
        assert exp_csv_path.endswith(SCORES_CSV_FILENAME), f"Error, incorrect file:\n{exp_csv_path}"

    print(f"categorization counts (total: {len(categorization)}):\n{categorization.value_counts()}\n\n")
    category_name, category_values = categorization.name, sorted(categorization.unique())
    colors = [plt.cm.Pastel1(i) for i in range(len(experiments))]

    def get_boxplot_dict_key(exp_csv_path: str) -> str:
        exp_dict = get_experiment_dict(file_path=exp_csv_path)
        return f"{exp_dict['experiment_name']}_{exp_dict['dataset_used']}"

    box_plot_dict = {get_boxplot_dict_key(exp_csv_path): None for exp_csv_path in experiments}
    for i, exp_csv_path in enumerate(experiments):

        experiment_df = get_df_from_csv(file_path=exp_csv_path)
        if df_filter is not None:
            experiment_df = df_filter(experiment_df)

        experiment_df = remove_k_sample_columns(df=experiment_df)

        experiment_df = experiment_df.iloc[experiment_df.index.isin(categorization.index)]

        if category_name not in experiment_df.columns:
            experiment_df[category_name] = categorization

        assert plot_score in experiment_df.columns
        assert category_name in experiment_df.columns

        box_plot_dict[get_boxplot_dict_key(exp_csv_path)] = get_scores_dict_by_categories(
            exp_df=experiment_df,
            score=plot_score,
            categorization=category_name
        )

    box_plot_xs = []
    box_plot_ys = []
    box_plot_colors = []
    for length in category_values:
        for i, exp_csv_path in enumerate(experiments):
            box_plot_xs.append(f"{length} - {get_boxplot_dict_key(exp_csv_path)}")
            box_plot_ys.append(box_plot_dict[get_boxplot_dict_key(exp_csv_path)][length])
            box_plot_colors.append(colors[i])

    bplot = draw_ax.boxplot(box_plot_ys, positions=range(len(box_plot_ys)), patch_artist=True)
    for box_patch, median_line, color in zip(bplot['boxes'], bplot['medians'], box_plot_colors):
        box_patch.set_facecolor(color)
        median_line.set_color('red')

    x_tick_gap = len(experiments)
    x_tick_start = (len(experiments) - 1) / 2
    x_tick_end = x_tick_start + x_tick_gap * len(category_values)
    draw_ax.set_xticks(np.arange(x_tick_start, x_tick_end, x_tick_gap), labels=-np.array(category_values))
    draw_ax.set_xticks(np.arange(x_tick_start, x_tick_end, x_tick_gap / 2), minor=True)
    draw_ax.grid(which='minor', axis='x')

    if legend:
        draw_ax.legend(
            [bplot["boxes"][i] for i in range(len(experiments))],
            [get_boxplot_dict_key(exp_csv_path) for exp_csv_path in experiments], loc='upper left'
        )
    draw_ax.set_ylabel(f'{plot_score}', loc='bottom')
    draw_ax.set_xlabel('last observation timestep', loc='left')

    draw_ax.set_ylim(*ylim)


def main(args: argparse.Namespace):
    print("BOXPLOTS:\n\n")

    score_names = []
    if args.unit == 'm':
        [score_names.append(score) for score in args.scores]
    elif args.unit == 'px':
        [score_names.append(f"{score}_px") for score in args.scores]
    else:
        raise NotImplementedError

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
        experiment_name='CV_predictor',
        dataset_used='occlusion_simulation',
        model_name='untrained',
        split='test'
    )
    ref_past_pred_lengths = ref_past_pred_lengths.iloc[ref_past_pred_lengths.index.isin(ref_index)]
    ref_past_pred_lengths = ref_past_pred_lengths['past_pred_length']

    df_filter = get_df_filter(ref_index=ref_index, filters=args.filter)
    ref_past_pred_lengths = df_filter(ref_past_pred_lengths)

    for plot_score in score_names:
        print(f"Creating boxplot figure for {plot_score}")
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        make_box_plot_occlusion_lengths(
            draw_ax=ax,
            experiments=subjects,
            plot_score=plot_score,
            categorization=ref_past_pred_lengths,
            df_filter=df_filter,
            ylim=YLIMS_DICT.get(plot_score, (None, None)),
            legend=args.legend
        )
        ax.set_title(f"{plot_score} vs. Last Observed timestep")

    plt.show()


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
    parser.add_argument('--filter', nargs='+', default=None,
                        help="select any number of options from:\n"
                             "\'occluded_ids\', \'fully_observed_ids\', \'difficult_dataset\', "
                             "\'difficult_occluded_ids\', \'moving\', \'idle\'")
    parser.add_argument('--scores', nargs='+', type=str, default=MIN_SCORES+MEAN_SCORES)
    parser.add_argument('--legend', action='store_true', default=False)
    args = parser.parse_args()

    main(args=args)
    print("Goodbye!")
