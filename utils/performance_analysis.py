import os
import yaml
import pandas as pd
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

from typing import Any, Dict, List, Optional, Tuple

from utils.config import REPO_ROOT


def get_perf_scores_df(experiment_name: str, model_name: Optional[str] = None, split: str = 'test') -> pd.DataFrame:

    target_path = os.path.join(REPO_ROOT, 'results', experiment_name, 'results')

    dataset_used = os.listdir(target_path)[0]
    target_path = os.path.join(target_path, dataset_used)
    assert os.path.exists(target_path)

    if model_name is None:
        model_name = os.listdir(target_path)[0]
    target_path = os.path.join(target_path, model_name, split, 'prediction_scores.csv')
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


def print_occlusion_length_counts():
    dataframe = get_perf_scores_df('occlusionformer_no_map')
    print("last observed timestep\t| case count")
    for i in range(0, 7):
        mini_df = dataframe[dataframe['past_pred_length'] == i]
        print(f"{-i}\t\t\t| {len(mini_df)}")
    print(f"total\t\t\t| {len(dataframe)}")


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


def draw_boxplots(
        draw_ax: matplotlib.axes.Axes, df: pd.DataFrame, column_data: str, column_boxes: str
) -> None:
    assert column_data in df.columns
    assert column_boxes in df.columns

    box_names = sorted(df[column_boxes].unique())
    box_data = []

    for box_category in box_names:

        mini_df = df[(df[column_boxes] == box_category) & (pd.notna(df[column_boxes]))]

        scores = mini_df[column_data].to_numpy()
        box_data.append(scores)

    draw_ax.boxplot(box_data)
    draw_ax.set_xticklabels(box_names)


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


def scatter_performance_scores(
        draw_ax: matplotlib.axes.Axes,
        x_df: pd.DataFrame, x_score_name: str,
        y_df: pd.DataFrame, y_score_name: str,
        xlabel_prefix: Optional[str] = None,
        ylabel_prefix: Optional[str] = None
) -> None:
    assert x_score_name in x_df.columns
    assert y_score_name in y_df.columns

    keep_indices = x_df.index.intersection(y_df.index)
    xs = x_df.loc[keep_indices, x_score_name].to_numpy()
    ys = y_df.loc[keep_indices, y_score_name].to_numpy()

    draw_ax.scatter(xs, ys, marker='x', color='black', alpha=0.7)

    x_label = f"{xlabel_prefix} | {x_score_name}" if xlabel_prefix is not None else x_score_name
    y_label = f"{ylabel_prefix} | {y_score_name}" if ylabel_prefix is not None else y_score_name

    draw_ax.set_xlabel(x_label, loc='left')
    draw_ax.set_ylabel(y_label, loc='bottom')


def get_occluded_identities(df: pd.DataFrame, idx: int):
    print(df.loc[idx].index.get_level_values('agent_id').to_list())
    print(df.loc[idx])
    instance_df = df.loc[idx]
    if (instance_df['past_pred_length'] != 0).sum() != 0:   # if there are agents who we do have to predict over the past
        return instance_df[instance_df['past_pred_length'] != 0].index.get_level_values('agent_id').to_list()
        # return df[df['past_pred_length'] != 0].loc[idx].index.get_level_values('agent_id').to_list()
    else:
        return []


def get_comparable_rows(
        base_df: pd.DataFrame,
        compare_df: pd.DataFrame,
        n_max_agents: Optional[int] = None
):
    # identifying the rows the two dataframes have in common
    common_rows = base_df.index.intersection(compare_df.index)
    compare_rows = common_rows.copy()
    if n_max_agents is not None:
        # defining a filter to remove instances with agent count larger than <n_max_agents>
        row_indices_leq_n_max_agents = (common_rows.get_level_values('idx').value_counts() <= n_max_agents)
        row_indices_leq_n_max_agents = row_indices_leq_n_max_agents[row_indices_leq_n_max_agents].index
        # the multiindex of rows to use for comparison
        compare_rows = common_rows[common_rows.get_level_values('idx').isin(row_indices_leq_n_max_agents)]
    return compare_rows


def make_box_plot_occlusion_lengths(
        draw_ax: matplotlib.axes.Axes,
        experiments: List[str],
        plot_score: str,
        categorization: Tuple[str, List[int]] = ('past_pred_length', range(1, 7)),
) -> None:
    category_name, category_values = categorization
    colors = [plt.cm.Pastel1(i) for i in range(len(experiments))]

    box_plot_dict = {experiment_name: None for experiment_name in experiments}
    for i, experiment in enumerate(experiments):

        experiment_df = get_perf_scores_df(experiment_name=experiment)
        experiment_df = remove_k_sample_columns(df=experiment_df)

        assert plot_score in experiment_df.columns
        assert category_name in experiment_df.columns

        pred_lengths = sorted(experiment_df[category_name].unique())

        experiment_data_dict = {int(pred_length): None for pred_length in pred_lengths}
        for pred_length in pred_lengths:
            mini_df = experiment_df[
                (experiment_df[category_name] == pred_length) & (pd.notna(experiment_df[plot_score]))
            ]

            scores = mini_df[plot_score].to_numpy()
            experiment_data_dict[int(pred_length)] = scores

        box_plot_dict[experiment] = experiment_data_dict

    box_plot_xs = []
    box_plot_ys = []
    box_plot_colors = []
    for length in category_values:
        for i, experiment in enumerate(experiments):
            box_plot_xs.append(f"{length} - {experiment}")
            box_plot_ys.append(box_plot_dict[experiment][length])
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

    draw_ax.legend([bplot["boxes"][i] for i in range(len(experiments))], experiments, loc='upper left')
    draw_ax.set_ylabel(f'{plot_score}', loc='bottom')
    draw_ax.set_xlabel('last observation timestep', loc='left')

    # draw_ax.set_title(f"{plot_score} vs. last observed timestep")


def oac_histogram(
        draw_ax: matplotlib.axes.Axes,
        experiment_name: str,
        plot_score: str,
        categorization: Tuple[str, Any],
        as_percentage: Optional[bool] = False,
        n_bins: int = 20
):
    # categorization [0, 1] <--> [column in experiment_df, value by which to filter that column]
    experiment_df = get_perf_scores_df(experiment_name)
    mini_df = experiment_df[
        (experiment_df[categorization[0]] == categorization[1]) & (pd.notna(experiment_df[plot_score]))
        ]

    scores = mini_df[plot_score].to_numpy()
    weights = np.ones_like(scores) / (scores.shape[0]) if as_percentage else None

    draw_ax.hist(scores, bins=n_bins, weights=weights)

    # draw_ax.set_xlabel(f"{plot_score}", loc='left')
    # draw_ax.set_title(f"{plot_score} histogram: {experiment_name} (last observed timestep: {-categorization[1]})")


def oac_histograms_versus_lastobs(
        draw_ax: matplotlib.axes.Axes,
        experiment_name: str,
        plot_score: str,
):
    experiment_df = get_perf_scores_df(experiment_name)

    mini_df = experiment_df[pd.notna(experiment_df[plot_score])]

    scores1 = mini_df[plot_score].to_numpy()
    past_pred_lengths = mini_df['past_pred_length'].to_numpy()
    unique, counts = np.unique(past_pred_lengths, return_counts=True)
    weights = 1 / counts[past_pred_lengths - 1]
    # weights = 1 / (counts[occl_lengths-1] * unique.shape[0])

    hist = draw_ax.hist2d(
        scores1, past_pred_lengths,
        bins=[np.linspace(0, 1, 21), np.arange(7) + 0.5],
        weights=weights
    )

    draw_ax.get_figure().colorbar(hist[3], ax=draw_ax)
    draw_ax.set_xlabel(plot_score)
    draw_ax.set_ylabel("last observed timestep")
    # draw_ax.set_title(f"OAC histogram: {experiment_name}")


def make_oac_histograms_figure(
        fig: matplotlib.figure.Figure,
        experiment_name: str,
        plot_score: str,
        as_percentage: bool = False
):
    gs = fig.add_gridspec(6, 2)

    ax_list = []
    for i in range(6):
        ax_list.append(fig.add_subplot(gs[i, 0]))
    ax_twodee = fig.add_subplot(gs[:, 1])

    for i, ax in enumerate(reversed(ax_list)):
        oac_histogram(draw_ax=ax, experiment_name=experiment_name, plot_score=plot_score,
                      categorization=('past_pred_length', i + 1), as_percentage=as_percentage)
    oac_histograms_versus_lastobs(draw_ax=ax_twodee, experiment_name=experiment_name, plot_score=plot_score)
    ax_list[-1].set_xlabel(plot_score)
    fig.suptitle(f"{plot_score} histograms: {experiment_name}")

