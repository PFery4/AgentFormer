import argparse
import matplotlib.pyplot as plt
import os.path
import pickle

from data.sdd_dataloader import HDF5DatasetSDD
from utils.config import Config, ModelConfig
from utils.sdd_visualize import visualize_input_and_predictions
from utils.performance_analysis import get_all_results_directories

FIG_SIZE = (14, 10)
FIG_DPI = 300

# TODO:
#   - FIX DATASET INSTANTIATION (USE NEW CLASS IMPLEMENTATIONS)
#   - BETTER ARGUMENT PASSING FOR CFG's (ALLOW FOR PASSING file paths, AND MAYBE AN OPTIONAL DEFAULT yml file)
#   - REMOVE --highlight_past FROM SCRIPT ARGUMENTS, SET AS A DEFAULT VALUE THAT CAN BE CHANGED FROM INSIDE THE SCRIPT


if __name__ == '__main__':
    # Script Controls #################################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', nargs='+', default=None)
    parser.add_argument('--instance_num', type=int, required=True)
    parser.add_argument('--ids', nargs='+', type=int, default=None)     # List[int]
    parser.add_argument('--highlight_past', action='store_true', default=False)
    parser.add_argument('--legend', action='store_true', default=False)
    args = parser.parse_args()

    print("QUALITATIVE EXAMPLE:\n\n")

    experiment_names = args.cfg
    assert args.instance_num is not None

    exp_dicts = get_all_results_directories()
    exp_dicts = [exp_dict for exp_dict in exp_dicts if exp_dict['split'] in 'test']
    exp_dicts = [exp_dict for exp_dict in exp_dicts if
                 exp_dict['dataset_used'] not in ['occlusion_simulation_difficult']]
    exp_dicts = [exp_dict for exp_dict in exp_dicts if exp_dict['experiment_name'] in experiment_names]

    for exp_dict in exp_dicts:
        # exp_dict.keys() --> ['experiment_name', 'dataset_used', 'model_name', 'split']

        # preparing the dataloader for the experiment
        model_config = ModelConfig(exp_dict['experiment_name'], tmp=False, create_dirs=False)
        dataset_config = Config(f"dataset_{exp_dict['dataset_used']}")
        dataloader_exp = HDF5DatasetSDD(dataset_config, split='test')       # TODO: optional dataset class

        # retrieve the corresponding entry name and dataset index
        instance_name = f"{args.instance_num}".rjust(8, '0')
        instance_index = dataloader_exp.get_instance_idx(instance_num=args.instance_num)

        # preparing the figure
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        fig.set_dpi(FIG_DPI)
        fig_title = f"{exp_dict['experiment_name']}_{instance_name}"
        fig.canvas.manager.set_window_title(fig_title)

        checkpoint_name = model_config.get_best_val_checkpoint_name()
        saved_preds_dir = os.path.join(
            model_config.result_dir, dataloader_exp.dataset_name, checkpoint_name, 'test'
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
            show_gt_agent_ids=args.ids,
            show_obs_agent_ids=None,
            show_pred_agent_ids=args.ids,
            past_pred_alpha=0.5,
            future_pred_alpha=0.1 if args.highlight_past else 0.5
        )
        if args.legend:
            ax.legend()
        ax.set_title(exp_dict['experiment_name'])
        fig.subplots_adjust(wspace=0.10, hspace=0.0)
    plt.show()
