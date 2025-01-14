import argparse
import matplotlib.pyplot as plt
import os.path
import pickle
import torch

from data.sdd_dataloader import dataset_dict
from utils.config import Config, ModelConfig
from utils.sdd_visualize import visualize_input_and_predictions


FIG_SIZE = (14, 10)
FIG_DPI = 120

torch.set_default_dtype(torch.float32)


def main(args: argparse.Namespace):
    if args.legacy:
        assert args.dataset_class == 'hdf5', "Legacy mode is only available with presaved HDF5 datasets" \
                                             "(use: --dataset_class hdf5)"
    print("QUALITATIVE EXAMPLE:\n\n")

    cfg = ModelConfig(cfg_id=args.cfg, tmp=False, create_dirs=False)

    # dataloader
    dataset_class = dataset_dict[args.dataset_class]
    dataset_cfg = Config(cfg_id=args.dataset_cfg)
    dataset_cfg.__setattr__('with_rgb_map', args.with_rgb_map)
    dataset_kwargs = dict(parser=dataset_cfg, split=args.data_split)
    if args.legacy:
        dataset_kwargs.update(legacy_mode=True)
    dataset = dataset_class(**dataset_kwargs)

    # retrieve the corresponding entry name and dataset index
    instance_index = dataset.get_instance_idx(instance_num=args.instance_num)
    instance_name = f"{args.instance_num}".rjust(8, '0')

    # preparing the figure
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.set_dpi(FIG_DPI)
    fig_title = f"{cfg.id}: #{instance_name}"
    fig.canvas.manager.set_window_title(fig_title)

    checkpoint_name = cfg.get_best_val_checkpoint_name()
    save_preds_dir = os.path.join(cfg.result_dir, dataset.dataset_name, checkpoint_name, args.data_split)

    # retrieve the input data dict
    input_dict = dataset.__getitem__(instance_index)

    # retrieve the prediction data dict
    pred_file = os.path.join(save_preds_dir, instance_name)
    assert os.path.exists(pred_file)
    with open(pred_file, 'rb') as f:
        pred_dict = pickle.load(f)

    visualize_input_and_predictions(
        draw_ax=ax,
        data_dict=input_dict,
        pred_dict=pred_dict,
        show_rgb_map=args.with_rgb_map,
        show_gt_agent_ids=args.ids,
        show_obs_agent_ids=None,
        show_pred_agent_ids=args.ids,
        past_pred_alpha=0.5,
        future_pred_alpha=0.1 if args.highlight_past else 0.5
    )
    if args.legend:
        ax.legend()
    ax.set_title(cfg.id)
    fig.subplots_adjust(wspace=0.10, hspace=0.0)

    plt.show()


if __name__ == '__main__':
    print("Hello!")
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, default=None,
                        help="Model config file (specified as either name or path")
    parser.add_argument('--dataset_cfg', type=str, required=True, default=None,
                        help="Dataset config file (specified as either name or path")
    parser.add_argument('--data_split', type=str, default='test',
                        help="\'train\' | \'val\' | \'test\'")
    parser.add_argument('--dataset_class', type=str, default='hdf5',
                        help="\'torch\' | \'hdf5\'")
    parser.add_argument('--legacy', action='store_true', default=False)
    parser.add_argument('--instance_num', type=int, required=True,
                        help="dataset instance to display.")
    parser.add_argument('--ids', nargs='+', type=int, default=None,
                        help="agent identities to show future ground truths and model predictions.")
    parser.add_argument('--highlight_past', action='store_true', default=False)
    parser.add_argument('--legend', action='store_true', default=False)
    parser.add_argument('--with_rgb_map', action='store_true', default=False)
    args = parser.parse_args()

    main(args=args)
    print("Goodbye!")
