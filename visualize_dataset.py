import argparse
import os.path
import matplotlib.pyplot as plt
import numpy as np

from data.sdd_dataloader import dataset_dict
from utils.config import Config
from utils.utils import prepare_seed
from utils.sdd_visualize import visualize


DEFAULT_ROWS, DEFAULT_COLS = 3, 3


def main(args: argparse.Namespace):
    assert args.dataset_class in ['hdf5', 'torch']
    assert args.save_path is not None or args.show, "You must choose to either --show the image," \
                                                    "or save it by providing a --save_path."
    if args.save_path is not None:
        assert os.path.exists(os.path.dirname(os.path.abspath(args.save_path)))
        assert args.save_path.endswith('.png')

    dataset_class = dataset_dict[args.dataset_class]
    data_cfg = Config(cfg_id=args.cfg)
    data_cfg.__setattr__('with_rgb_map', args.with_scene_map)
    assert data_cfg.dataset == "sdd"
    if data_cfg.dataset == "sdd":
        print(f"\nUsing dataset of class: {dataset_class}\n")
        dataset = dataset_class(parser=data_cfg, split=args.split)

    if args.idx is not None:
        rows = int(np.sqrt(len(args.idx)))
        cols = int(np.ceil(len(args.idx) / rows))
        assert rows * cols >= len(args.idx)
    else:
        args.idx = sorted(np.random.choice(range(len(dataset)), size=(DEFAULT_ROWS * DEFAULT_COLS), replace=False))
        rows, cols = DEFAULT_ROWS, DEFAULT_COLS

    prepare_seed(data_cfg.seed)

    fig, ax = plt.subplots(nrows=rows, ncols=cols)
    for i, instance_idx in enumerate(args.idx):
        row_idx, col_idx = i // rows, i % rows

        data_dict = dataset.__getitem__(instance_idx)

        visualize(
            data_dict=data_dict,
            draw_ax=ax[row_idx, col_idx]
        )
        ax[row_idx, col_idx].set_title(f"{instance_idx:08}")
        ax[row_idx, col_idx].set_xticks([])
        ax[row_idx, col_idx].set_yticks([])

    if args.save_path is not None:
        print(f"saving image under:\n{os.path.abspath(args.save_path)}")
        plt.savefig(os.path.abspath(args.save_path), bbox_inches='tight', transparent=True, dpi=100)

    if args.show:
        plt.show()


if __name__ == '__main__':
    print("Hello!")
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, default=None,
                        help="Dataset config file (specified as either name or path")
    parser.add_argument('--split', type=str, default='train',
                        help="\'train\' | \'val\' | \'test\'")
    parser.add_argument('--dataset_class', type=str, default='hdf5',
                        help="\'torch\' | \'hdf5\'")
    parser.add_argument('--idx', type=int, nargs='+', default=None)
    parser.add_argument('--legacy', action='store_true', default=False)
    parser.add_argument('--save_path', type=os.path.abspath, default=None,
                        help="path of a directory to save the images")
    parser.add_argument('--show', action='store_true', default=False)
    parser.add_argument('--with_scene_map', action='store_true', default=False)
    args = parser.parse_args()

    main(args=args)
    print("Goodbye!")
