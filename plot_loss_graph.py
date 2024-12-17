import argparse
import os.path
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utils.config import ModelConfig

COLOR_MAPS = {'train': 'cool', 'val': 'winter'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, nargs='+', required=True, default=None,
                        help="Model config file (specified as either name or path")
    parser.add_argument('--split', type=str, nargs='+', required=True, default=None,
                        help="\'train\' | \'val\'")
    parser.add_argument('--loss_names', type=str, nargs='+', default=None)
    parser.add_argument('--save_path', type=str, default=None,
                        help="path of a \'.png\' file to save the graph")
    parser.add_argument('--show', action='store_true', default=False)
    parser.add_argument('--unweighted_losses', action='store_true', default=False)
    args = parser.parse_args()

    assert all([name in ['train', 'val'] for name in args.split])
    assert args.save_path is not None or args.show, "You must choose to either --show the graph," \
                                                    "or save it by providing a --save_path."

    for cfg_str in args.cfg:
        cfg = ModelConfig(cfg_str, tmp=False, create_dirs=False)

        # create figure
        fig, ax = plt.subplots()

        for split in args.split:
            csv_file = os.path.join(cfg.log_dir, f'{split}_losses.csv')
            assert os.path.exists(csv_file)

            df = pd.read_csv(csv_file)

            # Beginning preprocessing
            # Remove rows with same content as columns
            df = df[~(df == df.columns).all(axis=1)]

            # Remove end of epoch placeholder rows
            df = df[~(df['tb_x'].str.contains('END OF EPOCH'))]
            df.reset_index(inplace=True, drop=True)

            # extract the loss names
            loss_names = [name for name in df.columns.tolist() if name not in ['tb_x', 'epoch', 'batch']]
            df = df.astype({'tb_x': int, 'epoch': int, 'batch': int})
            df = df.astype({name: float for name in loss_names})

            # multiply unweighted loss values by their weight factor
            loss_weights = {name: loss_dict['weight'] for name, loss_dict in cfg.loss_cfg.items()}
            if not args.unweighted_losses:
                for name, weight in loss_weights.items():
                    df[name] *= weight

            # get the x-axis values corresponding to the beginning of a new epoch
            new_epochs_tb_x = df['tb_x'][~(df['epoch'].eq(df['epoch'].shift()))].to_numpy()

            # identify the sequences that have been overwritten due to a rerun of the model from a previous checkpoint
            overwritten_rows = (df.groupby('tb_x').cumcount(ascending=False) != 0).to_numpy()
            kept_rows = (~overwritten_rows).nonzero()[0]
            seq_indices = (overwritten_rows[1:] != overwritten_rows[:-1])[overwritten_rows[1:]]
            seq_indices = seq_indices.nonzero()[0]
            seq_indices = np.append(seq_indices, overwritten_rows.nonzero()[0].shape[0])
            overwritten_rows = overwritten_rows.nonzero()[0]
            overwritten_sequences = [overwritten_rows[start:end] for start, end in zip(seq_indices[:-1], seq_indices[1:])]

            if args.loss_names is not None:
                assert all([loss_name in df.columns.tolist() for loss_name in args.loss_names])
                loss_names = args.loss_names

            colors = iter(mpl.colormaps[COLOR_MAPS[split]](np.linspace(0, 1, len(loss_names))))

            for loss in loss_names:
                c = next(colors)
                values = df[loss].to_numpy()

                for sequence in overwritten_sequences:
                    ax.plot(df['tb_x'].to_numpy()[sequence], values[sequence], c=c, alpha=0.4)
                ax.plot(df['tb_x'].to_numpy()[kept_rows], values[kept_rows], c=c, label=f"{split}_{loss}")

            ax.set_xticks(new_epochs_tb_x, minor=True)
            ax.set_title(cfg_str)
            ax.xaxis.grid(True, which='minor')
            fig.set_size_inches(16, 9)
            plt.legend(loc='upper right')

        if args.save_path is not None:
            assert os.path.exists(os.path.dirname(os.path.abspath(args.save_path)))
            assert args.save_path.endswith('.png')
            print(f"saving loss graph under:\n{os.path.abspath(args.save_path)}")
            plt.savefig(os.path.abspath(args.save_path), bbox_inches='tight', transparent=True, dpi=100)

    if args.show:
        plt.show()
    print("Goodbye!")
