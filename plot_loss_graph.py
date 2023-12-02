import argparse
import os.path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utils.config import Config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--split', default=None)
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--show', type=bool, default=False)
    args = parser.parse_args()

    assert args.split in ['train', 'val']
    assert args.save or args.show, "You must choose to either *show* or *save* the loss graph..."
    cfg = Config(args.cfg)

    csv_file = os.path.join(cfg.log_dir, f'{args.split}_losses.csv')
    print(f"{csv_file=}")

    assert os.path.exists(csv_file)

    df = pd.read_csv(csv_file)

    # # wip to show example of corrupted dataframe
    # lines = df.iloc[[12, 13, 14, 15]]
    # lines.index = [15.1, 15.2, 15.3, 15.4]
    # df = pd.concat([df, lines])
    # df.sort_index(inplace=True)
    # df.reset_index(inplace=True, drop=True)
    # lines = df.iloc[[29, 30, 31]]
    # lines.index = [31.1, 31.2, 31.3]
    # df = pd.concat([df, lines])
    # df.sort_index(inplace=True)
    # df.reset_index(inplace=True, drop=True)
    # df.loc[10000] = ['tb_x', 'epoch', 'batch', 'total_loss', 'mse', 'kld', 'sample']
    # df = df.sort_index().reset_index(drop=True)
    # print(df)

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

    # create figure
    fig, ax = plt.subplots()
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(loss_names))))

    for loss in loss_names:
        c = next(colors)
        values = df[loss].to_numpy()

        for sequence in overwritten_sequences:
            ax.plot(df['tb_x'].to_numpy()[sequence], values[sequence], c=c, alpha=0.4)
        ax.plot(df['tb_x'].to_numpy()[kept_rows], values[kept_rows], c=c, label=loss)

    ax.set_xticks(new_epochs_tb_x, minor=True)
    ax.xaxis.grid(True, which='minor')
    fig.set_size_inches(16, 9)
    plt.legend(loc='upper right')

    if args.save:
        save_path = os.path.join(cfg.log_dir, f'{args.split}_losses.png')
        print(f"saving loss graph under:\n{save_path}")

        plt.savefig(save_path, bbox_inches='tight', transparent=True, dpi=100)
    if args.show:
        plt.show()
    print("Goodbye!")
