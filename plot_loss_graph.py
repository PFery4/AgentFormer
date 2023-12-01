import argparse
import os.path
import pandas as pd

from utils.config import Config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--split', default=None)
    args = parser.parse_args()

    assert args.split in ['train', 'val']
    cfg = Config(args.cfg)

    csv_file = os.path.join(cfg.log_dir, f'{args.split}_losses.csv')
    print(f"{csv_file=}")

    assert os.path.exists(csv_file)

    df = pd.read_csv(csv_file)
    # print(df)

    # wip to show example of overwritten data
    lines = df.iloc[[12, 13, 14, 15]]
    lines.index = [15.1, 15.2, 15.3, 15.4]
    # print(f"{lines}")
    df2 = pd.concat([df, lines])
    df2.sort_index(inplace=True)
    df2.reset_index(inplace=True, drop=True)

    print(df2)

    df2['overwritten'] = df2.groupby('tb_x').cumcount(ascending=False)

    print(df2)

    print("Goodbye!")
