import argparse
import os.path

import pandas as pd
import torch

from data.sdd_dataloader import PresavedDatasetSDD
from utils.utils import prepare_seed
from utils.config import Config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--data_split', type=str, default='test')
    parser.add_argument('--checkpoint_name', default='best_val')        # can be 'best_val' / 'untrained' / <model_id>
    parser.add_argument('--tmp', action='store_true', default=False)
    args = parser.parse_args()

    split = args.data_split
    checkpoint_name = args.checkpoint_name

    cfg = Config(cfg_id=args.cfg, tmp=args.tmp, create_dirs=False)
    prepare_seed(cfg.seed)
    torch.set_default_dtype(torch.float32)

    # dataloader
    if cfg.dataset == 'sdd':
        sdd_test_set = PresavedDatasetSDD(parser=cfg, log=None, split=split)
    else:
        raise NotImplementedError

    if checkpoint_name == 'best_val':
        checkpoint_name = cfg.get_best_val_checkpoint_name()
        print(f"Best validation checkpoint name is: {checkpoint_name}")
    if checkpoint_name == 'untrained':
        saved_preds_dir = os.path.join(cfg.result_dir, sdd_test_set.dataset_name, 'untrained', split)
    else:
        saved_preds_dir = os.path.join(cfg.result_dir, sdd_test_set.dataset_name, checkpoint_name, split)
    assert os.path.exists(saved_preds_dir)

    score_csv_file = os.path.join(saved_preds_dir, 'prediction_scores.csv')
    assert os.path.exists(score_csv_file)

    scores_df = pd.read_csv(score_csv_file)

    # WE CAN PERFORM EXTRA SORTING / ANALYSIS ON <scores_df>
    print(f"{scores_df.columns=}")
    print(f"{scores_df.index=}")
    print(scores_df[['idx', 'agent_id', 'min_ADE', 'mean_ADE', 'min_FDE', 'mean_FDE']]
          .head(10))
    print(scores_df[['idx', 'agent_id', 'min_ADE', 'mean_ADE', 'min_FDE', 'mean_FDE']]
          .mean())

    print(f"{scores_df['idx'].value_counts()=}")
    # print(scores_df[['idx', 'agent_id', 'min_ADE', 'mean_ADE', 'min_FDE', 'mean_FDE']]
    #       .sort_values('min_ADE', ascending=True).head(5))
    #
    # instances_df = instance_collapse(scores_df)
    # print(f"{instances_df.columns=}")
    # print(f"{instances_df.index=}")
    # print(instances_df[['idx', 'agent_id', 'min_ADE', 'mean_ADE', 'min_FDE', 'mean_FDE']]
    #       .sort_values('min_ADE', ascending=True).head(5))

    instances_df = scores_df.set_index(['idx', 'agent_id'])
    instances_df = instances_df.groupby(level=0)
    instances_df = instances_df.agg('mean')
    print(instances_df[['min_ADE', 'mean_ADE', 'min_FDE', 'mean_FDE']]
          .head(5))
    print(instances_df[['min_ADE', 'mean_ADE', 'min_FDE', 'mean_FDE']]
          .mean())
