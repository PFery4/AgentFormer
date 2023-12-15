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
    parser.add_argument('--checkpoint_name', default=None)
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
    if checkpoint_name is not None:
        saved_preds_dir = os.path.join(cfg.result_dir, sdd_test_set.dataset_name, checkpoint_name, split)
    else:
        saved_preds_dir = os.path.join(cfg.result_dir, sdd_test_set.dataset_name, 'untrained', split)

    assert os.path.exists(saved_preds_dir)

    score_csv_file = os.path.join(saved_preds_dir, 'prediction_scores.csv')
    assert os.path.exists(score_csv_file)

    scores_df = pd.read_csv(score_csv_file)

    # WE CAN PERFORM EXTRA SORTING / ANALYSIS ON <scores_df>
    
