import argparse
import os.path

from utils.config import Config
from data.sdd_dataloader import TorchDataGeneratorSDD


# METRICS #############################################################################################################


def compute_sequence_ADE():
    pass


def compute_sequence_FDE():
    pass


def compute_occlusion_area_occupancy():
    # Park et al.'s DAO applied on the occlusion map
    pass


def compute_occlusion_area_count():
    # Park et al.'s DAC applied on the occlusion map
    pass


def compute_rf():
    pass


if __name__ == '__main__':

    # TODO: IMPLEMENT EVALUATION TOOLS AND TEST SCRIPT HERE

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--log_file', default=None)
    args = parser.parse_args()

    assert parser.cfg is not None

    cfg = Config(args.cfg)

    dataset = args.dataset.lower()
    results_dir = args.results_dir
    assert results_dir is not None

    log_file = args.log_file if args.log_file is not None else os.path.join(results_dir, 'log_eval.txt')
    log_file = open(log_file, 'a+')

    cfg = Config()

    if dataset == 'sdd':
        torch_dataset = TorchDataGeneratorSDD()
    else:
        raise NotImplementedError

    pass
