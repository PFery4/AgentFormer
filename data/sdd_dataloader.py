import os.path
import random
from typing import Tuple
from io import TextIOWrapper
import torch
import sys

from utils.config import Config
from utils.utils import print_log, get_timestring

# imports from https://github.com/PFery4/occlusion-prediction
from src.data.sdd_dataloader import StanfordDroneDataset
import src.data.sdd_extract as sdd_extract


class AgentFormerDataGeneratorForSDD:
    """
    This class wraps the dataset classes implemented in the occlusion-prediction repo in such a way that
    they are directly usable as 'generator' objects in the source code of AgentFormer.
    """

    def __init__(self, parser: Config, log: TextIOWrapper, split: str = 'train', phase: str = 'training'):
        self.past_frames = parser.past_frames
        self.min_past_frames = parser.min_past_frames
        self.frame_skip = parser.get('frame_skip', 1)
        self.phase = phase
        self.split = split
        assert phase in ['training', 'testing'], 'error'
        assert split in ['train', 'val', 'test'], 'error'
        assert parser.dataset == "sdd", f"error: wrong dataset name: {parser.dataset} (should be \"sdd\")"

        self.sdd_config = sdd_extract.get_config(parser.sdd_config_file_name)
        full_dataset = StanfordDroneDataset(self.sdd_config)

        # TODO: investigate whether a split strategy such as the one used here won't possibly result in data leakage
        split_proportions = [0.7, 0.2, 0.1]
        train_size = int(split_proportions[0] * len(full_dataset))
        val_size = int(split_proportions[1] * len(full_dataset))
        test_size = len(full_dataset) - (train_size + val_size)

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        print_log("\n-------------------------- loading %s data --------------------------" % split, log=log)
        if self.split == 'train':
            self.dataset = train_dataset
        elif self.split == 'val':
            self.dataset = val_dataset
        elif self.split == 'test':
            self.dataset = test_dataset
        else:
            assert False, 'error'

        self.num_total_samples = len(self.dataset)

        print_log(f'total num samples: {self.num_total_samples}', log)
        print_log("------------------------------ done --------------------------------\n", log=log)

        self.sample_list = list(range(self.num_total_samples))
        self.index = 0

        self.traj_scale = parser.traj_scale

    def shuffle(self) -> None:
        random.shuffle(self.sample_list)

    def is_epoch_end(self) -> bool:
        if self.index >= self.num_total_samples:
            self.index = 0
            return True
        else:
            return False

    def convert_to_preprocessor_data(self, extracted_data: dict) -> dict:

        pre_motion_3D = []
        fut_motion_3D = []
        pre_motion_mask = []
        fut_motion_mask = []
        valid_id = []

        for agent in extracted_data["agents"]:
            pre_motion_3D.append(
                torch.from_numpy(agent.get_traj_section(extracted_data["past_window"])).float()
            )
            fut_motion_3D.append(
                torch.from_numpy(agent.get_traj_section(extracted_data["future_window"])).float()
            )
            pre_motion_mask.append(
                torch.from_numpy(agent.get_data_availability_mask(extracted_data["past_window"])).float()
            )
            fut_motion_mask.append(
                torch.from_numpy(agent.get_data_availability_mask(extracted_data["future_window"])).float()
            )
            valid_id.append(float(agent.id))

        # TODO: DO SOMETHING WITH heading (check how nuscenes does it)
        heading = None
        # TODO: DO SOMETHING WITH pred_mask (check how nuscenes does it)
        pred_mask = None
        # TODO: DO SOMETHING WITH scene_map (check how nuscenes does it)
        scene_map = None

        data = {
            'pre_motion_3D': pre_motion_3D,
            'fut_motion_3D': fut_motion_3D,
            'fut_motion_mask': fut_motion_mask,
            'pre_motion_mask': pre_motion_mask,
            # 'pre_data': None,
            # 'fut_data': None,
            'heading': heading,
            'valid_id': valid_id,
            'traj_scale': self.traj_scale,
            'pred_mask': pred_mask,
            'scene_map': scene_map,
            # 'seq': None,
            # 'frame': None
        }

        return data

    def next_sample(self) -> dict:
        sample_index = self.sample_list[self.index]
        self.index += 1

        data = self.dataset.__getitem__(sample_index)
        [print(k, v) for k, v in data.items()]
        return self.convert_to_preprocessor_data(data)

    def __call__(self) -> dict:
        return self.next_sample()


if __name__ == '__main__':
    print("Hello World!")
    print(sdd_extract.REPO_ROOT)

    config = Config("sdd_agentformer_pre")
    log = open(os.path.join(config.log_dir, "log.txt"), "a+")
    time_str = get_timestring()
    print_log("time str: {}".format(time_str), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch version : {}".format(torch.__version__), log)
    print_log("cudnn version : {}".format(torch.backends.cudnn.version()), log)

    generator = AgentFormerDataGeneratorForSDD(config, log, split="train")
    generator.shuffle()
    print("CALLING")
    print(generator())
