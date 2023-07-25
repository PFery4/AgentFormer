import os.path
import random
from io import TextIOWrapper
import torch
import sys
import numpy as np

from data.map import GeometricMap
from utils.config import Config
from utils.utils import print_log, get_timestring

# imports from https://github.com/PFery4/occlusion-prediction
from src.data.sdd_dataloader import StanfordDroneDataset, StanfordDroneDatasetWithOcclusionSim
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
        if not parser.get("sdd_occlusion_data", False):
            full_dataset = StanfordDroneDataset(self.sdd_config)
            self.motion_processing = self.generate_motion_threedee
        else:
            full_dataset = StanfordDroneDatasetWithOcclusionSim(self.sdd_config)
            self.motion_processing = self.generate_motion_threedee_with_occlusion
        print(f"instantiating dataloader from {full_dataset.__class__} class")

        # TODO: investigate whether a split strategy such as the one used here won't possibly result in data leakage
        # No, it won't, so long as normalization does not involve data from the test/val splits.
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

    @staticmethod
    def torchimg_to_geometricmap(image_tensor: torch.Tensor):
        # todo: investigate if scaling factor matters here
        scaling = 3.0
        homography = np.array([[scaling, 0., 0.],
                               [0., scaling, 0.],
                               [0., 0., scaling]])
        np_img = np.array(image_tensor.permute(0, 1, 2).numpy() * 255, dtype=np.uint8)
        return GeometricMap(data=np_img, homography=homography)

    @staticmethod
    def generate_motion_threedee(extracted_data: dict, target_dict: dict) -> None:

        # pre_threedee = []
        # fut_threedee = []
        pre_mask = []
        fut_mask = []
        valid_id = []

        full_threedee = []

        for agent in extracted_data["agents"]:
            # pre_threedee.append(
            #     torch.from_numpy(agent.get_traj_section(extracted_data["past_window"])).float()
            # )
            # fut_threedee.append(
            #     torch.from_numpy(agent.get_traj_section(extracted_data["future_window"])).float()
            # )
            pre_mask.append(
                torch.from_numpy(agent.get_data_availability_mask(extracted_data["past_window"])).float()
            )
            fut_mask.append(
                torch.from_numpy(agent.get_data_availability_mask(extracted_data["future_window"])).float()
            )
            full_threedee.append(
                torch.from_numpy(agent.get_traj_section(extracted_data["full_window"])).float()
            )
            valid_id.append(float(agent.id))

        # target_dict['pre_motion_3D'] = pre_threedee
        # target_dict['fut_motion_3D'] = fut_threedee
        target_dict['pre_motion_mask'] = pre_mask
        target_dict['fut_motion_mask'] = fut_mask
        target_dict['valid_id'] = valid_id
        target_dict['full_motion_3D'] = full_threedee
        target_dict['obs_mask'] = [
            torch.from_numpy(np.concatenate(
                (np.ones_like(extracted_data["past_window"]),
                 np.zeros_like(extracted_data["future_window"]))
            ))
        ] * len(extracted_data["agents"])

    @staticmethod
    def generate_motion_threedee_with_occlusion(extracted_data: dict, target_dict: dict) -> None:
        pre_masks = []
        fut_masks = []
        valid_id = []

        full_threedee = []
        obs_mask = []
        for agent, occlusion_mask in zip(extracted_data["agents"], extracted_data["full_window_occlusion_masks"]):
            try:
                last_observed_timestep = np.where(occlusion_mask[:len(extracted_data["past_window"])])[0][-1]
            except IndexError:         # agent's past is completely unobserved, the ego has no knowledge of the agent
                # print(f"IGNORING AGENT {agent.id}: fully occluded")
                continue
            full_mot = agent.get_traj_section(extracted_data["full_window"])
            full_threedee.append(torch.from_numpy(full_mot).float())

            observed = occlusion_mask
            observed[len(extracted_data["past_window"]):] = False

            obs_mask.append(torch.from_numpy(observed.astype(float)))

            pre_mask = agent.get_data_availability_mask(extracted_data["full_window"])
            pre_mask[last_observed_timestep+1:] = 0.0
            pre_masks.append(torch.from_numpy(pre_mask).float())

            fut_mask = agent.get_data_availability_mask(extracted_data["full_window"])
            fut_mask[:last_observed_timestep+1] = 0.0
            fut_masks.append(torch.from_numpy(fut_mask).float())

            valid_id.append(float(agent.id))

        target_dict['full_motion_3D'] = full_threedee
        target_dict['obs_mask'] = obs_mask
        target_dict['pre_motion_mask'] = pre_masks
        target_dict['fut_motion_mask'] = fut_masks
        target_dict['valid_id'] = valid_id

    def convert_to_preprocessor_data(self, extracted_data: dict) -> dict:

        data = dict()
        self.motion_processing(extracted_data=extracted_data, target_dict=data)

        heading = None
        # from the nuscenes implementation:
        # pred mask is a numpy array, of shape (n_agents,), with values either 1 or 0
        pred_mask = None

        scene_map = self.torchimg_to_geometricmap(extracted_data["image_tensor"])

        data['timesteps'] = torch.from_numpy(
            np.arange(len(extracted_data["full_window"])) - len(extracted_data["past_window"]) + 1
        )
        data['heading'] = heading
        data['traj_scale'] = self.traj_scale
        data['pred_mask'] = pred_mask
        data['scene_map'] = scene_map
        data['seq'] = extracted_data["scene"] + "_" + extracted_data["video"]
        data['frame'] = extracted_data["timestep"]

        return data

    def next_sample(self) -> dict:
        sample_index = self.sample_list[self.index]
        self.index += 1

        data = self.dataset.__getitem__(sample_index)
        return self.convert_to_preprocessor_data(data)

    def __call__(self) -> dict:
        return self.next_sample()


if __name__ == '__main__':
    print(sdd_extract.REPO_ROOT)

    n_calls = 10
    # config_str = "sdd_agentformer_pre"
    config_str = "sdd_occlusion_agentformer_pre"

    ####################################################################################################################
    config = Config(config_str)
    log = open(os.path.join(config.log_dir, "log.txt"), "a+")
    time_str = get_timestring()
    print_log("time str: {}".format(time_str), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch version : {}".format(torch.__version__), log)
    print_log("cudnn version : {}".format(torch.backends.cudnn.version()), log)

    generator = AgentFormerDataGeneratorForSDD(config, log, split="train")
    generator.shuffle()
    for i in range(n_calls):
        print("\nCALLING")
        [print(f"{k}: {v}") for k, v in generator().items()]
