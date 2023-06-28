import torch, os, numpy as np
import cv2
from io import TextIOWrapper
from typing import Tuple, List, Union

from data.map import GeometricMap
from utils.config import Config


class preprocess(object):
    class_names = {
        'Pedestrian': 1, 'Car': 2, 'Cyclist': 3, 'Truck': 4, 'Van': 5, 'Tram': 6, 'Person': 7, 'Misc': 8,
        'DontCare': 9, 'Traffic_cone': 10, 'Construction_vehicle': 11, 'Barrier': 12, 'Motorcycle': 13,
        'Bicycle': 14, 'Bus': 15, 'Trailer': 16, 'Emergency': 17, 'Construction': 18
    }

    def __init__(self, data_root: str, seq_name: str, parser: Config, log: TextIOWrapper,
                 split: str = 'train', phase: str = 'training'):
        self.parser = parser        # Config
        self.dataset = parser.dataset       # str
        self.data_root = data_root          # str
        self.past_frames = parser.past_frames       # int
        self.future_frames = parser.future_frames   # int
        self.frame_skip = parser.get('frame_skip', 1)       # int
        self.min_past_frames = parser.get('min_past_frames', self.past_frames)      # int
        self.min_future_frames = parser.get('min_future_frames', self.future_frames)        # int
        self.traj_scale = parser.traj_scale     # int
        self.past_traj_scale = parser.traj_scale        # int
        self.load_map = parser.get('load_map', False)       # bool
        self.map_version = parser.get('map_version', '0.1')     # str
        self.seq_name = seq_name        # str
        self.split = split          # str
        self.phase = phase          # str
        self.log = log              # TextIOWrapper

        if parser.dataset == 'nuscenes_pred':
            # label_path = os.path.join(data_root, 'label/{}/{}.txt'.format(split, seq_name))
            label_path = os.path.join(data_root, "label", split, f"{seq_name}.txt")
            delimiter = ' '
        elif parser.dataset in {'eth', 'hotel', 'univ', 'zara1', 'zara2'}:
            # label_path = f'{data_root}/{parser.dataset}/{seq_name}.txt'
            label_path = os.path.join(data_root, parser.dataset, f"{seq_name}.txt")
            delimiter = ' '
        else:
            assert False, 'error'

        # read the annotation .txt file
        self.gt = np.genfromtxt(label_path, delimiter=delimiter, dtype=str)     # np.ndarray

        # frames correspond to the timestep of the observation
        frames = self.gt[:, 0].astype(np.float32).astype(int)
        fr_start, fr_end = frames.min(), frames.max()
        self.init_frame = fr_start      # np.int64
        self.num_fr = fr_end + 1 - fr_start     # np.int64

        if self.load_map:
            self.load_scene_map()
        else:
            self.geom_scene_map = None

        for row_index in range(len(self.gt)):
            self.gt[row_index][2] = self.class_names[self.gt[row_index][2]]
        self.gt = self.gt.astype('float32')     # np.ndarray
        self.xind, self.zind = 13, 15           # column index for the x and z positions of the agent (within the .txt)

    def GetID(self, data: np.ndarray) -> List[np.float32]:
        id = []
        for i in range(data.shape[0]):
            id.append(data[i, 1].copy())
        return id

    def TotalFrame(self) -> np.int64:
        """
        returns the total number of timesteps in the contained data
        """
        return self.num_fr

    def PreData(self, frame: np.int64) -> List[np.ndarray]:
        DataList = []
        for i in range(self.past_frames):
            if frame - i < self.init_frame:
                data = []
            data = self.gt[self.gt[:, 0] == (frame - i * self.frame_skip)]
            DataList.append(data)
        return DataList
    
    def FutureData(self, frame: np.int64) -> List[np.ndarray]:
        DataList = []
        for i in range(1, self.future_frames + 1):
            data = self.gt[self.gt[:, 0] == (frame + i * self.frame_skip)]
            DataList.append(data)
        return DataList

    def get_valid_id(self, pre_data: List[np.ndarray], fut_data: List[np.ndarray]) -> List[np.float32]:
        """
        within the past and future datablocks, produce the list of agents for which a coordinate is available at
        every timestep
        """
        cur_id = self.GetID(pre_data[0])        # the agent ids at t=0
        valid_id = []
        for idx in cur_id:
            exist_pre = [(False if isinstance(data, list) else (idx in data[:, 1])) for data in pre_data[:self.min_past_frames]]
            exist_fut = [(False if isinstance(data, list) else (idx in data[:, 1])) for data in fut_data[:self.min_future_frames]]
            if np.all(exist_pre) and np.all(exist_fut):
                valid_id.append(idx)
        return valid_id

    def get_pred_mask(self, cur_data: np.ndarray, valid_id: List[np.float32]) -> np.ndarray:
        pred_mask = np.zeros(len(valid_id), dtype=int)
        for i, idx in enumerate(valid_id):
            pred_mask[i] = cur_data[cur_data[:, 1] == idx].squeeze()[-1]
        return pred_mask

    def get_heading(self, cur_data: np.ndarray, valid_id: List[np.float32]) -> np.ndarray:
        heading = np.zeros(len(valid_id))
        for i, idx in enumerate(valid_id):
            heading[i] = cur_data[cur_data[:, 1] == idx].squeeze()[16]
        return heading

    def load_scene_map(self) -> None:
        map_file = f'{self.data_root}/map_{self.map_version}/{self.seq_name}.png'
        map_vis_file = f'{self.data_root}/map_{self.map_version}/vis_{self.seq_name}.png'
        map_meta_file = f'{self.data_root}/map_{self.map_version}/meta_{self.seq_name}.txt'
        self.scene_map = np.transpose(cv2.imread(map_file), (2, 0, 1))
        self.scene_vis_map = np.transpose(cv2.cvtColor(cv2.imread(map_vis_file), cv2.COLOR_BGR2RGB), (2, 0, 1))
        self.meta = np.loadtxt(map_meta_file)
        self.map_origin = self.meta[:2]
        self.map_scale = scale = self.meta[2]
        homography = np.array([[scale, 0., 0.], [0., scale, 0.], [0., 0., scale]])
        self.geom_scene_map = GeometricMap(self.scene_map, homography, self.map_origin)
        self.scene_vis_map = GeometricMap(self.scene_vis_map, homography, self.map_origin)

    def PreMotion(self, DataTuple: List[np.ndarray], valid_id: List[np.float32])\
            -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        motion = []
        mask = []
        for identity in valid_id:
            mask_i = torch.zeros(self.past_frames)
            box_3d = torch.zeros([self.past_frames, 2])
            for j in range(self.past_frames):
                past_data = DataTuple[j]              # past_data
                if len(past_data) > 0 and identity in past_data[:, 1]:
                    found_data = past_data[past_data[:, 1] == identity].squeeze()[[self.xind, self.zind]] / self.past_traj_scale
                    box_3d[self.past_frames-1 - j, :] = torch.from_numpy(found_data).float()
                    mask_i[self.past_frames-1 - j] = 1.0
                elif j > 0:
                    box_3d[self.past_frames-1 - j, :] = box_3d[self.past_frames - j, :]    # if none, copy from previous
                else:
                    raise ValueError('current id missing in the first frame!')
            motion.append(box_3d)
            mask.append(mask_i)
        return motion, mask

    def FutureMotion(self, DataTuple: List[np.ndarray], valid_id: List[np.float32])\
            -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        motion = []
        mask = []
        for identity in valid_id:
            mask_i = torch.zeros(self.future_frames)
            pos_3d = torch.zeros([self.future_frames, 2])
            for j in range(self.future_frames):
                fut_data = DataTuple[j]              # cur_data
                if len(fut_data) > 0 and identity in fut_data[:, 1]:
                    found_data = fut_data[fut_data[:, 1] == identity].squeeze()[[self.xind, self.zind]] / self.traj_scale
                    pos_3d[j, :] = torch.from_numpy(found_data).float()
                    mask_i[j] = 1.0
                elif j > 0:
                    pos_3d[j, :] = pos_3d[j - 1, :]    # if none, copy from previous
                else:
                    raise ValueError('current id missing in the first frame!')
            motion.append(pos_3d)
            mask.append(mask_i)
        return motion, mask

    def __call__(self, frame: np.int64) -> Union[dict, None]:
        """
        generate the data for an observation window
        :param frame: the timestep corresponding to t0
        """

        assert 0 <= frame - self.init_frame <= self.TotalFrame() - 1,\
            'frame is %d, total is %d' % (frame, self.TotalFrame())

        pre_data = self.PreData(frame)
        fut_data = self.FutureData(frame)
        valid_id = self.get_valid_id(pre_data, fut_data)

        if len(pre_data[0]) == 0 or len(fut_data[0]) == 0 or len(valid_id) == 0:
            return None

        if self.dataset == 'nuscenes_pred':
            pred_mask = self.get_pred_mask(pre_data[0], valid_id)
            heading = self.get_heading(pre_data[0], valid_id)
        else:
            pred_mask = None
            heading = None

        pre_motion_3D, pre_motion_mask = self.PreMotion(pre_data, valid_id)
        fut_motion_3D, fut_motion_mask = self.FutureMotion(fut_data, valid_id)

        data = {
            'pre_motion_3D': pre_motion_3D,
            'fut_motion_3D': fut_motion_3D,
            'fut_motion_mask': fut_motion_mask,
            'pre_motion_mask': pre_motion_mask,
            'pre_data': pre_data,
            'fut_data': fut_data,
            'heading': heading,
            'valid_id': valid_id,
            'traj_scale': self.traj_scale,
            'pred_mask': pred_mask,
            'scene_map': self.geom_scene_map,
            'seq': self.seq_name,
            'frame': frame
        }

        return data
