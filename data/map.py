import cv2
import numpy as np
from matplotlib.path import Path
import os
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn.functional as fctl
import torchvision.transforms.functional
from PIL import Image

from typing import Callable, Optional
Tensor = torch.Tensor
softmax = fctl.softmax
log_softmax = fctl.log_softmax
Size = torch.Size


def compute_occlusion_map(
        map_dimensions: Size,                       # [H, W]
        visibility_polygon_coordinates: Tensor      # [*, 2]
) -> Tensor:                                        # [H, W]
    occ_y = torch.arange(map_dimensions[0])
    occ_x = torch.arange(map_dimensions[1])
    xy = torch.dstack((torch.meshgrid(occ_x, occ_y))).reshape((-1, 2))
    mpath = Path(visibility_polygon_coordinates)
    return torch.from_numpy(mpath.contains_points(xy).reshape(map_dimensions)).to(torch.bool).T


def compute_distance_transformed_map(
        occlusion_map: Tensor,      # [H, W]
        scaling: float = 1.0
) -> Tensor:                        # [H, W]
    return (torch.where(
        ~occlusion_map,
        torch.from_numpy(-distance_transform_edt(~occlusion_map)),
        torch.from_numpy(distance_transform_edt(occlusion_map))
    ) * scaling).to(torch.float32)


def compute_clipped_map(
        map_tensor: Tensor      # [H, W]
) -> Tensor:                    # [H, W]
    return -torch.clamp(map_tensor, min=0.)


def apply_function_over_whole_map(
        map_tensor: Tensor,                     # [H, W]
        function: Callable[[Tensor], Tensor]
) -> Tensor:
    return function(map_tensor.view(-1)).view(map_tensor.shape)


def compute_probability_map(dt_map: Tensor) -> Tensor:
    return apply_function_over_whole_map(dt_map, lambda x: softmax(compute_clipped_map(x), dim=0))


def compute_nlog_probability_map(dt_map: Tensor) -> Tensor:
    return apply_function_over_whole_map(dt_map, lambda x: -log_softmax(compute_clipped_map(x), dim=0))


def apply_homography(
        points: Tensor,         # [*, 2]
        homography: Tensor      # [3, 3]
) -> Tensor:                    # [*, 2]
    homogeneous_points = torch.cat((points, torch.ones([*points.shape[:-1], 1])), dim=-1).transpose(-1, -2)
    return (homography @ homogeneous_points).transpose(-1, -2)[..., :-1]


class HomographyMatrix:

    def __init__(
            self,
            matrix: Tensor = torch.eye(3)   # [3, 3]
    ):
        self._frame = matrix

    def set_homography(self, matrix: Tensor) -> None:
        self._frame = matrix

    def translate(
            self,
            point: Tensor   # [2]
    ) -> None:
        self._frame[:2, 2] += point

    def scale(self, factor: float) -> None:
        self._frame[0, 0] *= factor
        self._frame[1, 1] *= factor

    def rotate(
            self,
            theta: float    # [radians]
    ) -> None:
        cos = np.cos(theta)
        sin = np.sin(theta)

        rotation_matrix = torch.Tensor(
            [[cos, -sin, 0.],
             [sin, cos, 0.],
             [0., 0., 1.]]
        )
        self._frame = rotation_matrix @ self._frame

    def rotate_about(
            self,
            point: Tensor,  # [2]
            theta: float    # [radians]
    ) -> None:
        p_x, p_y = point[...]

        cos = np.cos(theta)
        sin = np.sin(theta)

        rotation_matrix = torch.Tensor(
            [[cos, -sin, -cos * p_x + sin * p_y + p_x],
             [sin, cos, -sin * p_x - cos * p_y + p_y],
             [0., 0., 1.]]
        )
        self._frame = rotation_matrix @ self._frame

    def transform_points(
            self,
            points: Tensor  # [*, 2]
    ) -> Tensor:            # [*, 2]
        return apply_homography(points=points, homography=self._frame)


class BaseMap:

    def get_data(self) -> Optional[Tensor]:
        raise NotImplementedError

    def get_resolution(self) -> Size:
        raise NotImplementedError

    def crop(self, crop_coords: Tensor, resolution: int) -> None:
        raise NotImplementedError

    def rotate_around_center(self, theta: float) -> None:
        raise NotImplementedError


class PILMap(BaseMap):
    def __init__(self, image_path: os.PathLike):
        """
        This implementation won't actually load the image contained within <image_path>,
        but just operate based on the image's resolution.
        This class can be used to save processing time in use cases where the rgb scene map is unnecessary.

        (Image.open is a lazy operation)
        https://pillow.readthedocs.io/en/stable/reference/Image.html#functions
        """
        self._resolution = Image.open(image_path).size[::-1]         # [H, W]
        self._data = None

    def get_data(self) -> Optional[Tensor]:
        return self._data

    def get_resolution(self) -> Size:   # [H, W]
        return Size(self._resolution)

    def crop(
            self,
            crop_coords: Tensor,    # [2, 2]
            resolution: int
    ) -> None:
        self._resolution = (resolution, resolution)

    def rotate_around_center(self, theta: float) -> None:
        pass


class TensorMap(BaseMap):

    convert_to_tensor = torchvision.transforms.ToTensor()

    def __init__(self, image_path: os.PathLike):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.convert_to_tensor(image)
        self._data = image       # [C, H, W]

    def get_data(self) -> Optional[Tensor]:
        return self._data

    def get_resolution(self) -> Size:   # [H, W]
        return self._data.shape[1:]

    def crop(
            self,
            crop_coords: Tensor,    # [2, 2]
            resolution: int
    ) -> None:
        top = torch.round(crop_coords[0, 1]).to(torch.int64)
        left = torch.round(crop_coords[0, 0]).to(torch.int64)
        side = torch.round(crop_coords[1, 0] - crop_coords[0, 0]).to(torch.int64)

        self._data = torchvision.transforms.functional.resized_crop(
            self._data, top=top, left=left, height=side, width=side, size=resolution
        )

    def rotate_around_center(self, theta: float) -> None:
        self._data = torchvision.transforms.functional.rotate(self._data, angle=theta)


class MapManager:
    """
    This class is a map manager, whose purpose is to contain a map, and a corresponding homography matrix.
    """
    def __init__(self, map_object: BaseMap, homography: HomographyMatrix):
        self._map = map_object
        self._homography = homography

    def get_map(self) -> Optional[Tensor]:
        return self._map.get_data()

    def get_map_dimensions(self) -> Size:   # [H, W]
        return self._map.get_resolution()

    def homography_translation(self, point: Tensor) -> None:
        # point [2]
        self._homography.translate(point=point)

    def homography_scaling(self, factor: float) -> None:
        self._homography.scale(factor=factor)

    def rotate_around_center(
            self,
            theta: float    # [degrees]
    ) -> None:
        self._map.rotate_around_center(theta=theta)

        center_point = (torch.tensor(self.get_map_dimensions(), dtype=torch.float64) * 0.5).flip(dims=(0,))     # [x, y]

        # converting theta to radians, and multiplying by -1
        # this is because the reference frame of the image is reversed
        # (the origin of the image is in the top left corner)
        self._homography.rotate_about(
            point=center_point,
            theta=(-theta * np.pi * 0.0055555555555555555555555555555556)
        )

    def map_cropping(self, crop_coordinates: Tensor, resolution: int) -> None:
        self._map.crop(crop_coords=crop_coordinates, resolution=resolution)

    def to_map_points(self, points: Tensor) -> Tensor:
        return self._homography.transform_points(points=points)

    def set_homography(self, matrix: Tensor) -> None:
        self._homography.set_homography(matrix=matrix)


MAP_DICT = {
    True: TensorMap,
    False: PILMap
}
