import cv2
import numpy as np
import os
import torch
import torchvision.transforms.functional
from PIL import Image

from typing import Optional
Tensor = torch.Tensor
Size = torch.Size


class TorchGeometricMap:

    def __init__(
            self,
            map_image: Tensor,                  # [C, H, W]
            homography: Tensor = torch.eye(3)   # [3, 3]
    ):
        self.image = map_image
        self.homography = homography

    def get_map_dimensions(self):
        return self.image.shape[1:]     # [H, W]

    def homography_translation(
            self,
            point: Tensor   # [2]
    ) -> None:
        self.homography[:2, 2] += point

    def homography_scaling(self, factor: float) -> None:
        self.homography[0, 0] *= factor
        self.homography[1, 1] *= factor

    def rotate_around_center(
            self,
            theta: float    # [degrees]
    ) -> None:
        self.image = torchvision.transforms.functional.rotate(self.image, angle=theta)
        cy, cx = self.get_map_dimensions()

        cx *= 0.5
        cy *= 0.5

        # angle needs to be converted to rad first, we apply the negative of the angle to remain
        # consistent with the image representation of the scene.
        # due to the fact the coordinate system of the image is flipped (origin in the top left corner).
        cos = np.cos(-theta * np.pi * 0.0055555555555555555555555555555556)
        sin = np.sin(-theta * np.pi * 0.0055555555555555555555555555555556)

        matrix = torch.Tensor(
            [[cos, -sin, -cos * cx + sin * cy + cx],
             [sin, cos, -sin * cx - cos * cy + cy],
             [0., 0., 1.]]
        )
        self.homography = matrix @ self.homography

    def crop(self, crop_coords: Tensor, resolution: int):
        """
        Careful: This method is not responsible for changing the homography matrix
        after the cropping has been performed.
        To maintain consistency, you need to compute the corresponding homography and use set_homography.
        """
        top = torch.round(crop_coords[0, 1]).to(torch.int64)
        left = torch.round(crop_coords[0, 0]).to(torch.int64)
        side = torch.round(crop_coords[1, 0] - crop_coords[0, 0]).to(torch.int64)

        self.image = torchvision.transforms.functional.resized_crop(
            self.image, top=top, left=left, height=side, width=side, size=resolution
        )

    def to_map_points(
            self,
            points: Tensor      # [*, 2]
    ) -> Tensor:                # [*, 2]
        homogeneous_points = torch.cat((points, torch.ones([*points.shape[:-1], 1])), dim=-1).transpose(-1, -2)
        mapped_points = (self.homography @ homogeneous_points).transpose(-1, -2)[..., :-1]
        return mapped_points

    def set_homography(self, matrix: Tensor) -> None:
        self.homography = matrix


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
        homogeneous_points = torch.cat((points, torch.ones([*points.shape[:-1], 1])), dim=-1).transpose(-1, -2)
        mapped_points = (self._frame @ homogeneous_points).transpose(-1, -2)[..., :-1]
        return mapped_points


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
