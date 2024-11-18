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

    def __init__(self, map_image: Tensor, homography: Tensor = torch.eye(3)):
        self.image = map_image          # [C, H, W]
        self.homography = homography    # [3, 3]

    def get_map_dimensions(self):
        return self.image.shape[1:]     # [H, W]

    def homography_translation(self, point: Tensor) -> None:
        # point [2]
        self.homography[:2, 2] += point

    def homography_scaling(self, factor: float) -> None:
        self.homography[0, 0] *= factor
        self.homography[1, 1] *= factor

    def rotate_around_center(self, theta: float) -> None:
        self.image = torchvision.transforms.functional.rotate(self.image, angle=theta)
        cy, cx = self.get_map_dimensions()

        cx *= 0.5
        cy *= 0.5

        # angle is in degrees, needs to be converted to rad first, we apply the negative of the angle to remain
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

    def to_map_points(self, points: Tensor) -> Tensor:
        # points [*, 2]
        homogeneous_points = torch.cat((points, torch.ones([*points.shape[:-1], 1])), dim=-1).transpose(-1, -2)
        mapped_points = (self.homography @ homogeneous_points).transpose(-1, -2)[..., :-1]
        return mapped_points

    def set_homography(self, matrix: Tensor) -> None:
        self.homography = matrix


class HomographyMatrix:

    def __init__(self, matrix: Tensor = torch.eye(3)):
        self._frame = matrix    # [3, 3]

    def set_homography(self, matrix: Tensor) -> None:
        self._frame = matrix

    def translate(self, point: Tensor) -> None:
        # point [2]
        self._frame[:2, 2] += point

    def scale(self, factor: float) -> None:
        self._frame[0, 0] *= factor
        self._frame[1, 1] *= factor

    def rotate(self, theta: float) -> None:
        # theta expressed in radians

        cos = np.cos(theta)
        sin = np.sin(theta)

        rotation_matrix = torch.Tensor(
            [[cos, -sin, 0.],
             [sin, cos, 0.],
             [0., 0., 1.]]
        )
        self._frame = rotation_matrix @ self._frame

    def rotate_about(self, point: Tensor, theta: float) -> None:
        # point [2]
        # theta expressed in radians
        p_x, p_y = point[...]

        cos = np.cos(theta)
        sin = np.sin(theta)

        rotation_matrix = torch.Tensor(
            [[cos, -sin, -cos * p_x + sin * p_y + p_x],
             [sin, cos, -sin * p_x - cos * p_y + p_y],
             [0., 0., 1.]]
        )
        self._frame = rotation_matrix @ self._frame

    def transform_points(self, points: Tensor) -> Tensor:
        # points [*, 2]
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

        (Image.open is a lazy operation)
        https://pillow.readthedocs.io/en/stable/reference/Image.html#functions
        """
        self._resolution = Image.open(image_path).size[::-1]         # [H, W]
        self._data = None

    def get_data(self) -> Optional[Tensor]:
        return self._data

    def get_resolution(self) -> Size:
        return Size(self._resolution)

    def crop(self, crop_coords: Tensor, resolution: int) -> None:
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

    def get_resolution(self) -> Size:
        return self._data.shape[1:]       # [H, W]

    def crop(self, crop_coords: Tensor, resolution: int) -> None:
        # crop_coords [2, 2]
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

    def get_map_dimensions(self) -> Size:
        return self._map.get_resolution()       # [H, W]

    def homography_translation(self, point: Tensor) -> None:
        # point [2]
        self._homography.translate(point=point)

    def homography_scaling(self, factor: float) -> None:
        self._homography.scale(factor=factor)

    def rotate_around_center(self, theta: float) -> None:
        # theta is expressed in *degrees*
        self._map.rotate_around_center(theta=theta)

        # print(f"{self.get_map_dimensions()=}")
        # print(f"{torch.tensor(self.get_map_dimensions())=}")
        center_point = (torch.tensor(self.get_map_dimensions()) * 0.5).flip(dims=(0,))      # [x, y]
        # cy, cx = self.get_map_dimensions()
        # cx *= 0.5
        # cy *= 0.5
        # center_point = Tensor([cx, cy])

        # converting theta to radians, and multiplying by -1
        # this is because the reference frame of the image is reversed
        # (the origin of the image is in the top left corner)
        # theta *= (-0.0055555555555555555555555555555556 * np.pi)
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
