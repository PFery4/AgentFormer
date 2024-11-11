import torch
import numpy as np

Tensor = torch.Tensor
import torchvision.transforms.functional
from PIL import Image
import cv2
import os
from data.homography_warper import get_rotation_matrix2d, warp_affine_crop


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

    def __init__(self, homography: Tensor = torch.eye(3)):
        self.homography = homography    # [3, 3]

    def set_homography(self, homography: Tensor) -> None:
        self.homography = homography

    def translate(self, point: Tensor) -> None:
        # point [2]
        self.homography[:2, 2] += point

    def scale(self, factor: float) -> None:
        self.homography[0, 0] *= factor
        self.homography[1, 1] *= factor

    def rotate(self, theta: float) -> None:
        # theta expressed in radians

        cos = np.cos(theta)
        sin = np.sin(theta)

        rotation_matrix = torch.Tensor(
            [[cos, -sin, 0.],
             [sin, cos, 0.],
             [0., 0., 1.]]
        )
        self.homography = rotation_matrix @ self.homography

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
        self.homography = rotation_matrix @ self.homography

    def transform_points(self, points: Tensor) -> Tensor:
        # points [*, 2]
        homogeneous_points = torch.cat((points, torch.ones([*points.shape[:-1], 1])), dim=-1).transpose(-1, -2)
        mapped_points = (self.homography @ homogeneous_points).transpose(-1, -2)[..., :-1]
        return mapped_points


class BaseMap:
    def get_resolution(self) -> Tensor:
        raise NotImplementedError

    def crop(self, crop_coords: Tensor, resolution: int) -> None:
        raise NotImplementedError

    def rotate_around_center(self, theta: float) -> None:
        raise NotImplementedError


class DummyMap(BaseMap):
    def __init__(self, image_path: os.PathLike):
        """
        This implementation won't actually load the image contained within <image_path>,
        but just operate based on the image's resolution.
        """
        self.map_resolution = Image.open(image_path).size[::-1]         # [H, W]

    def get_resolution(self) -> Tensor:
        return torch.Tensor(self.map_resolution)

    def crop(self, crop_coords: Tensor, resolution: int) -> None:
        self.map_resolution = (resolution, resolution)

    def rotate_around_center(self, theta: float) -> None:
        pass


class TensorMap(BaseMap):
    def __init__(self, map_tensor: Tensor):
        self.map_data = map_tensor       # [C, H, W]

    def get_resolution(self) -> Tensor:
        return self.map_data.shape[1:]       # [H, W]

    def crop(self, crop_coords: Tensor, resolution: int) -> None:
        # crop_coords [2, 2]
        top = torch.round(crop_coords[0, 1]).to(torch.int64)
        left = torch.round(crop_coords[0, 0]).to(torch.int64)
        side = torch.round(crop_coords[1, 0] - crop_coords[0, 0]).to(torch.int64)

        self.map_data = torchvision.transforms.functional.resized_crop(
            self.map_data, top=top, left=left, height=side, width=side, size=resolution
        )

    def rotate_around_center(self, theta: float) -> None:
        self.map_data = torchvision.transforms.functional.rotate(self.map_data, angle=theta)
