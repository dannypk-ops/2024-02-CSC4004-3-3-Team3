#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal
from utils.general_utils import PILtoTorch
import cv2

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False,
                 crack_points = None,
                 depth_path = None, normal_path=None
                 ):
        super(Camera, self).__init__()

        self.scale = scale
        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        if depth_path is not None and normal_path is not None:
            self.depth_map = self.read_array(depth_path)
            self.normal_map = self.read_array(normal_path)

        # crack Point Information
        if crack_points is not None and crack_points != []:
            if self.scale != 1.0:
                new_pixels = crack_points[0]['pixels']
                downsampled_pixels = [[int(x//self.scale), int(y//self.scale)] for x, y in new_pixels]
                crack_points[0]['pixels'] = downsampled_pixels
            self.cracked_points = crack_points
        else:
            self.cracked_points = None

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        if image is not None:
            resized_image_rgb = PILtoTorch(image, resolution)
            gt_image = resized_image_rgb[:3, ...]
            self.alpha_mask = None
            self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
            if resized_image_rgb.shape[0] == 4:
                self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
            else: 
                self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        # self.image_width = self.original_image.shape[2]
        # self.image_height = self.original_image.shape[1]
        self.image_width = resolution[0]
        self.image_height = resolution[1]

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.alpha_mask)
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        # camera intrinsic
        self.intrinsic = np.zeros((3, 3))
        self.intrinsic[0, 0] = fov2focal(FoVx, self.image_width)
        self.intrinsic[1, 1] = fov2focal(FoVy, self.image_height)
        self.intrinsic[0, 2] = self.image_width / 2
        self.intrinsic[1, 2] = self.image_height / 2
        self.intrinsic[2, 2] = 1.0

        # camera extrinsic
        self.w2c = np.zeros((4, 4))
        self.w2c[:3, :3] = self.R
        self.w2c[:3, 3] = self.T
        self.w2c[3, 3] = 1.0

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    def get_cracked_points(self):
        return self.cracked_points
        
    def read_array(self, path):
        """Reads a depth or normal map from a binary file."""
        with open(path, "rb") as fid:
            # Read width, height, and channels (header info)
            width, height, channels = np.genfromtxt(
                fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
            )
            # Skip the header to reach the binary data
            fid.seek(0)
            num_delimiter = 0
            while True:
                byte = fid.read(1)
                if byte == b"&":
                    num_delimiter += 1
                    if num_delimiter >= 3:
                        break
            
            # Read the binary data as a float32 array
            array = np.fromfile(fid, np.float32)

        # Reshape and reorder the array
        array = array.reshape((width, height, channels), order="F")
        return np.transpose(array, (1, 0, 2)).squeeze()


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

