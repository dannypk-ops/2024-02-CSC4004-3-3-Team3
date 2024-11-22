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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

import open3d as o3d
import numpy as np
from scene.cameras import Camera
import random

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation, original = False):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            if original:
                return actual_covariance
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.point_cloud = None

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling, 
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def get_covariance_original(self, scaling_modifier = 1):
        return self.covariance_activation(self._scaling, scaling_modifier, self._rotation, original=True)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def mark_crack_points(self, cam, modify = False, color = 'R'):
        # 해당 이미지 좌표에 대응되는 영역에 존재하는 가우시안들을 찾는다.
        # 해당 가우시안들의 색상을 변경한다. ( marking )
        mask = self.get_marked_gaussians(cam)

        if modify:
            self.modify_gaussians_color(mask, color)

        return mask
    
    def get_marked_gaussians(self, cam, distance_threshold = 0.7, epsilon=1e-8):
        import open3d as o3d
        from scipy.spatial import cKDTree

        means3D = self.get_xyz.detach().cpu().numpy()
        points_2D, points_camera = self.get_image_camera_coordinate_of_gaussian(cam)

        cracked_points = cam.cracked_points[0]['pixels']
        cracked_points = np.array(cracked_points)

        min_x, min_y = cracked_points.min(0)
        max_x, max_y = cracked_points.max(0)

        mark_range = (max_x - min_x, max_y - min_y)
        base_pixel = (min_x, min_y)

        padding = 30

        # 범위 설정
        x_range = (base_pixel[0] - padding, base_pixel[0] + mark_range[0] + padding)
        y_range = (base_pixel[1] - padding, base_pixel[1] + mark_range[1] + padding)

        marked_mask = np.zeros(points_2D.shape[0], dtype=bool)

        # 범위 내부에 있는 mask 생성
        marked_mask = (
            (points_2D[:, 0] >= x_range[0]) & (points_2D[:, 0] <= x_range[1]) &
            (points_2D[:, 1] >= y_range[0]) & (points_2D[:, 1] <= y_range[1])
        )   

        # 동일한 pixel에 해당하는 가우시안들에 대해서, density가 threshold보다 높은 애들만 masking하는 mask 생성.
        target_means3D = means3D[marked_mask]
        tree = cKDTree(means3D)
        counts = np.array([len(tree.query_ball_point(point, distance_threshold)) for point in target_means3D])
        
        valid_indices = np.where(counts >= 500)[0]
        marked_indices = np.where(marked_mask)[0]
        valid_indices_in_means3D = marked_indices[valid_indices]

        valid_mask = np.zeros(means3D.shape[0], dtype=bool)
        valid_mask[valid_indices_in_means3D] = True

        # depth를 고려하여, 가까이 있는(depth가 작은) Gaussian들만을 고려한다.
        depth = points_camera[:, 2]
        depth_map = cam.depth_map[cracked_points[:,0], cracked_points[:,1]]
        valid_num = np.sum(depth_map != 0)

        # 유효한 depth value의 mean
        if valid_num != 0:
            depth_value = depth_map.sum() / valid_num
        else:
            depth_value = 0
            
        if depth_value == 0:
            print('No valid depth')
            return np.array([])
        
        depth_mask = np.abs(depth - depth_value) < 1.0
        final_mask = np.logical_and(depth_mask, valid_mask)

        return final_mask

    def modify_gaussians_color(self, mask, color = 'R'):
        from custom_functions import RGB2SH

        if color == 'R':
            feature = torch.from_numpy(RGB2SH(np.array([1,0,0]))).unsqueeze(0)
        elif color == 'G':
            feature = torch.from_numpy(RGB2SH(np.array([0,1,0]))).unsqueeze(0)
        else:
            feature = torch.from_numpy(RGB2SH(np.array([0,0,1]))).unsqueeze(0)

        self._features_dc[mask] = feature.expand(mask.sum(), 1, 3).float().cuda()
        self._features_rest[mask] = feature.expand(mask.sum(), 15, 3).float().cuda()
        self._opacity[mask] = torch.zeros((mask.sum(), 1), device="cuda")

    def get_index(self, cam, mask, margin_error = 50):
        
        points2D, camera_points = self.get_image_camera_coordinate_of_gaussian(cam)
        points2D = torch.from_numpy(points2D).float().cuda()

        true_indices = torch.nonzero(mask, as_tuple=True)[0].tolist()

        min_depth = 1000
        point_index = 0
        point_index_list = []
        for index in true_indices:
            if camera_points[index, 2] < min_depth:
                min_depth = camera_points[index, 2]
                point_index = index
                point_index_list.append(index)

        return point_index_list, point_index 

    def novelViewRenderer(self, cam, mask, pipe):
        from gaussian_renderer import render

        # create open3d Point Cloud object.
        if self.point_cloud is None:
            self.point_cloud = o3d.geometry.PointCloud()
            self.point_cloud.points = o3d.utility.Vector3dVector(self.get_xyz.detach().cpu().numpy())
        
        R, T = self.create_camera_near_point(self.point_cloud, ref_cam = cam, distance=1.0, mask=mask)

        nvCamera = Camera((cam.depth_map.shape[0], cam.depth_map.shape[1]), colmap_id=cam.uid, R=R, T=T, 
                  FoVx=cam.FoVx, FoVy=cam.FoVy, depth_params=None,
                  image=None, invdepthmap=None, depth_path=None, normal_path=None,
                  image_name=f"{cam.image_name}_based_novel_view", uid=cam.uid, data_device='cuda',
                  train_test_exp=False, is_test_dataset=False, is_test_view=False)
        
        # rendering
        render_pkg = render(nvCamera, self, pipe, torch.zeros(3).cuda(), use_trained_exp=False, separate_sh=False)
        
        return render_pkg["render"].detach().cpu().numpy().transpose(1,2,0)

    def create_camera_near_point(self, point_cloud, ref_cam, distance, mask=None):
        """
        특정 포인트의 노멀 벡터를 기준으로 일정 거리만큼 떨어진 위치에 카메라를 생성합니다.
        
        Args:
            point_cloud (o3d.geometry.PointCloud): Open3D Point Cloud 객체
            ref_cam (np.ndarray): 기준 카메라의 회전 및 변환 정보를 담은 4x4 변환 행렬
            distance (float): 타겟 포인트와 카메라 간의 거리
            mask (np.ndarray): 관심 있는 포인트의 마스크
        Returns:
            transform_matrix (np.ndarray): 4x4 동차 변환 행렬 (world2cam)
        """
        # 노멀 벡터 계산 (필요 시)
        if point_cloud.has_normals():
            pcd = point_cloud
        else:   
            pcd = self.compute_normals_with_pca(point_cloud)

        # target_point = np.asarray(pcd.points)[mask].mean(0)
        target_point = np.asarray(pcd.points)[mask][-1]

        # 마스크된 포인트들의 평균 노멀 벡터를 사용
        normal_vector = np.asarray(pcd.normals)[mask].mean(0)

        # 카메라 위치: target_point에서 normal_vector 방향으로 distance 떨어진 위치
        camera_position = target_point - normal_vector * distance

        # z_axis 설정: 카메라가 target_point를 정확히 바라보도록 설정
        z_axis = (target_point - camera_position)
        z_axis /= np.linalg.norm(z_axis)  # 단위 벡터로 정규화
        
        # 기준 카메라의 상향 벡터 추출
        ref_z_vector = ref_cam.R[:3, 2]  # ref_cam의 y축을 기준으로 상향 벡터 설정
        ref_z_vector /= np.linalg.norm(ref_z_vector)

        ref_y_vector = ref_cam.R[:3, 1]
        ref_x_vector = ref_cam.R[:3, 0]

        # x_axis와 y_axis 계산
        y_axis = np.cross(z_axis, ref_z_vector)

        if np.dot(y_axis, ref_y_vector) < 0:
            y_axis = -y_axis
        y_axis /= np.linalg.norm(y_axis)

        x_axis = np.cross(z_axis, y_axis)
        if np.dot(x_axis, ref_x_vector) < 0:
            x_axis = -x_axis

        # 3x3 회전 행렬 생성 (카메라 좌표계의 방향 설정)
        rotation_matrix = np.vstack((x_axis, y_axis, z_axis)).T

        # 4x4 동차 변환 행렬 생성
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = -rotation_matrix @ camera_position  # translation 적용 (world 좌표계를 cam 좌표계로 변환)

        # return rotation_matrix, -rotation_matrix @ camera_position
        return rotation_matrix, -camera_position @ rotation_matrix

    
    def get_image_camera_coordinate_of_gaussian(self, cam):
        means3D = self.get_xyz.detach().cpu().numpy()
        homo_means3D = np.hstack([means3D, np.ones((means3D.shape[0], 1))])

        # Cam information
        intrinsic = cam.intrinsic
        R, T = cam.R, cam.T
        ref_pose = np.eye(4)
        ref_pose[:3, :3] = R
        ref_pose[:3, 3] = T

        # world to camera
        points_camera = (ref_pose @ homo_means3D.T).T[:, :3]

        # camera to image
        points_2D_homogeneous = (intrinsic @ points_camera.T).T

        # Normalize by the third coordinate to get (x, y)
        points_2D = points_2D_homogeneous[:, :2] / points_2D_homogeneous[:, 2].reshape(-1,1)

        return points_2D.astype(int), points_camera
    

    def compute_normals_with_pca(self, point_cloud, radii=[1.0], max_nn=100):
        import numpy as np
        import open3d as o3d
        from sklearn.decomposition import PCA
        """
        다양한 반경에서 PCA를 사용하여 멀티 스케일 노멀 벡터를 계산합니다.
        
        Args:
            point_cloud (o3d.geometry.PointCloud): Open3D Point Cloud 객체
            radii (list of float): 다양한 반경을 리스트로 지정
            max_nn (int): 각 반경에서 사용할 최대 이웃 점 개수
            
        Returns:
            point_cloud (o3d.geometry.PointCloud): 최종 노멀 벡터를 가진 Point Cloud
        """
        # KD-Tree 생성
        pcd_tree = o3d.geometry.KDTreeFlann(point_cloud)
        normals = []

        for i, point in enumerate(point_cloud.points):
            multi_scale_normals = []

            # 각 반경에 대해 노멀 벡터 계산
            for radius in radii:
                _, idx, _ = pcd_tree.search_hybrid_vector_3d(point, radius, max_nn)
                
                if len(idx) >= 3:  # 최소한 3개의 이웃이 필요함
                    neighbor_points = np.asarray(point_cloud.points)[idx]
                    
                    # PCA로 노멀 벡터 계산
                    pca = PCA(n_components=3)
                    pca.fit(neighbor_points)
                    
                    # 가장 작은 고유값에 해당하는 고유 벡터를 노멀 벡터로 사용
                    normal_vector = pca.components_[-1]
                    multi_scale_normals.append(normal_vector)
            
            # 각 반경의 노멀 벡터를 가중 평균하여 최종 노멀을 생성 (가중치가 필요할 경우 수정 가능)
            if multi_scale_normals:
                averaged_normal = np.mean(multi_scale_normals, axis=0)
                averaged_normal /= np.linalg.norm(averaged_normal)  # 정규화
                normals.append(averaged_normal)
            else:
                normals.append([0, 0, 0])

        # 결과를 Open3D 포맷으로 변환하여 노멀 벡터 업데이트
        point_cloud.normals = o3d.utility.Vector3dVector(normals)
        
        return point_cloud