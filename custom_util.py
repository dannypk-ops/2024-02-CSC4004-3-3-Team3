from utils.sh_utils import eval_sh
from scene.gaussian_model import GaussianModel
import open3d as o3d
import torch
import numpy as np


import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]  

def sh2RGB(pc : GaussianModel, viewpoint_camera):
    shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

    return colors_precomp

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def create_point(Point, Color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(Point)
    pcd.colors = o3d.utility.Vector3dVector(Color)

    return pcd

CAM_POINTS = np.array([
        [ 0,   0,   0],
        [-1,  -1, 1.5],
        [ 1,  -1, 1.5],
        [ 1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [ 0.5, 1, 1.5],
        [ 0, 1.2, 1.5]])

CAM_LINES = np.array([
    [1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4], [5,7], [7,6]])

def create_camera_actor(g, scale=0.05):
    """ build open3d camera polydata """
    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES))

    color = (g * 1.0, 0.5 * (1-g), 0.9 * (1-g))
    camera_actor.paint_uniform_color(color)
    return camera_actor

def vis_gs(pc, cam=None, save=False, mask=None):
    from utils.sh_utils import eval_sh
    pcd = o3d.geometry.PointCloud()
    
    if mask is not None:
        shs_view = pc.get_features[mask].transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        dir_pp = (pc.get_xyz)[mask]
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        
        point = o3d.utility.Vector3dVector(pc.get_xyz[mask].detach().cpu().numpy())
        color = o3d.utility.Vector3dVector(colors_precomp.detach().cpu().numpy())
        
        pcd.points = point
        pcd.colors = color
    
    else:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        dir_pp = (pc.get_xyz)
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        
        point = o3d.utility.Vector3dVector(pc.get_xyz.detach().cpu().numpy())
        color = o3d.utility.Vector3dVector(colors_precomp.detach().cpu().numpy())
        
        pcd.points = point
        pcd.colors = color
    
    if cam is not None:
        total_list = [pcd]
        if isinstance(cam, list):
            for c in cam:
                cam_actor = create_camera_actor(True)
                ref_pose = np.eye(4)
                R, T = c.R, c.T
                ref_pose[:3, :3] = R
                ref_pose[:3, 3] = T
                
                ref_pose = np.linalg.inv(ref_pose)
                cam_actor.transform(ref_pose)
                # cam_actor.rotate(R)
                # cam_actor.translate(T)
                total_list.append(cam_actor)
        else :
            cam_actor = create_camera_actor(True)
            c = cam
            ref_pose = np.eye(4)
            R, T = c.R, c.T
            ref_pose[:3, :3] = R
            ref_pose[:3, 3] = T
            
            ref_pose = np.linalg.inv(ref_pose)
            cam_actor.transform(ref_pose)
            # cam_actor.rotate(R)
            # cam_actor.translate(T)
            total_list.append(cam_actor)
        return o3d.visualization.draw_geometries(total_list)
    
    if save:
        o3d.io.write_point_cloud("test.ply", pcd)
        
    o3d.visualization.draw_geometries([pcd])


def save_numpy_img(image, path):
    import numpy as np
    from PIL import Image

    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(path)
