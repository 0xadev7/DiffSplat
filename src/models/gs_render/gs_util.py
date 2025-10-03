from typing import *
from torch import Tensor

import os
import numpy as np
from plyfile import PlyData, PlyElement
import torch
from kiui.op import inverse_sigmoid

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

SH_C0 = 0.28209479177387814

class Camera:
    def __init__(self,
        C2W: Tensor, fxfycxcy: Tensor, h: int, w: int,
        znear: float = 0.01, zfar: float = 100.,
    ):
        self.fxfycxcy = fxfycxcy.clone().float()
        self.C2W = C2W.clone().float()
        self.W2C = self.C2W.inverse()

        self.znear = znear
        self.zfar = zfar
        self.h = h
        self.w = w

        fx, fy, cx, cy = self.fxfycxcy[0], self.fxfycxcy[1], self.fxfycxcy[2], self.fxfycxcy[3]
        self.tanfovX = 1 / (2 * fx)  # `tanHalfFovX` actually
        self.tanfovY = 1 / (2 * fy)  # `tanHalfFovY` actually
        self.fovX = 2 * torch.atan(self.tanfovX)
        self.fovY = 2 * torch.atan(self.tanfovY)
        self.shiftX = 2 * cx - 1
        self.shiftY = 2 * cy - 1

        def getProjectionMatrix(znear, zfar, fovX, fovY, shiftX, shiftY):
            tanHalfFovY = torch.tan((fovY / 2))
            tanHalfFovX = torch.tan((fovX / 2))

            top = tanHalfFovY * znear
            bottom = -top
            right = tanHalfFovX * znear
            left = -right

            P = torch.zeros(4, 4, device=fovX.device)

            z_sign = 1

            P[0, 0] = 2 * znear / (right - left)
            P[1, 1] = 2 * znear / (top - bottom)
            P[0, 2] = (right + left) / (right - left) + shiftX
            P[1, 2] = (top + bottom) / (top - bottom) + shiftY
            P[3, 2] = z_sign
            P[2, 2] = z_sign * zfar / (zfar - znear)
            P[2, 3] = -(zfar * znear) / (zfar - znear)
            return P

        self.world_view_transform = self.W2C.transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(self.znear, self.zfar, self.fovX, self.fovY, self.shiftX, self.shiftY).transpose(0, 1)
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = self.C2W[:3, 3]


class GaussianModel:
    
    def __init__(self):
        self.xyz = None
        self.rgb = None
        self.scale = None
        self.rotation = None
        self.opacity = None

        self.sh_degree = 0

    def _attributes_like_second(self, num_rest: int) -> list[tuple[str, str]]:
        # Build dtype list exactly like the Second model
        dtype_list = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
            ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ]
        for i in range(num_rest):
            dtype_list.append((f"f_rest_{i}", "f4"))
        dtype_list.append(("opacity", "f4"))
        dtype_list.extend([("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4")])
        dtype_list.extend([("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")])
        return dtype_list

    def _num_f_rest(self) -> int:
        # Match the Second model layout
        L = (self.sh_degree + 1) ** 2
        return 3 * max(L - 1, 0)  # 3 channels for all non-DC SH coeffs

    def set_data(self, xyz: Tensor, rgb: Tensor, scale: Tensor, rotation: Tensor, opacity: Tensor):
        self.xyz = xyz
        self.rgb = rgb
        self.scale = scale
        self.rotation = rotation
        self.opacity = opacity
        return self

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> "GaussianModel":
        self.xyz = self.xyz.to(device, dtype)
        self.rgb = self.rgb.to(device, dtype)
        self.scale = self.scale.to(device, dtype)
        self.rotation = self.rotation.to(device, dtype)
        self.opacity = self.opacity.to(device, dtype)
        return self

    def save_ply(self, path: str, opacity_threshold: float = 0., compatible: bool = True):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self.xyz.detach().cpu().numpy()
        f_dc = self.rgb.detach().cpu().numpy()
        rgb = (f_dc * 255.).clip(0., 255.).astype(np.uint8)
        opacity = self.opacity.detach().cpu().numpy()
        scale = self.scale.detach().cpu().numpy()
        rotation = self.rotation.detach().cpu().numpy()

        # Filter out points with low opacity
        mask = (opacity > opacity_threshold).squeeze()
        xyz = xyz[mask]
        f_dc = f_dc[mask]
        opacity = opacity[mask]
        scale = scale[mask]
        rotation = rotation[mask]
        rgb = rgb[mask]

        # Invert activation to make it compatible with the original ply format
        if compatible:
            opacity = inverse_sigmoid(torch.from_numpy(opacity)).numpy()
            scale = torch.log(torch.from_numpy(scale) + 1e-8).numpy()
            f_dc = (torch.from_numpy(f_dc) - 0.5).numpy() / 0.28209479177387814

        dtype_full = [(attribute, "f4") for attribute in self._construct_list_of_attributes()]
        dtype_full.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, f_dc, opacity, scale, rotation, rgb), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def save_ply_buffer(self, buffer, opacity_threshold: float = 0., compatible: bool = True):
        xyz = self.xyz.detach().cpu().numpy()
        f_dc = self.rgb.detach().cpu().numpy()
        rgb = (f_dc * 255.).clip(0., 255.).astype(np.uint8)
        opacity = self.opacity.detach().cpu().numpy()
        scale = self.scale.detach().cpu().numpy()
        rotation = self.rotation.detach().cpu().numpy()

        # Filter out points with low opacity
        mask = (opacity > opacity_threshold).squeeze()
        xyz = xyz[mask]
        f_dc = f_dc[mask]
        opacity = opacity[mask]
        scale = scale[mask]
        rotation = rotation[mask]
        rgb = rgb[mask]

        # Invert activation to make it compatible with the original ply format
        if compatible:
            opacity = inverse_sigmoid(torch.from_numpy(opacity)).numpy()
            scale = torch.log(torch.from_numpy(scale) + 1e-8).numpy()
            f_dc = (torch.from_numpy(f_dc) - 0.5).numpy() / 0.28209479177387814

        dtype_full = [(attribute, "f4") for attribute in self._construct_list_of_attributes()]
        dtype_full.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, f_dc, opacity, scale, rotation, rgb), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(buffer)

    def save_ply_buffer_sn17(self, buffer, opacity_threshold: float = 0.0):
        """
        Write a PLY identical in schema & value conventions to the Second model.
        `target` can be a file path (str/Path) or an open binary file-like object.
        """
        import numpy as np
        import torch
        from plyfile import PlyData, PlyElement

        # Pull tensors
        xyz = self.xyz.detach().cpu().numpy()            # (N, 3)
        rgb = self.rgb.detach().cpu().numpy()            # (N, 3) assumed colors_precomp in [0,1]
        opacity = self.opacity.detach().cpu().numpy()    # (N, 1) or (N,)
        scale = self.scale.detach().cpu().numpy()        # (N, 3) linear domain
        rotation = self.rotation.detach().cpu().numpy()  # (N, 4)

        # Flatten opacity shape
        if opacity.ndim == 2 and opacity.shape[1] == 1:
            opacity = opacity[:, 0]

        # Apply mask BEFORE inversions (threshold in [0,1] domain)
        mask = (opacity > opacity_threshold)
        xyz = xyz[mask]
        rgb = rgb[mask]
        opacity = opacity[mask]
        scale = scale[mask]
        rotation = rotation[mask]

        N = xyz.shape[0]
        if N == 0:
            raise ValueError("All points were filtered out by opacity_threshold.")

        # Normals (zeros) to match the Second model
        normals = np.zeros_like(xyz, dtype=np.float32)   # (N, 3)

        # ---- Value space conversions to match Second model ----
        # f_dc from baked RGB (colors_precomp): rgb = SH_C0 * f_dc + 0.5
        f_dc = ((torch.from_numpy(rgb).float() - 0.5) / SH_C0).numpy().astype(np.float32)  # (N, 3)

        # f_rest: emit zeros with correct width (3*((L^2)-1))
        num_rest = self._num_f_rest()
        if num_rest > 0:
            f_rest = np.zeros((N, num_rest), dtype=np.float32)
        else:
            f_rest = np.zeros((N, 0), dtype=np.float32)

        # opacity as *logit* (pre-sigmoid)
        op_logits = torch.logit(torch.from_numpy(opacity).float().clamp(1e-6, 1-1e-6), eps=1e-6)
        op_logits = op_logits.numpy().astype(np.float32)  # (N,)

        # scale as *log* (pre-exp)
        scale_log = torch.log(torch.from_numpy(scale).float() + 1e-8).numpy().astype(np.float32)  # (N,3)

        rot32 = rotation.astype(np.float32)  # (N,4)

        # ---- Structured array with exact schema/order ----
        dtype_list = self._attributes_like_second(num_rest)
        elements = np.empty(N, dtype=dtype_list)

        elements["x"]  = xyz[:, 0].astype(np.float32)
        elements["y"]  = xyz[:, 1].astype(np.float32)
        elements["z"]  = xyz[:, 2].astype(np.float32)
        elements["nx"] = normals[:, 0]
        elements["ny"] = normals[:, 1]
        elements["nz"] = normals[:, 2]

        elements["f_dc_0"] = f_dc[:, 0]
        elements["f_dc_1"] = f_dc[:, 1]
        elements["f_dc_2"] = f_dc[:, 2]

        # Fill f_rest_* if present
        if num_rest > 0:
            # contiguous block assign via view
            for i in range(num_rest):
                elements[f"f_rest_{i}"] = f_rest[:, i]

        elements["opacity"] = op_logits
        elements["scale_0"] = scale_log[:, 0]
        elements["scale_1"] = scale_log[:, 1]
        elements["scale_2"] = scale_log[:, 2]

        elements["rot_0"] = rot32[:, 0]
        elements["rot_1"] = rot32[:, 1]
        elements["rot_2"] = rot32[:, 2]
        elements["rot_3"] = rot32[:, 3]

        el = PlyElement.describe(elements, "vertex")

        PlyData([el]).write(buffer)
            
    def load_ply(self, path: str, compatible: bool = True):
        plydata = PlyData.read(path)

        xyz = np.stack((
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ), axis=1)
        f_dc = np.stack((
            np.asarray(plydata.elements[0]["f_dc_0"]),
            np.asarray(plydata.elements[0]["f_dc_1"]),
            np.asarray(plydata.elements[0]["f_dc_2"]),
        ), axis=1)
        opacity = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        scale = np.stack((
            np.asarray(plydata.elements[0]["scale_0"]),
            np.asarray(plydata.elements[0]["scale_1"]),
            np.asarray(plydata.elements[0]["scale_2"]),
        ), axis=1)
        rotation = np.stack((
            np.asarray(plydata.elements[0]["rot_0"]),
            np.asarray(plydata.elements[0]["rot_1"]),
            np.asarray(plydata.elements[0]["rot_2"]),
            np.asarray(plydata.elements[0]["rot_3"]),
        ), axis=1)

        self.xyz = torch.from_numpy(xyz).float()
        self.rgb = torch.from_numpy(f_dc).float()
        self.opacity = torch.from_numpy(opacity).float()
        self.scale = torch.from_numpy(scale).float()
        self.rotation = torch.from_numpy(rotation).float()

        if compatible:
            self.opacity = torch.sigmoid(self.opacity)
            self.scale = torch.exp(self.scale)
            self.rgb = 0.28209479177387814 * self.rgb + 0.5

    def _construct_list_of_attributes(self):
        l = ["x", "y", "z"]
        for i in range(self.rgb.shape[1]):
            l.append(f"f_dc_{i}")
        l.append("opacity")
        for i in range(self.scale.shape[1]):
            l.append(f"scale_{i}")
        for i in range(self.rotation.shape[1]):
            l.append(f"rot_{i}")
        return l


def render(
    pc: GaussianModel,
    height: int,
    width: int,
    C2W: Tensor,
    fxfycxcy: Tensor,
    znear: float = 0.01,
    zfar: float = 100.,
    bg_color: Union[Tensor, Tuple[float, float, float]] = (1., 1., 1.),
    scaling_modifier: float = 1.,
    render_dn: bool = False,
):
    viewpoint_camera = Camera(C2W, fxfycxcy, height, width, znear, zfar)

    if not isinstance(bg_color, Tensor):
        bg_color = torch.tensor(list(bg_color), dtype=torch.float32, device=C2W.device)
    else:
        bg_color = bg_color.to(C2W.device, dtype=torch.float32)

    pc = pc.to(dtype=torch.float32)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.h),
        image_width=int(viewpoint_camera.w),
        tanfovx=viewpoint_camera.tanfovX,
        tanfovy=viewpoint_camera.tanfovY,
        kernel_size=0.,  # cf. Mip-Splatting; not used
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        # cf. RaDe-GS
        require_depth=render_dn,
        require_coord=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    image, _, _, _, depth, _, alpha, normal = rasterizer(  # not used: radii, coord, mcoord, mdepth
        means3D=pc.xyz,
        means2D=torch.zeros_like(pc.xyz, dtype=torch.float32, device=pc.xyz.device),
        shs=None,
        colors_precomp=pc.rgb,
        opacities=pc.opacity,
        scales=pc.scale,
        rotations=pc.rotation,
        cov3D_precomp=None,
    )

    return {
        "image": image,
        "alpha": alpha,
        "depth": depth,
        "normal": normal,
    }
