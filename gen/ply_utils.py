import math
import numpy as np
from io import BytesIO
from plyfile import PlyData, PlyElement


def _rot_x(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _project_points(X, fxfy, cx=0.5, cy=0.5):
    # NDC-style intrinsics used in your code: u = fx*x/z + cx, v = fy*y/z + cy
    # Camera is at (0,0,R) looking toward origin (-Z), so we transform to camera frame:
    # In your orbit rig, the canonical view points the camera along -Z with +Y up.
    # Here, we just treat world==camera with camera at (0,0,R) looking at origin.
    # Move camera to origin by translating Z: Zc = R - Z (since camera at +Z)
    Xc = X.copy()
    Xc[:, 2] = 1.4 - Xc[:, 2]  # R = 1.4
    # Guard near-plane
    eps = 1e-6
    Z = np.clip(Xc[:, 2], eps, None)
    u = fxfy * (Xc[:, 0] / Z) + cx
    v = fxfy * (Xc[:, 1] / Z) + cy
    return np.stack([u, v], axis=1)


def _detect_up_axis(X):
    # Simple heuristic: which axis already aligns “up”? pick axis whose positive
    # direction correlates with tallest extent vs. centroid AND has the smallest skew.
    extents = X.max(0) - X.min(0)
    # If Y extent is already >= Z extent by a margin, assume Y-up
    if extents[1] >= extents[2] * 0.9:
        return "y"
    # Otherwise consider it Z-up
    return "z"


def hygiene_ply_bytes(
    ply_bytes: bytes,
    *,
    from_up: str | None = None,  # None -> auto-detect
    fxfy: float = 1.0,  # use your self.opt.fxfy
    target_occ: float = 0.70,  # target screen bbox height (or width) fraction
    occ_mode: str = "maxdim",  # "height", "width", or "maxdim"
    scale_clamp: tuple[float, float] = (0.5, 2.0),  # prevent wild rescaling
    keep_rotation_if_y_up: bool = True,
) -> bytes:
    bio = BytesIO(ply_bytes)
    ply = PlyData.read(bio)
    v = ply["vertex"].data
    names_lower = [n.lower() for n in v.dtype.names]
    assert set(["x", "y", "z"]).issubset(set(names_lower)), "PLY must have x,y,z"

    # Map original order
    x_i, y_i, z_i = [names_lower.index(k) for k in ("x", "y", "z")]
    X = np.stack(
        [v[v.dtype.names[x_i]], v[v.dtype.names[y_i]], v[v.dtype.names[z_i]]], axis=1
    ).astype(np.float64)

    # 1) Center -> origin
    X -= X.mean(axis=0, keepdims=True)

    # 2) Up-axis (auto unless specified)
    if from_up is None:
        from_up = _detect_up_axis(X)

    # Only rotate if needed
    if from_up.lower() == "z":
        R_up = _rot_x(-math.pi / 2)  # Z-up -> Y-up
        X = (R_up @ X.T).T
    elif from_up.lower() == "y":
        pass
    else:
        raise ValueError("from_up must be None/'y'/'z'")

    # 3) Screen-space occupancy scale (project with your intrinsics)
    UV = _project_points(X, fxfy=fxfy, cx=0.5, cy=0.5)
    u_min, v_min = UV.min(0)
    u_max, v_max = UV.max(0)
    w = float(u_max - u_min)
    h = float(v_max - v_min)
    cur_occ = h if occ_mode == "height" else w if occ_mode == "width" else max(w, h)

    # If current occupancy is 0 (degenerate), bail
    if cur_occ <= 1e-6:
        # write back as-is
        out = BytesIO()
        PlyData(ply.elements, text=ply.text).write(out)
        return out.getvalue()

    scale_target = target_occ / cur_occ
    scale_target = float(np.clip(scale_target, scale_clamp[0], scale_clamp[1]))
    X *= scale_target

    # 4) Write back WITHOUT changing vertex dtype or property order
    v_new = v.copy()
    v_new[v.dtype.names[x_i]] = X[:, 0]
    v_new[v.dtype.names[y_i]] = X[:, 1]
    v_new[v.dtype.names[z_i]] = X[:, 2]

    out = BytesIO()
    # Recreate vertex with identical dtype & property order
    vertex_el = PlyElement.describe(v_new, "vertex")
    other = [e for e in ply.elements if e.name != "vertex"]
    PlyData([vertex_el, *other], text=ply.text).write(out)
    return out.getvalue()
