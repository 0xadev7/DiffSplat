import math
import numpy as np
from io import BytesIO

try:
    from plyfile import PlyData, PlyElement
except Exception:
    PlyData = None  # we'll check at runtime

def _rot_x(t):
    c,s = math.cos(t), math.sin(t)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float64)

def _rot_y(t):
    c,s = math.cos(t), math.sin(t)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float64)

def _rot_z(t):
    c,s = math.cos(t), math.sin(t)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)

def _up_rotation(from_up: str) -> np.ndarray:
    """R such that R * v maps input 'up' to +Y. Supported: 'y' (no-op), 'z' (rotate -90° about +X)."""
    if from_up.lower() == "y":
        return np.eye(3)
    if from_up.lower() == "z":
        return _rot_x(-math.pi/2)
    raise ValueError("from_up must be 'y' or 'z'")

def _euler_extra(rx_deg=0.0, ry_deg=0.0, rz_deg=0.0):
    Rx = _rot_x(math.radians(rx_deg))
    Ry = _rot_y(math.radians(ry_deg))
    Rz = _rot_z(math.radians(rz_deg))
    return Rz @ Ry @ Rx

def hygiene_ply_bytes(
    ply_bytes: bytes,
    *,
    from_up: str = "z",
    R: float = 1.4,
    fov_deg: float = 49.0,
    k: float = 0.70,
    extra_rx: float = 0.0,
    extra_ry: float = 0.0,
    extra_rz: float = 0.0,
) -> bytes:
    """
    - Center -> origin
    - Orient -> +Y up   (assuming source up is 'z' or 'y')
    - Uniform scale so a camera at distance R with vertical FOV fov_deg
      sees ~k frame occupancy (radius coverage).

    Returns new PLY bytes.
    """
    if PlyData is None:
        raise RuntimeError("Please `pip install plyfile` to enable PLY hygiene.")

    bio = BytesIO(ply_bytes)
    ply = PlyData.read(bio)

    v = ply["vertex"].data
    if not {"x","y","z"}.issubset({n.lower() for n in v.dtype.names}):
        raise ValueError("PLY must contain vertex properties x, y, z")

    X = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)

    # 1) translate centroid -> origin
    X -= X.mean(axis=0, keepdims=True)

    # 2) rotate to +Y up (+ optional extra tweak)
    R_up = _up_rotation(from_up)
    R_ex = _euler_extra(extra_rx, extra_ry, extra_rz)
    X = (R_ex @ (R_up @ X.T)).T

    # 3) uniform scale to target occupancy
    r = np.linalg.norm(X, axis=1).max()
    if r <= 0:
        raise ValueError("Degenerate geometry (zero radius).")
    target = k * R * math.tan(math.radians(fov_deg) / 2.0)
    s = target / r
    X *= s

    # write back (preserve all other properties)
    v_new = v.copy()
    v_new["x"] = X[:,0]
    v_new["y"] = X[:,1]
    v_new["z"] = X[:,2]

    out = BytesIO()
    PlyData([PlyElement.describe(v_new, "vertex"), *[e for e in ply.elements if e.name!="vertex"]],
            text=ply.text).write(out)
    return out.getvalue()
