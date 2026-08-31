"""Generate metric depth maps from LiDAR point clouds and KITTI-format calibration.

Usage::

    python -m sava_sensormae_toolbox.utils.generate_depthmap \
        --lidar  data/samples/KITTI/Velodyne/007454.bin \
        --calib  data/samples/KITTI/Calib/007454.txt \
        --image  data/samples/KITTI/RGB/007454.png \
        --out    data/samples/KITTI/Depth/007454.npy

The output is a float32 ``.npy`` array of shape ``[H, W]`` containing metric
depth in metres, suitable as input to the RGB + Depth inference pipelines.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import Delaunay


# ── Calibration ──────────────────────────────────────────────────────────

def parse_kitti_calib(calib_path: str) -> dict:
    """Parse a KITTI calibration ``.txt`` file.

    Returns dict with ``P2``, ``Tr_velo_to_cam``, and ``R0_rect_4x4``.
    """
    data = {}
    with open(calib_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, vals = line.split(":", 1)
            data[key.strip()] = np.array(vals.strip().split(), dtype=np.float32)

    P2 = data["P2"].reshape(3, 4)
    R0 = data["R0_rect"].reshape(3, 3)
    Tr = data["Tr_velo_to_cam"].reshape(3, 4)
    Tr_4x4 = np.eye(4, dtype=np.float32)
    Tr_4x4[:3, :] = Tr
    R0_4x4 = np.eye(4, dtype=np.float32)
    R0_4x4[:3, :3] = R0

    return {"P2": P2, "Tr_velo_to_cam": Tr_4x4, "R0_rect_4x4": R0_4x4}


# ── Depth generation ─────────────────────────────────────────────────────

def generate_metric_depth(
    lidar_path: str,
    calib: dict,
    image_size: tuple,
    lidar_channels: int = 4,
) -> np.ndarray:
    """Project LiDAR to image, Delaunay-interpolate, return ``[H, W]`` metric depth.

    Args:
        lidar_path: Path to ``.bin`` point cloud (float32, ``[N, lidar_channels]``).
        calib: Parsed calibration dict (from :func:`parse_kitti_calib`).
        image_size: ``(width, height)`` of the target image.
        lidar_channels: Columns per point (typically 4: x, y, z, reflectance).
    """
    img_w, img_h = image_size
    pcd = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, lidar_channels)

    P2 = calib["P2"]
    Tr = calib["Tr_velo_to_cam"]
    R0 = calib["R0_rect_4x4"]

    # LiDAR → camera → image
    pcd_hom = np.hstack((pcd[:, :3], np.ones((len(pcd), 1), dtype=np.float32)))
    pcd_cam = (R0 @ Tr @ pcd_hom.T).T
    pcd_proj = (P2 @ pcd_cam.T).T
    depth_vals = pcd_cam[:, 2]

    valid = depth_vals > 0.1
    pcd_proj, depth_vals = pcd_proj[valid], depth_vals[valid]
    u = pcd_proj[:, 0] / pcd_proj[:, 2]
    v = pcd_proj[:, 1] / pcd_proj[:, 2]

    in_img = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    u, v, depth_vals = u[in_img], v[in_img], depth_vals[in_img]
    if len(u) < 3:
        return np.zeros((img_h, img_w), dtype=np.float32)

    # Delaunay triangulation + barycentric interpolation
    uv = np.stack([u, v], axis=1).astype(np.float64)
    tri = Delaunay(uv)

    # Reject overly large / depth-discontinuous triangles
    max_edge_px = max(img_w, img_h) * 0.15
    max_depth_ratio = 3.0
    verts = uv[tri.simplices]
    d = depth_vals[tri.simplices]
    edges = np.stack([
        np.linalg.norm(verts[:, 1] - verts[:, 0], axis=1),
        np.linalg.norm(verts[:, 2] - verts[:, 1], axis=1),
        np.linalg.norm(verts[:, 0] - verts[:, 2], axis=1),
    ], axis=1)
    simplex_valid = (
        (edges.max(axis=1) < max_edge_px)
        & ((d.max(axis=1) / d.min(axis=1).clip(1e-3)) < max_depth_ratio)
    )

    # Query every pixel
    gx, gy = np.meshgrid(np.arange(img_w), np.arange(img_h))
    query = np.column_stack((gx.ravel(), gy.ravel()))
    simplex_idx = tri.find_simplex(query)

    depth_flat = np.zeros(len(query), dtype=np.float32)
    valid_mask = (simplex_idx >= 0) & simplex_valid[np.clip(simplex_idx, 0, None)]

    if valid_mask.sum() > 0:
        triangles = tri.simplices[simplex_idx[valid_mask]]
        tri_pts = uv[triangles]
        p = query[valid_mask].astype(np.float64)
        p0, p1, p2 = tri_pts[:, 0], tri_pts[:, 1], tri_pts[:, 2]
        v0, v1, v2 = p1 - p0, p2 - p0, p - p0
        d00 = (v0 * v0).sum(1)
        d01 = (v0 * v1).sum(1)
        d11 = (v1 * v1).sum(1)
        d20 = (v2 * v0).sum(1)
        d21 = (v2 * v1).sum(1)
        denom = np.where(
            np.abs(d00 * d11 - d01 * d01) < 1e-10,
            1e-10,
            d00 * d11 - d01 * d01,
        )
        bv = (d11 * d20 - d01 * d21) / denom
        bw = (d00 * d21 - d01 * d20) / denom
        bu = 1.0 - bv - bw
        depth_flat[valid_mask] = (
            bu * depth_vals[triangles[:, 0]]
            + bv * depth_vals[triangles[:, 1]]
            + bw * depth_vals[triangles[:, 2]]
        ).astype(np.float32)

    return depth_flat.reshape(img_h, img_w)

# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate metric depth .npy from LiDAR + calibration",
    )
    parser.add_argument("--lidar", required=True, help="Path to .bin LiDAR file")
    parser.add_argument("--calib", required=True, help="Path to KITTI .txt calib file")
    parser.add_argument("--image", required=True, help="Path to RGB image (for size)")
    parser.add_argument("--out", required=True, help="Output .npy path")
    parser.add_argument("--channels", type=int, default=4,
                        help="LiDAR channels per point (default: 4)")
    args = parser.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {args.image}")
    h, w = img.shape[:2]

    calib = parse_kitti_calib(args.calib)
    depth = generate_metric_depth(args.lidar, calib, (w, h), args.channels)
    print(f"Delaunay depth: shape={depth.shape}, "
          f"non-zero={np.count_nonzero(depth)}/{depth.size}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, depth)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
