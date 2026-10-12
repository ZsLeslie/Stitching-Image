#!/usr/bin/env python3
"""Estimate the existing single blade plane from laser ranges, without image scoring.

Ranges are in metres. By default the laser origin is approximated by the
camera centre and its direction by the camera +Z optical axis. Use the
calibration arguments when the sensor geometry is known.
"""


import numpy as np

from blade_geometry import build_blade_frame, normalize, plane_from_distance


def estimate_lrf_plane(C, R_wc, ranges, laser_direction_camera=(0, 0, 1),
                       laser_offset_camera=(0, 0, 0), max_residual_m=1.0,
                       min_samples=3, lrf_status=None):
    C = np.asarray(C, dtype=np.float64)
    R_wc = np.asarray(R_wc, dtype=np.float64)
    ranges = np.asarray(ranges, dtype=np.float64)
    if C.ndim != 2 or C.shape[1] != 3 or len(C) < 2:
        raise ValueError("At least two camera positions of shape (N, 3) are required")
    count = len(C)
    statuses = np.full(count, "", dtype="U1") if lrf_status is None else np.asarray(lrf_status, dtype=str)
    if statuses.shape != (count,):
        raise ValueError("LRFStatus must match the image count")
    # Actual M4E EXIF in this dataset uses the string 'Normal'. Missing status
    # is allowed, but an explicitly non-normal or unknown status is rejected.
    normalized_status = np.char.lower(np.char.strip(statuses))
    status_ok = (normalized_status == "") | (normalized_status == "normal")
    if R_wc.shape != (count, 3, 3) or ranges.shape != (count,):
        raise ValueError("C, R_wc and laser ranges must have matching image counts")
    if not np.isfinite(C).all() or not np.isfinite(R_wc).all():
        raise ValueError("Camera positions and rotations must be finite")
    if not np.allclose(R_wc.transpose(0, 2, 1) @ R_wc, np.eye(3), atol=1e-5):
        raise ValueError("R_wc must contain orthonormal rotations")
    if not np.allclose(np.linalg.det(R_wc), 1.0, atol=1e-5):
        raise ValueError("R_wc must contain proper rotations")
    if not np.isfinite(max_residual_m) or max_residual_m <= 0 or min_samples < 1:
        raise ValueError("max_residual_m must be positive and min_samples at least 1")
    direction = np.asarray(laser_direction_camera, dtype=np.float64)
    offset = np.asarray(laser_offset_camera, dtype=np.float64)
    if direction.shape != (3,) or offset.shape != (3,):
        raise ValueError("Laser direction and offset must each have three components")
    if not np.isfinite(direction).all() or not np.isfinite(offset).all():
        raise ValueError("Laser direction and offset must be finite")
    direction = normalize(direction)

    # Keep exactly the same orientation model as the photometric optimizer.
    singular_values = np.linalg.svd(C - C.mean(axis=0), compute_uv=False)
    if singular_values[0] < 1e-6:
        raise ValueError("Camera trajectory has no usable length direction")
    C_mean, n_world, e_u, e_v = build_blade_frame(C, R_wc)
    ray_directions = R_wc @ direction
    laser_origins = C + R_wc @ offset
    cos_incidence = ray_directions @ n_world

    # n.(P_i-C_mean), P_i = laser_origin_i + L_i * ray_direction_i.
    # The position correction is essential when cameras have different standoffs.
    candidates = np.full(count, np.nan)
    valid = np.isfinite(ranges) & (ranges > 0) & (cos_incidence > 1e-3) & status_ok
    candidates[valid] = (
        (laser_origins[valid] - C_mean) @ n_world
        + ranges[valid] * cos_incidence[valid]
    )
    if np.count_nonzero(valid) < min_samples:
        raise ValueError(f"Need {min_samples} valid forward laser returns; "
                         f"found {np.count_nonzero(valid)}")
    initial_D = float(np.median(candidates[valid]))
    inliers = valid & (np.abs(candidates - initial_D) <= max_residual_m)
    required = max(min_samples, int(np.ceil(np.count_nonzero(valid) / 2)))
    if np.count_nonzero(inliers) < required:
        raise ValueError("Laser returns do not support a consistent single plane: "
                         "check target hits, calibration, poses and residual tolerance")
    D = float(np.median(candidates[inliers]))
    O = plane_from_distance(C_mean, n_world, D)
    camera_distances = (O - C) @ n_world
    if D <= 0.1 or np.any(camera_distances <= 0.1):
        raise ValueError("Estimated plane must be in front of every camera (> 0.1 m)")
    residuals = candidates - D
    return dict(
        D=D, O=O, n_world=n_world, e_u=e_u, e_v=e_v, C_mean=C_mean,
        lrf_target_distance=ranges, lrf_plane_distance=candidates,
        lrf_status=statuses,
        lrf_valid=valid, lrf_inlier=inliers, lrf_residual_m=residuals,
        lrf_cos_incidence=cos_incidence, camera_plane_distance=camera_distances,
        laser_direction_camera=direction, laser_offset_camera=offset,
        max_residual_m=max_residual_m,
        distance_source="LRFTargetDistance",
    )
