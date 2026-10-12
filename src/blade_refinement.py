"""Conservative blade-only registration on a shared LRF-constrained surface.

The extra surface model is a depth profile, not a recovered 3-D blade mesh.
Contours constrain lateral alignment only. Longitudinal corrections require
verified, non-duplicate texture correspondences.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares


@dataclass
class Surface:
    origin: np.ndarray
    eu: np.ndarray
    ev: np.ndarray
    normal: np.ndarray
    coefficients: np.ndarray  # depth = a*v*v + b*v + c, metres

    def depth(self, v):
        return np.polyval(self.coefficients, v)

    def world(self, uv):
        uv = np.asarray(uv, dtype=float)
        return (self.origin + uv[..., 0, None] * self.eu
                + uv[..., 1, None] * self.ev
                + self.depth(uv[..., 1])[..., None] * self.normal)

    def image_to_uv(self, xy, C, R, K):
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        rays = np.column_stack((xy, np.ones(len(xy)))) @ np.linalg.inv(K).T @ R.T
        c = C - self.origin
        cv, cn = c @ self.ev, c @ self.normal
        rv, rn = rays @ self.ev, rays @ self.normal
        a, b, d = self.coefficients
        A = a * rv * rv
        B = (2 * a * cv + b) * rv - rn
        Q = a * cv * cv + b * cv + d - cn
        t = np.full(len(xy), np.nan)
        linear = np.abs(A) < 1e-12
        good = linear & (np.abs(B) > 1e-12)
        t[good] = -Q / B[good]
        discr = B * B - 4 * A * Q
        good = (~linear) & (discr >= 0)
        # Stable quadratic roots; choose the nearest positive intersection.
        q = -.5 * (B[good] + np.copysign(np.sqrt(discr[good]), B[good]))
        with np.errstate(divide="ignore", invalid="ignore"):
            r1, r2 = q / A[good], Q / q
        r1[(r1 <= 0) | ~np.isfinite(r1)] = np.inf
        r2[(r2 <= 0) | ~np.isfinite(r2)] = np.inf
        t[good] = np.minimum(r1, r2)
        t[(t <= 0) | ~np.isfinite(t)] = np.nan
        points = c + t[:, None] * rays
        return np.column_stack((points @ self.eu, points @ self.ev))

    def project(self, uv, C, R, K):
        camera = (self.world(uv) - C) @ R
        pixel = camera @ K.T
        with np.errstate(divide="ignore", invalid="ignore"):
            xy = pixel[..., :2] / pixel[..., 2, None]
        return xy, camera[..., 2] > .1


def fit_surface(pose, plane, mode="quadratic"):
    C, R = pose["C"], pose["R_wc"]
    origin, eu, ev, n = (plane[k] for k in ("C_mean", "e_u", "e_v", "n_world"))
    ranges = pose["lrf_target_distance"]
    statuses = np.char.lower(np.char.strip(pose.get("lrf_status", np.full(len(C), ""))))
    valid = np.isfinite(ranges) & (ranges > 0) & ((statuses == "normal") | (statuses == ""))
    P = C[valid] + ranges[valid, None] * R[valid, :, 2]
    v, depths = (P - origin) @ ev, (P - origin) @ n
    coeff = np.array([0., 0., float(plane["D"])])
    details = {"requested_model": mode, "model": "flat", "valid_returns": int(valid.sum())}
    if mode == "quadratic":
        if len(v) < 5 or np.ptp(v) < 5:
            details["fallback"] = "Insufficient span or valid LRF returns for a depth profile"
        else:
            span = max(np.max(np.abs(v)), 1.)
            X = np.column_stack(((v / span) ** 2, v / span, np.ones(len(v))))
            initial = np.linalg.lstsq(X, depths, rcond=None)[0]
            def residual(q):
                # Normalized by a modelling tolerance, not claimed sensor precision.
                return np.r_[(X @ q - depths) / .3, q[:2] / 5.]
            result = least_squares(residual, initial, loss="soft_l1", f_scale=1.)
            candidate = result.x / np.array([span ** 2, span, 1.])
            slopes = 2 * candidate[0] * v + candidate[1]
            if result.success and np.all(np.isfinite(candidate)) and np.max(np.abs(slopes)) < .35:
                coeff = candidate
                details["model"] = "quadratic"
            else:
                details["fallback"] = "Depth profile failed convergence or slope guard"
    residuals = depths - np.polyval(coeff, v)
    details.update(coefficients=coeff.tolist(), span_positions_m=v.tolist(),
                   target_depths_m=depths.tolist(), residuals_m=residuals.tolist(),
                   residual_rms_m=float(np.sqrt(np.mean(residuals ** 2))))
    return Surface(origin, eu, ev, n, coeff), details


def texture_constraint(i, j, features, surface, pose, scale):
    ki, di = features[i]; kj, dj = features[j]
    record = {"i": i, "j": j, "type": "texture_xy", "accepted": False}
    if di is None or dj is None or len(di) < 2 or len(dj) < 2:
        return dict(record, reason="too few texture descriptors")
    matcher = cv2.BFMatcher()
    def ratio(a, b):
        return [q[0] for q in matcher.knnMatch(a, b, k=2)
                if len(q) == 2 and q[0].distance < .75 * q[1].distance]
    backward = {m.queryIdx: m.trainIdx for m in ratio(dj, di)}
    matches = [m for m in ratio(di, dj) if backward.get(m.trainIdx) == m.queryIdx]
    a = np.array([ki[m.queryIdx] for m in matches]).reshape(-1, 2)
    b = np.array([kj[m.trainIdx] for m in matches]).reshape(-1, 2)
    if len(a) < 6:
        return dict(record, reason="fewer than six mutual texture matches", matches=len(a))
    # SIFT may return multiple orientations at one location; these are not
    # independent measurements and must not satisfy the minimum sample count.
    _, unique = np.unique(np.round(a / 5).astype(int), axis=0, return_index=True)
    a, b = a[unique], b[unique]
    K = pose["K"].copy(); K[:, :2] *= scale
    ua = surface.image_to_uv(a, pose["C"][i], pose["R_wc"][i], K[i])
    ub = surface.image_to_uv(b, pose["C"][j], pose["R_wc"][j], K[j])
    good = np.isfinite(ua).all(axis=1) & np.isfinite(ub).all(axis=1)
    good &= (np.abs(ub-ua) < [1.5, 2.5]).all(axis=1)
    ua, ub = ua[good], ub[good]
    if len(ua) < 6:
        return dict(record, reason="too few unique matches near geometric prediction", matches=len(ua))
    offsets = ub - ua
    consensus = np.linalg.norm(offsets[:, None] - offsets[None], axis=2) < .025
    inside = consensus[np.argmax(consensus.sum(axis=1))]
    ua, ub = ua[inside], ub[inside]
    if len(ua) < 6 or np.linalg.norm(np.ptp(ua, axis=0)) < .25:
        return dict(record, reason="texture consensus is too small or spatially concentrated", matches=len(ua))
    # Deterministic held-out matches are not passed to the global optimizer.
    order = np.argsort(ua[:, 1]); ua, ub = ua[order], ub[order]
    holdout = np.arange(len(ua)) % 3 == 0
    return dict(record, accepted=True, matches=len(ua), a=ua[~holdout], b=ub[~holdout],
                holdout_a=ua[holdout], holdout_b=ub[holdout], reason="verified translation consensus")


def blue_markers(image, mask):
    """Extract distinctive blue inspection marks; preserve arrow direction."""
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue = ((hsv[...,0] >= 90) & (hsv[...,0] <= 125) & (hsv[...,1] >= 100)
            & (hsv[...,2] >= 60)).astype(np.uint8)*255
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    blue &= cv2.dilate(mask, np.ones((7,7), np.uint8))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(blue)
    markers = []
    factor = w / 2640.
    for k in range(1, count):
        x, y, bw, bh, area = stats[k]
        if (area < max(8,30*factor**2) or area > 1500*factor**2 or min(bw,bh)<3
                or max(bw,bh)>100*factor or min(x,y)<4 or x+bw>=w-4 or y+bh>=h-4):
            continue
        patch = (labels[y:y+bh,x:x+bw] == k).astype(np.float32)
        side = max(bw,bh)+8
        square = np.zeros((side,side), np.float32)
        dy, dx = (side-bh)//2, (side-bw)//2
        square[dy:dy+bh,dx:dx+bw] = patch
        square = cv2.resize(square, (64,64))
        square = cv2.GaussianBlur(square, (3,3), .8)
        square /= max(float(np.linalg.norm(square)), 1e-8)
        markers.append({"xy": centers[k], "shape": square, "area": int(area)})
    return markers


def marker_constraint(i, j, markers, surface, pose, scale):
    record = {"i":i, "j":j, "type":"marker_xy", "accepted":False}
    mi, mj = markers[i], markers[j]
    if not mi or not mj:
        return dict(record, reason="no visible blue markers in both images")
    K = pose["K"].copy(); K[:, :2] *= scale
    ai = np.array([m["xy"] for m in mi]); bj = np.array([m["xy"] for m in mj])
    ua = surface.image_to_uv(ai, pose["C"][i], pose["R_wc"][i], K[i])
    ub = surface.image_to_uv(bj, pose["C"][j], pose["R_wc"][j], K[j])
    scores = np.full((len(mi),len(mj)), -1.)
    for a, ma in enumerate(mi):
        for b, mb in enumerate(mj):
            delta = ub[b]-ua[a]
            if np.isfinite(delta).all() and (np.abs(delta)<[1.5,2.5]).all():
                scores[a,b] = float(np.sum(ma["shape"]*mb["shape"]))
    matches = []
    for a in range(len(mi)):
        b = int(np.argmax(scores[a])); score = scores[a,b]
        if score < .85 or int(np.argmax(scores[:,b])) != a:
            continue
        row_alt = np.delete(scores[a], b); col_alt = np.delete(scores[:,b], a)
        second = max(float(row_alt.max()) if len(row_alt) else -1.,
                     float(col_alt.max()) if len(col_alt) else -1.)
        if score-second >= .1:
            matches.append((a,b,score))
    if not matches:
        return dict(record, reason="marker shapes ambiguous or not bidirectionally unique")
    a = np.array([ua[q[0]] for q in matches]); b = np.array([ub[q[1]] for q in matches])
    offsets = b-a
    coherent = np.linalg.norm(offsets-np.median(offsets,axis=0),axis=1)<.04
    if len(matches)==1 and matches[0][2]<.92:
        return dict(record, reason="single marker shape confidence below 0.92")
    if len(matches)>1 and not coherent.all():
        return dict(record, reason="multiple marker displacements disagree")
    # Markers support translation only. Single-marker registration has no
    # independent geometric holdout; report that limitation explicitly.
    return dict(record, accepted=True, matches=len(matches), a=a, b=b,
                holdout_a=np.empty((0,2)), holdout_b=np.empty((0,2)),
                source_xy=[mi[q[0]]["xy"].tolist() for q in matches],
                target_xy=[mj[q[1]]["xy"].tolist() for q in matches],
                shape_scores=[q[2] for q in matches], independent_holdout=False,
                reason="unique directed blue-marker shapes and consistent geometric displacement")


