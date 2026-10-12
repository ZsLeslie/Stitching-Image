"""Mask-derived boundaries, conservative metric registration and masked blending."""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def mask_geometry(mask):
    """Keep supplied geometry; exclude only image-cut boundaries and zero-width tips.

    Inputs must describe one blade. Holes do not become external side boundaries.
    No colour, central-position, width, polynomial or morphology filter is used.
    """
    inside = mask > 0
    h, w = inside.shape
    rows = np.flatnonzero(inside.any(axis=1))
    if len(rows) < 20:
        raise ValueError('Mask has insufficient blade extent')
    left = inside.argmax(axis=1)
    right = w-1-inside[:, ::-1].argmax(axis=1)
    rows = rows[(rows > 2) & (rows < h-3)]
    rows = rows[right[rows]-left[rows] >= 2]
    # Sample the measured boundary, never interpolate across deleted image rows.
    rows = rows[::max(1, h//1400)]
    lrows = rows[left[rows] > 2]
    rrows = rows[right[rows] < w-3]
    if min(len(lrows), len(rrows)) < 20:
        raise ValueError('Mask lacks two observable side boundaries')
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline = np.concatenate([q[:, 0] for q in contours]).astype(float)
    return dict(left=np.column_stack((left[lrows], lrows)).astype(float),
                right=np.column_stack((right[rrows], rrows)).astype(float), outline=outline)


def ordered(curve):
    curve = curve[np.isfinite(curve).all(axis=1)]
    curve = curve[np.argsort(curve[:, 1])]
    _, index = np.unique(curve[:, 1], return_index=True)
    return curve[index]


def project_geometry(geometry, surface, C, R, K):
    curves = [ordered(surface.image_to_uv(geometry[side], C, R, K)) for side in ('left', 'right')]
    if min(map(len, curves)) < 20:
        raise ValueError('Insufficient forward mask boundary intersections')
    outline = surface.image_to_uv(geometry['outline'], C, R, K)
    outline = outline[np.isfinite(outline).all(axis=1)]
    if not len(outline):
        raise ValueError('Mask has no forward surface intersection')
    return curves, outline


def sample_curve(curve, v):
    """Interpolation is valid only across locally supported sample intervals."""
    y = curve[:, 1]
    k = np.clip(np.searchsorted(y, v, side='right'), 1, len(y)-1)
    gap_limit = max(.03, 4*float(np.median(np.diff(y))))
    valid = (v >= y[0]) & (v <= y[-1]) & ((y[k]-y[k-1]) <= gap_limit)
    return np.interp(v, y, curve[:, 0]), valid


def contour_constraint(i, j, ci, cj):
    record = dict(i=i, j=j, type='edge_x', accepted=False)
    lo = max(q[0, 1] for q in ci+cj)+.05
    hi = min(q[-1, 1] for q in ci+cj)-.05
    if hi-lo < .3:
        return dict(record, reason='insufficient common boundary span')
    v = np.linspace(lo, hi, 120)
    sampled = [sample_curve(c, v) for c in ci+cj]
    valid = np.logical_and.reduce([s[1] for s in sampled])
    if valid.sum() < 32:
        return dict(record, reason='insufficient continuously observed boundary samples')
    il, ir, jl, jr = [s[0][valid] for s in sampled]
    wi, wj = np.abs(ir-il), np.abs(jr-jl)
    ratio = float(np.median(np.maximum(wi, wj)/np.maximum(np.minimum(wi, wj), .02)))
    shift = .5*(jl+jr-il-ir)
    record.update(width_ratio=ratio, overlap_span_m=float(hi-lo), samples=int(valid.sum()))
    if ratio > 1.35 or abs(np.median(shift)) > 1.5 or np.median(abs(shift-np.median(shift))) > .25:
        return dict(record, reason='boundaries disagree beyond the small-correction model')
    return dict(record, accepted=True, v=v[valid], ci=ci, cj=cj,
                reason='separate left/right mask boundaries; longitudinal motion requires point matches')


def transform(uv, q, pivot, inverse=False):
    """q = tx, ty, transverse shear, transverse scale delta (not a 3-D pose)."""
    uv = np.asarray(uv)
    result = uv.copy()
    if inverse:
        result[..., 1] = uv[..., 1]-q[1]
        result[..., 0] = pivot[0]+(uv[..., 0]-pivot[0]-q[0]-q[2]*(result[..., 1]-pivot[1]))/(1+q[3])
    else:
        result[..., 0] = pivot[0]+(1+q[3])*(uv[..., 0]-pivot[0])+q[0]+q[2]*(uv[..., 1]-pivot[1])
        result[..., 1] = uv[..., 1]+q[1]
    return result


def project_corrected_grid(u, v, q, pivot, surface, C, R, K):
    """Inverse-correct and project a broadcast UV grid without NxMx3 temporaries."""
    sv = np.asarray(v, dtype=float)-q[1]
    su = (pivot[0]+(np.asarray(u, dtype=float)-pivot[0]-q[0]
          -q[2]*(sv-pivot[1]))/(1+q[3]))
    depth = np.polyval(surface.coefficients, sv)
    origin_camera = (surface.origin-C) @ R
    eu_camera, ev_camera, normal_camera = surface.eu @ R, surface.ev @ R, surface.normal @ R
    camera = [origin_camera[k]+su*eu_camera[k]+sv*ev_camera[k]+depth*normal_camera[k]
              for k in range(3)]
    denominator = K[2,0]*camera[0]+K[2,1]*camera[1]+K[2,2]*camera[2]
    with np.errstate(divide='ignore', invalid='ignore'):
        x = (K[0,0]*camera[0]+K[0,1]*camera[1]+K[0,2]*camera[2])/denominator
        y = (K[1,0]*camera[0]+K[1,1]*camera[1]+K[1,2]*camera[2])/denominator
    return x, y, camera[2] > .1


def contour_errors(link, corrections, pivots, selection):
    v = link['v'][selection]
    values, support = [], []
    for index, curves in ((link['i'], link['ci']), (link['j'], link['cj'])):
        for curve in curves:
            x, valid = sample_curve(curve, v-corrections[index, 1])
            xy = transform(np.column_stack((x, v-corrections[index, 1])), corrections[index], pivots[index])
            values.append(xy[:, 0]); support.append(valid)
    return values[0]-values[2], values[1]-values[3], np.logical_and.reduce(support)


def smooth_error(values, tolerance=.025):
    return float(np.mean(np.hypot(values, tolerance)-tolerance))


def robust_residual(values, tolerance):
    values = np.asarray(values).ravel()
    loss = np.hypot(values/tolerance, 1)-1
    return np.sign(values)*np.sqrt(2*loss/max(1, len(values)))


def components(count, links):
    groups = [{i} for i in range(count)]
    for link in links:
        a = next(g for g in groups if link['i'] in g)
        b = next(g for g in groups if link['j'] in g)
        if a is not b:
            a.update(b); groups.remove(b)
    return groups


def evaluate(link, corrections, pivots):
    if not link['accepted']:
        return {k:v for k,v in link.items() if k not in ('ci', 'cj') and not isinstance(v, np.ndarray)}
    zero = np.zeros_like(corrections)
    record = {k:v for k,v in link.items() if k not in ('ci', 'cj') and not isinstance(v, np.ndarray)}
    if link['type'] == 'edge_x':
        bl, br, base_valid = contour_errors(link, zero, pivots, slice(1, None, 2))
        al, ar, valid = contour_errors(link, corrections, pivots, slice(1, None, 2))
        valid &= base_valid
        record['support_fraction'] = float(valid.mean())
        # Point-supported longitudinal motion can legitimately shorten an overlap.
        # Require physical extent and sample support instead of 90% of the old span.
        supported_v = link['v'][1::2][valid]
        if valid.sum() < 16 or valid.mean() < .5 or np.ptp(supported_v) < .3:
            return dict(record, validation_issue='lost held-out boundary support')
        for prefix, l, r in (('before', bl, br), ('after', al, ar)):
            values = np.r_[l[valid], r[valid]]
            record[prefix+'_holdout_median_m'] = float(np.median(abs(values)))
            record[prefix+'_robust_m'] = smooth_error(values)
            record[prefix+'_left_median_m'] = float(np.median(abs(l[valid])))
            record[prefix+'_right_median_m'] = float(np.median(abs(r[valid])))
    else:
        heldout = len(link['holdout_a']) > 0
        a, b = (link['holdout_a'], link['holdout_b']) if heldout else (link['a'], link['b'])
        after = transform(a, corrections[link['i']], pivots[link['i']])-transform(b, corrections[link['j']], pivots[link['j']])
        basis = 'holdout' if heldout else 'fitted'
        record[f'before_{basis}_median_m'] = float(np.median(np.linalg.norm(a-b, axis=1)))
        record[f'after_{basis}_median_m'] = float(np.median(np.linalg.norm(after, axis=1)))
    return record


def unsafe(records, slack=.03):
    for r in records:
        if r.get('validation_issue'):
            return r['validation_issue']
        for key in ('robust', 'left_median', 'right_median', 'holdout_median', 'fitted_median'):
            before, after = r.get('before_'+key+'_m'), r.get('after_'+key+'_m')
            if before is not None and after is not None and after > before+slack:
                return f'{r["i"]}->{r["j"]}: {key} worsened by {after-before:.4f} m'
    return None


def solve(count, links, pivots, max_shift, affine=False, initial=None):
    gx = components(count, links)
    gy = components(count, [q for q in links if q['type'] != 'edge_x'])
    anchor_x = {sorted(g)[len(g)//2] for g in gx}
    anchor_y = {sorted(g)[len(g)//2] for g in gy}
    variables = [(i, axis) for i in range(count) for axis in range(4 if affine else 2)
                 if i not in (anchor_y if axis == 1 else anchor_x)]
    def unpack(x):
        result = np.zeros((count, 4))
        for value, (i, axis) in zip(x, variables): result[i, axis] = value
        return result
    if not variables:
        return np.zeros((count, 4)), None
    limits = np.array([max_shift if axis < 2 else (.01 if axis == 2 else .03) for _, axis in variables])
    def residual(x):
        q = unpack(x); values = []
        for link in links:
            if link['type'] == 'edge_x':
                l, r, valid = contour_errors(link, q, pivots, slice(None, None, 2))
                values.extend(robust_residual(np.r_[l, r], .025))
                # Fixed-length penalty: losing observed overlap never lowers the loss.
                values.extend((~valid).astype(float)*10/np.sqrt(len(valid)))
            else:
                d = transform(link['a'], q[link['i']], pivots[link['i']])-transform(link['b'], q[link['j']], pivots[link['j']])
                values.extend(robust_residual(np.linalg.norm(d, axis=1), .01))
        values.extend((q/np.array([2., 1.5, .004, .01])).ravel())
        return np.asarray(values)
    x0 = np.array([initial[i, axis] if initial is not None else 0. for i, axis in variables])
    # Each pair residual depends only on the two participating images.  Supplying
    # this structure lets SciPy perturb independent image variables together
    # instead of re-evaluating every link once per optimization variable.
    residual_rows = []
    for link in links:
        if link['type'] == 'edge_x':
            samples = len(link['v'][::2])
            residual_rows.append((3*samples, {link['i'], link['j']}))
        else:
            residual_rows.append((len(link['a']), {link['i'], link['j']}))
    row_count = sum(length for length, _ in residual_rows)+count*4
    sparsity = lil_matrix((row_count, len(variables)), dtype=np.int8)
    column_for = {item: column for column, item in enumerate(variables)}
    row = 0
    for length, images in residual_rows:
        columns = [column for column, (image, _) in enumerate(variables) if image in images]
        if columns:
            sparsity[row:row+length, columns] = 1
        row += length
    for image in range(count):
        for axis in range(4):
            column = column_for.get((image, axis))
            if column is not None:
                sparsity[row+image*4+axis, column] = 1
    fit = least_squares(residual, x0, bounds=(-limits, limits), max_nfev=250,
                        loss='linear', jac_sparsity=sparsity.tocsr(),
                        tr_options={'atol': 1e-11, 'btol': 1e-11})
    if not fit.success or not np.isfinite(fit.x).all():
        return None, 'optimizer did not converge'
    if np.any(abs(fit.x) >= .99*limits):
        return None, 'correction reached its configured bound'
    return unpack(fit.x), None


def refine(count, links, pivots, max_shift=3., model='lateral-affine'):
    trusted = [q for q in links if q['accepted']]
    final = np.zeros((count, 4)); decisions = []; rejected = set()
    for group in components(count, trusted):
        local = [q for q in trusted if q['i'] in group]
        if not local: continue
        correction, reason = solve(count, local, pivots, max_shift)
        records = [evaluate(q, correction, pivots) for q in local] if correction is not None else []
        reason = reason or unsafe(records)
        chosen = 'translation'
        if reason:
            correction = np.zeros((count, 4)); chosen = 'zero_correction'
            rejected.update((q['type'], q['i'], q['j']) for q in local)
        elif model == 'lateral-affine':
            trial, affine_reason = solve(count, local, pivots, max_shift, affine=True, initial=correction)
            if trial is not None:
                test_records = [evaluate(q, trial, pivots) for q in local]
                affine_reason = unsafe(test_records)
                # Also protect the validated translation solution, not just the LRF baseline.
                compare = [dict(after, **{k.replace('after_', 'before_', 1):v for k,v in before.items()
                                          if k.startswith('after_')}) for before,after in zip(records,test_records)]
                affine_reason = affine_reason or unsafe(compare, slack=.01)
                old = [r['after_robust_m'] for r in records if 'after_robust_m' in r]
                new = [r['after_robust_m'] for r in test_records if 'after_robust_m' in r]
                if not affine_reason and old and np.mean(new) < .95*np.mean(old):
                    correction = trial; chosen = 'lateral-affine'
                elif not affine_reason:
                    affine_reason = 'less than 5% improvement over translation on held-out contours'
            reason = affine_reason
        final[sorted(group)] = correction[sorted(group)]
        decisions.append(dict(images=sorted(group), model=chosen, reason=reason))
    records = [evaluate(q, final, pivots) for q in links]
    for r in records:
        r['initially_accepted'] = r['accepted']
        if (r['type'], r['i'], r['j']) in rejected:
            r.update(accepted=False, reason='dependent group reverted to LRF geometry')
    return final, records, dict(groups=decisions, requested_model=model,
                               corrections=final.tolist(), pivots=pivots.tolist(),
                               columns=['tx_m', 'ty_m', 'transverse_shear', 'transverse_scale_delta'])


class MaskBlender:
    """Choose one source silhouette; never average blade pixels with background."""
    def __init__(self, shape):
        self.fg = np.zeros((*shape, 3), np.float32)
        self.bg = np.zeros_like(self.fg)
        self.fw = np.zeros(shape, np.float32); self.bw = np.zeros_like(self.fw)
        self.best = np.zeros_like(self.fw); self.alpha = np.zeros_like(self.fw)

    def add(self, premultiplied_fg, premultiplied_bg, alpha, weight):
        self.fg += premultiplied_fg*weight[..., None]
        self.bg += premultiplied_bg*weight[..., None]
        self.fw += alpha*weight; self.bw += (1-alpha)*weight
        take = weight > self.best
        self.alpha[take] = alpha[take]; self.best[take] = weight[take]

    def finish(self):
        fg = self.fg/np.maximum(self.fw[..., None], 1e-12)
        bg = self.bg/np.maximum(self.bw[..., None], 1e-12)
        out = fg*self.alpha[..., None]+bg*(1-self.alpha[..., None])
        valid = self.best > 1e-10
        out[~valid] = 255
        return np.clip(np.rint(out), 0, 255).astype(np.uint8), valid
