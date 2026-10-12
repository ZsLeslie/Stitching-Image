import argparse
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from prepare_exif_poses import read_exif_directory, build_pose_records
from estimate_blade_distance_lrf import estimate_lrf_plane
from blade_refinement import (fit_surface, texture_constraint, blue_markers, marker_constraint)
from sam2_alignment import (mask_geometry, project_geometry, contour_constraint,
                            refine, transform, project_corrected_grid, MaskBlender)



def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def format_elapsed(seconds):
    hours, remainder = divmod(float(seconds), 3600.)
    minutes, seconds = divmod(remainder, 60.)
    return f'{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}'


def load_blade_mask(mask_dir, image_name, image_shape, analysis_size):
    """Load an external binary mask in the undistorted image coordinate system."""
    path = Path(mask_dir) / (Path(image_name).stem + ".png")
    if not path.is_file():
        raise FileNotFoundError(f"Required blade mask is missing: {path}")
    supplied = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if supplied is None:
        raise ValueError(f"Cannot read blade mask: {path}")
    if supplied.shape != tuple(image_shape):
        raise ValueError(f"Blade mask must match the undistorted image shape {tuple(image_shape)}; "
                         f"got {supplied.shape}: {path}")
    if supplied.dtype == np.uint8:
        allowed = np.zeros(256, dtype=bool)
        allowed[[0, 1, 255]] = True
        is_binary = allowed[supplied].all()
    else:
        is_binary = ((supplied == 0) | (supplied == 1) | (supplied == 255)).all()
    if not is_binary:
        raise ValueError(f"Expected a binary mask (0/255 or 0/1), not a colour overlay: {path}")
    binary = (supplied > 0).astype(np.uint8)*255
    if not binary.any():
        raise ValueError(f"Blade mask contains no foreground: {path}")
    return cv2.resize(binary, analysis_size, interpolation=cv2.INTER_NEAREST)


def read_inputs(args):
    if args.pose_npz:
        with np.load(args.pose_npz, allow_pickle=False) as archive:
            pose = dict(archive)
    else:
        metadata = read_exif_directory(args.exif_dir)
        records, _ = build_pose_records(metadata, args)
        # Keep EXIF-derived inputs in memory; no intermediate archive round trip.
        pose = {key: np.stack([r[key] for r in records]) for key in ("C", "R_wc", "K")}
        pose.update(files=np.array([r["file"] for r in records]),
                    lrf_target_distance=np.array([r.get("lrf_target_distance", float("nan"))
                                                  for r in records], dtype=np.float64),
                    lrf_status=np.array([r.get("lrf_status", "") for r in records], dtype=str))
    if len(pose["C"]) < 3:
        raise ValueError("At least three images are required")
    names = [str(x) for x in pose["files"]]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate image basenames: use a single unambiguous image sequence")
    plane = estimate_lrf_plane(pose["C"], pose["R_wc"], pose["lrf_target_distance"],
                               lrf_status=pose.get("lrf_status"))
    return pose, plane, names


def analyze(args, pose, plane, names):
    surface, surface_info = fit_surface(pose, plane, args.surface)
    def analyze_image(item):
        i, name = item
        image = cv2.imread(str(args.image_dir/name))
        if image is None:
            raise ValueError(f"Cannot read image: {name}")
        h, w = image.shape[:2]
        mask = load_blade_mask(args.mask_dir, name, (h, w), (w, h))
        geometry = mask_geometry(mask)
        curve, outline = project_geometry(geometry, surface, pose['C'][i], pose['R_wc'][i], pose['K'][i])
        small = cv2.resize(image, None, fx=args.analysis_scale, fy=args.analysis_scale, interpolation=cv2.INTER_AREA)
        smask = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        distance = cv2.distanceTransform(smask, cv2.DIST_L2, 5)
        texture_mask = (distance >= 3).astype(np.uint8)*255
        sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=.006, edgeThreshold=15)
        kp, desc = sift.detectAndCompute(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), texture_mask)
        feature = (np.array([q.pt for q in kp]).reshape(-1, 2), desc)
        return (w, h), curve, outline, feature, blue_markers(small, smask)

    workers = min(args.analysis_workers, len(names))
    available_threads = max(1, cv2.getNumberOfCPUs())
    opencv_threads = args.opencv_threads or max(1, available_threads//workers)
    print(f'Analysis: {workers} image worker(s), {opencv_threads} OpenCV thread(s)/operation', flush=True)
    previous_opencv_threads = cv2.getNumThreads()
    cv2.setNumThreads(opencv_threads)
    results = []
    try:
        for (i, name), result in bounded_parallel_map(analyze_image, enumerate(names), workers):
            results.append(result)
            print(f"Analyzing {i+1}/{len(names)}: {name}", flush=True)
    finally:
        cv2.setNumThreads(previous_opencv_threads)
    sizes, curves, outlines, features, markers = map(list, zip(*results))
    links = []
    for i in range(len(names)):
        for j in range(i+1, min(i+3, len(names))):
            edge = contour_constraint(i, j, curves[i], curves[j])
            if j == i+1 or edge['accepted']:
                links.extend([edge, texture_constraint(i, j, features, surface, pose, args.analysis_scale),
                              marker_constraint(i, j, markers, surface, pose, args.analysis_scale)])
    pivots = np.array([np.median(np.concatenate(c), axis=0) for c in curves])
    corrections, evaluations, registration = refine(len(names), links, pivots, args.max_shift_m, args.correction_model)
    errors = build_error_report(evaluations, names, surface_info, args.ppm)
    errors['registration'] = registration
    errors['mask_input'] = dict(directory=str(args.mask_dir.resolve()), boundary_resolution='original image',
                               foreground='external SAM2 binary masks', colour_segmentation=False)
    print('Registration:', ', '.join(g['model'] for g in registration['groups']), flush=True)
    return surface, corrections, sizes, outlines, errors


def build_error_report(evaluations, names, surface_info, ppm):
    """Separate contour, texture and fitted-marker residuals; missing is not zero."""
    pairs = []
    for q in evaluations:
        before = after = None
        basis = "unavailable"
        if q["accepted"] or q.get("initially_accepted"):
            if "after_holdout_median_m" in q:
                basis = "contour_holdout" if q["type"] == "edge_x" else "texture_holdout"
                before, after = q["before_holdout_median_m"], q["after_holdout_median_m"]
            elif "after_fitted_median_m" in q:
                basis = "fitted_correspondences"
                before, after = q["before_fitted_median_m"], q["after_fitted_median_m"]
        pairs.append(dict(file_i=names[q["i"]], file_j=names[q["j"]],
                          type=q["type"], accepted=q["accepted"], error_basis=basis,
                          before_median_m=before, after_median_m=after,
                          before_median_output_px=None if before is None else before*ppm,
                          after_median_output_px=None if after is None else after*ppm,
                          reason=q.get("reason", ""),
                          initially_accepted=q.get("initially_accepted", q["accepted"]),
                          width_ratio=q.get("width_ratio"), support_fraction=q.get("support_fraction"),
                          before_robust_m=q.get("before_robust_m"), after_robust_m=q.get("after_robust_m"),
                          before_left_median_m=q.get("before_left_median_m"), after_left_median_m=q.get("after_left_median_m"),
                          before_right_median_m=q.get("before_right_median_m"), after_right_median_m=q.get("after_right_median_m")))
    summary = {}
    for kind in ("edge_x", "texture_xy", "marker_xy"):
        selected = [p for p in pairs if p["type"] == kind]
        measured = [p for p in selected if p["accepted"] and p["after_median_m"] is not None]
        metrics = {}
        for key in ("before_median_m", "after_median_m", "before_median_output_px", "after_median_output_px"):
            metrics[key] = float(np.median([p[key] for p in measured])) if measured else None
        summary[kind] = dict(accepted_pairs=sum(p["accepted"] for p in selected),
                             measured_pairs=len(measured),
                             median_of_pair_medians=metrics)
    return dict(
        status="available" if any(p["after_median_m"] is not None for p in pairs) else "unavailable",
        image_count=len(names), ppm=ppm, summary=summary, pairs=pairs,
        lrf_fit=dict(model=surface_info["model"], valid_returns=surface_info["valid_returns"],
                     depth_residual_rms_m=surface_info["residual_rms_m"]),
        )


def bounded_parallel_map(function, items, workers):
    """Yield ordered results while keeping at most ``workers`` tiles in memory."""
    iterator = iter(items)
    if workers == 1:
        for item in iterator:
            yield item, function(item)
        return
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='render') as executor:
        pending = deque()
        for _ in range(workers):
            try:
                item = next(iterator)
            except StopIteration:
                break
            pending.append((item, executor.submit(function, item)))
        while pending:
            item, future = pending.popleft()
            yield item, future.result()
            try:
                following = next(iterator)
            except StopIteration:
                continue
            pending.append((following, executor.submit(function, following)))


def render(args, pose, surface, corrections, names, sizes, outlines, pivots):
    blade_uv = np.concatenate([transform(q, corrections[i], pivots[i]) for i,q in enumerate(outlines)])
    corner_uv = []
    footprint_v = []
    for i, (w, h) in enumerate(sizes):
        corners = np.array([[0,0], [w-1,0], [w-1,h-1], [0,h-1]], dtype=float)
        uv = surface.image_to_uv(corners, pose['C'][i], pose['R_wc'][i], pose['K'][i])
        if not np.isfinite(uv).all():
            raise ValueError('An image corner has no forward surface intersection')
        corner_uv.append(transform(uv, corrections[i], pivots[i]))
        # A dense source-image grid gives a conservative vertical footprint.
        # It is evaluated once, then avoids projecting unrelated images per tile.
        gx = np.unique(np.r_[np.arange(0,w,64),w-1])
        gy = np.unique(np.r_[np.arange(0,h,64),h-1])
        px,py = np.meshgrid(gx,gy)
        footprint = surface.image_to_uv(np.column_stack((px.ravel(),py.ravel())),
                                        pose['C'][i],pose['R_wc'][i],pose['K'][i])
        footprint = footprint[np.isfinite(footprint).all(axis=1)]
        if not len(footprint):raise ValueError(f'Image {names[i]} has no surface footprint')
        footprint = transform(footprint,corrections[i],pivots[i])
        footprint_v.append((float(footprint[:,1].min()-.25),float(footprint[:,1].max()+.25)))
    corners = np.concatenate(corner_uv)
    u0, u1 = blade_uv[:,0].min()-args.lateral_margin_m, blade_uv[:,0].max()+args.lateral_margin_m
    v0, v1 = corners[:,1].min(), corners[:,1].max()
    width, height = int(np.ceil((u1-u0)*args.ppm))+1, int(np.ceil((v1-v0)*args.ppm))+1
    if width*height > args.max_megapixels*1e6:
        raise ValueError(f'Output {width}x{height} exceeds the configured canvas limit')
    workers = args.render_workers
    available_threads = max(1, cv2.getNumberOfCPUs())
    opencv_threads = args.opencv_threads or max(1, available_threads//workers)
    print(f'Rendering {width}x{height}, {1000/args.ppm:.2f} nominal mm/px; '
          f'{workers} tile worker(s), {opencv_threads} OpenCV thread(s)/operation', flush=True)
    sources = []
    for name, (w,h) in zip(names, sizes):
        image = cv2.imread(str(args.image_dir/name))
        if image is None: raise ValueError(f'Cannot read image: {name}')
        mask = load_blade_mask(args.mask_dir, name, (h,w), (w,h))
        # Premultiply BEFORE resampling, so ground cannot bleed into blade edges.
        sources.append((cv2.bitwise_and(image,image,mask=mask),
                        cv2.bitwise_and(image,image,mask=255-mask), mask))
    output = np.full((height,width,3),255,np.uint8)
    valid_output = np.zeros((height,width),bool)
    u_axis = u0+np.arange(width)/args.ppm
    def render_tile(bounds):
        y0, y1 = bounds
        v_axis = v1-np.arange(y0,y1)/args.ppm
        U,V = u_axis[None,:],v_axis[:,None]
        blend = MaskBlender((y1-y0,width))
        tile_min_v,tile_max_v=float(v_axis[-1]),float(v_axis[0])
        active = (range(len(sources)) if args.no_footprint_culling else
                  [i for i,(lo,hi) in enumerate(footprint_v) if hi>=tile_min_v and lo<=tile_max_v])
        for i in active:
            fg,bg,mask=sources[i]
            x,y,front=project_corrected_grid(U,V,corrections[i],pivots[i],surface,
                                             pose['C'][i],pose['R_wc'][i],pose['K'][i])
            h,w = mask.shape
            inside = front & np.isfinite(x) & np.isfinite(y) & (x>=0) & (x<w-1) & (y>=0) & (y<h-1)
            if not inside.any():continue
            mx=np.where(inside,x,-1).astype(np.float32);my=np.where(inside,y,-1).astype(np.float32)
            f=cv2.remap(fg,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
            b=cv2.remap(bg,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
            alpha=cv2.remap(mask,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).astype(np.float32)/255
            border=np.minimum(np.minimum(x+1,w-x),np.minimum(y+1,h-y))
            weight=np.clip(border/args.feather_px,0,1)*np.exp(-args.blend_sharpness*(y/max(h-1,1)-.5)**2)
            weight=np.where(inside,weight,0).astype(np.float32)
            blend.add(f,b,alpha,weight)
        tile,tile_valid=blend.finish()
        return tile,tile_valid,len(active)

    tiles = [(y0,min(height,y0+args.tile_height)) for y0 in range(0,height,args.tile_height)]
    previous_opencv_threads = cv2.getNumThreads()
    cv2.setNumThreads(opencv_threads)
    started = time.perf_counter()
    try:
        projected_image_tiles=0
        for (y0,y1),(tile,tile_valid,active_count) in bounded_parallel_map(render_tile,tiles,workers):
            output[y0:y1],valid_output[y0:y1]=tile,tile_valid
            projected_image_tiles += active_count
            if y0%(args.tile_height*8)==0:print(f'  rows {y1}/{height}',flush=True)
    finally:
        cv2.setNumThreads(previous_opencv_threads)
    render_seconds = time.perf_counter()-started
    ys,xs=np.where(valid_output)
    if not len(xs):raise ValueError('No valid output pixels')
    output=output[ys.min():ys.max()+1,xs.min():xs.max()+1]
    filename=f'blade_refined_highres.{args.image_format}'
    options=[cv2.IMWRITE_JPEG_QUALITY,98] if args.image_format=='jpg' else [cv2.IMWRITE_PNG_COMPRESSION,3]
    if not cv2.imwrite(str(args.output_dir/filename),output,options):raise RuntimeError(f'Failed to save {filename}')
    return dict(file=filename,width=output.shape[1],height=output.shape[0],
                nominal_mm_per_pixel=1000/args.ppm,
                render_workers=workers,opencv_threads_per_operation=opencv_threads,
                render_seconds=render_seconds,
                footprint_culling=not args.no_footprint_culling,
                projected_image_tiles=projected_image_tiles,
                possible_image_tiles=len(tiles)*len(sources),
                blend='separate masked foreground/background; strongest-view silhouette')


def main():
    program_started = time.perf_counter()
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=base.parent/"02")
    parser.add_argument("--exif-dir", type=Path)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--image-format", choices=["jpg", "png"], default="jpg",
                        help="Save one final image (default: high-quality JPG)")
    parser.add_argument("--pose-npz", type=Path)
    parser.add_argument("--mask-dir", type=Path, default=None,
                        help="Required binary masks, same stem and size as undistorted images (default: masks/)")
    parser.add_argument("--surface", choices=["flat", "quadratic"], default="quadratic")
    parser.add_argument("--analysis-scale", type=float, default=.5)
    parser.add_argument("--analysis-workers", type=int, default=3,
                        help="Parallel image-analysis workers (default: 2)")
    parser.add_argument("--ppm", type=float, default=200.)
    parser.add_argument("--max-shift-m", type=float, default=3.)
    parser.add_argument("--lateral-margin-m", type=float, default=1.5)
    parser.add_argument("--feather-px", type=float, default=120.)
    parser.add_argument("--blend-sharpness", type=float, default=24.)
    parser.add_argument("--tile-height", type=int, default=512)
    parser.add_argument("--render-workers", type=int, default=4,
                        help="Parallel output-tile workers; use 1 for sequential rendering (default: 2)")
    parser.add_argument("--opencv-threads", type=int, default=0,
                        help="OpenCV threads per worker; 0 divides available CPUs among workers")
    parser.add_argument("--no-footprint-culling", action="store_true",
                        help="Project every image into every tile (slow diagnostic reference)")
    parser.add_argument("--max-megapixels", type=float, default=150.)
    parser.add_argument("--analyze-only", action="store_true")
    for name in ("fx", "fy", "cx", "cy"):
        parser.add_argument("--"+name, type=float)
    parser.add_argument("--correction-model", choices=["translation", "lateral-affine"], default="lateral-affine")
    args = parser.parse_args()
    args.image_dir = args.image_dir or args.data_dir/"images_undistorted"
    args.exif_dir = args.exif_dir or args.image_dir.parent/"images"
    args.mask_dir = args.mask_dir or args.image_dir.parent/"masks"
    args.output_dir = args.output_dir or args.image_dir.parent/"stitch_result"
    if not (0 < args.analysis_scale <= 1 and args.ppm > 0 and args.tile_height > 0
            and args.analysis_workers > 0 and args.render_workers > 0 and args.opencv_threads >= 0
            and args.feather_px > 0 and args.max_shift_m > 0 and args.lateral_margin_m >= 0):
        parser.error("Invalid scale, resolution, tile size, feather width or correction bound")
    if any(q is not None for q in (args.fx,args.fy,args.cx,args.cy)) and not all(
            q is not None for q in (args.fx,args.fy,args.cx,args.cy)):
        parser.error("Specify all four camera intrinsics together")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pose, plane, names = read_inputs(args)
    surface, corrections, sizes, outlines, errors = analyze(args, pose, plane, names)
    if not args.analyze_only:
        errors["output"] = render(args, pose, surface, corrections, names, sizes, outlines,
                                  np.array(errors["registration"]["pivots"]))
    write_json(args.output_dir/"stitch_errors.json", errors)
    if errors["status"] == "unavailable":
        print("Stitching error unavailable: no reliable image correspondences.")
    print("Saved:", args.output_dir)
    elapsed = time.perf_counter()-program_started
    print(f'Total elapsed: {elapsed:.2f} s ({format_elapsed(elapsed)})', flush=True)


if __name__ == "__main__":
    main()
