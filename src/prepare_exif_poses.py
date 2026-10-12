#!/usr/bin/env python3
"""Read EXIF and construct camera poses in memory for the stitching pipeline."""

import json
import math
import subprocess
from pathlib import Path

import numpy as np


# ================================================================
# 工具函数
# ================================================================

def get_tag(meta, name, default=None):
    """
    exiftool 使用 -G1 后字段可能类似：
        XMP-drone-dji:GimbalYawDegree
        Composite:GPSLatitude

    这里按 ':' 后面的名字寻找。
    """
    if name in meta:
        return meta[name]

    for k, v in meta.items():
        if k.split(":")[-1] == name:
            return v

    return default


def as_float(v, default=None):
    if v is None:
        return default

    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ================================================================
# 读取 DJI EXIF / XMP
# ================================================================

def read_exif_directory(image_dir):
    image_dir = Path(image_dir)

    cmd = [
        "exiftool",
        "-j",
        "-n",
        "-G1",
        "-r",
        "-ext", "jpg",
        "-ext", "jpeg",
        "-ext", "JPG",
        "-ext", "JPEG",
        str(image_dir),
    ]


    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    metadata = json.loads(result.stdout)

    metadata.sort(
        key=lambda x: str(get_tag(x, "SourceFile", ""))
    )

    return metadata


# ================================================================
# DJI DewarpData
# ================================================================

def parse_dewarp_data(value):
    """
    常见格式：

    3713.29,3713.29,7.02,-8.72,k1,k2,p1,p2,k3

    或：

    2023-03-13;3713.29,3713.29,7.02,-8.72,...

    DJI 定义：
        fx, fy
        cx_offset, cy_offset
        k1,k2,p1,p2,k3

    注意：
        cx_offset / cy_offset 的原点是图像中心，
        不是左上角。
    """

    if value is None:
        return None

    text = str(value).strip()

    # 如果前面有 calibration date
    if ";" in text:
        text = text.split(";")[-1]

    values = []

    for item in text.split(","):
        try:
            values.append(float(item.strip()))
        except ValueError:
            pass

    if len(values) < 4:
        return None

    return values


def get_camera_intrinsics(meta,
                          fx_override=None,
                          fy_override=None,
                          cx_override=None,
                          cy_override=None):

    width = as_float(get_tag(meta, "ImageWidth"))
    height = as_float(get_tag(meta, "ImageHeight"))

    if width is None:
        width = as_float(get_tag(meta, "ExifImageWidth"))

    if height is None:
        height = as_float(get_tag(meta, "ExifImageHeight"))

    # ------------------------------------------------------------
    # 如果用户明确提供了去畸变后的 K，就优先使用
    # ------------------------------------------------------------
    if (
        fx_override is not None
        and fy_override is not None
        and cx_override is not None
        and cy_override is not None
    ):
        return np.array([
            [fx_override, 0.0, cx_override],
            [0.0, fy_override, cy_override],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    # ------------------------------------------------------------
    # 否则读取 DJI DewarpData
    # ------------------------------------------------------------
    dewarp = get_tag(meta, "DewarpData")

    values = parse_dewarp_data(dewarp)

    if values is None:
        raise RuntimeError(
            "没有找到有效 DewarpData。\n"
            "如果去畸变时已经有 K，请使用：\n"
            "--fx --fy --cx --cy"
        )

    if width is None or height is None:
        raise RuntimeError(
            "无法取得 ImageWidth / ImageHeight"
        )

    fx = values[0]
    fy = values[1]

    # DJI DewarpData 中是相对于图像中心的偏移
    dx = values[2]
    dy = values[3]

    cx = width / 2.0 + dx
    cy = height / 2.0 + dy

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    return K


# ================================================================
# GPS: WGS84 LLA -> ECEF -> ENU
# 不依赖 pyproj / pymap3d
# ================================================================

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def lla_to_ecef(lat_deg, lon_deg, alt):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)

    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    N = WGS84_A / math.sqrt(
        1.0 - WGS84_E2 * sin_lat * sin_lat
    )

    x = (N + alt) * cos_lat * cos_lon
    y = (N + alt) * cos_lat * sin_lon
    z = (
        N * (1.0 - WGS84_E2) + alt
    ) * sin_lat

    return np.array([x, y, z], dtype=np.float64)


def ecef_to_enu_matrix(lat0_deg, lon0_deg):
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)

    slat = math.sin(lat0)
    clat = math.cos(lat0)
    slon = math.sin(lon0)
    clon = math.cos(lon0)

    # ECEF -> ENU
    R = np.array([
        [-slon,       clon,       0.0],
        [-slat*clon, -slat*slon, clat],
        [ clat*clon,  clat*slon, slat],
    ], dtype=np.float64)

    return R


def lla_to_enu(lat, lon, alt,
               lat0, lon0, alt0):

    p = lla_to_ecef(lat, lon, alt)
    p0 = lla_to_ecef(lat0, lon0, alt0)

    R = ecef_to_enu_matrix(lat0, lon0)

    return R @ (p - p0)


# ================================================================
# 旋转矩阵
# ================================================================

def Rx(a):
    c = math.cos(a)
    s = math.sin(a)

    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s,  c],
    ])


def Ry(a):
    c = math.cos(a)
    s = math.sin(a)

    return np.array([
        [ c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ])


def Rz(a):
    c = math.cos(a)
    s = math.sin(a)

    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


# ================================================================
# DJI Gimbal YPR -> OpenCV camera -> ENU
# ================================================================

def dji_gimbal_to_Rwc(yaw_deg,
                      pitch_deg,
                      roll_deg):
    """
    输出：
        R_wc

    含义：
        OpenCV camera frame -> ENU world frame

    OpenCV camera:
        +X = image right
        +Y = image down
        +Z = camera forward

    DJI geographic frame:
        NED:
        +X = North
        +Y = East
        +Z = Down

    DJI yaw:
         0 deg = North
        +90 deg = East

    DJI pitch:
          0 deg = horizontal
        -90 deg = looking downward
    """

    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    # ------------------------------------------------------------
    # Payload body attitude in NED
    #
    # Body:
    #   +X forward
    #   +Y right
    #   +Z down
    #
    # Aerospace Z-Y-X yaw/pitch/roll
    # ------------------------------------------------------------
    R_ned_from_body = (
        Rz(yaw)
        @ Ry(pitch)
        @ Rx(roll)
    )

    # ------------------------------------------------------------
    # OpenCV camera -> payload/body
    #
    # CV +X right     -> body +Y
    # CV +Y down      -> body +Z
    # CV +Z forward   -> body +X
    # ------------------------------------------------------------
    R_body_from_cv = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)

    # ------------------------------------------------------------
    # NED -> ENU
    #
    # NED [North, East, Down]
    # ENU [East, North, Up]
    # ------------------------------------------------------------
    R_enu_from_ned = np.array([
        [0.0, 1.0,  0.0],
        [1.0, 0.0,  0.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float64)

    R_wc = (
        R_enu_from_ned
        @ R_ned_from_body
        @ R_body_from_cv
    )

    return R_wc


# ================================================================
# 从 metadata 建立 pose
# ================================================================

def build_pose_records(metadata, args):

    temp = []

    for meta in metadata:

        source = str(
            get_tag(meta, "SourceFile", "")
        )

        lat = as_float(
            get_tag(meta, "GPSLatitude")
        )

        lon = as_float(
            get_tag(meta, "GPSLongitude")
        )

        # 优先 DJI AbsoluteAltitude
        alt = as_float(
            get_tag(meta, "AbsoluteAltitude")
        )

        if alt is None:
            alt = as_float(
                get_tag(meta, "GPSAltitude")
            )

        yaw = as_float(
            get_tag(meta, "GimbalYawDegree")
        )

        pitch = as_float(
            get_tag(meta, "GimbalPitchDegree")
        )

        roll = as_float(
            get_tag(meta, "GimbalRollDegree")
        )

        missing = []

        if lat is None:
            missing.append("GPSLatitude")
        if lon is None:
            missing.append("GPSLongitude")
        if alt is None:
            missing.append("Altitude")
        if yaw is None:
            missing.append("GimbalYawDegree")
        if pitch is None:
            missing.append("GimbalPitchDegree")
        if roll is None:
            missing.append("GimbalRollDegree")

        if missing:
            print(
                "[SKIP]",
                Path(source).name,
                "missing:",
                ",".join(missing)
            )
            continue

        K = get_camera_intrinsics(
            meta,
            args.fx,
            args.fy,
            args.cx,
            args.cy,
        )

        temp.append({
            "file": Path(source).name,
            "source": source,
            "lat": lat,
            "lon": lon,
            "alt": alt,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "K": K,
            # Missing/invalid ranges remain NaN; the LRF plane estimator
            # validates positivity and consistency before using them.
            "lrf_target_distance": as_float(
                get_tag(meta, "LRFTargetDistance"), float("nan")
            ),
            "lrf_status": str(get_tag(meta, "LRFStatus", "")).strip(),
        })

    if not temp:
        raise RuntimeError(
            "没有找到包含完整 GPS/Gimbal 信息的照片"
        )

    # 第一张照片作为 ENU 原点
    lat0 = temp[0]["lat"]
    lon0 = temp[0]["lon"]
    alt0 = temp[0]["alt"]

    records = []

    for item in temp:

        C = lla_to_enu(
            item["lat"],
            item["lon"],
            item["alt"],
            lat0,
            lon0,
            alt0,
        )

        R_wc = dji_gimbal_to_Rwc(
            item["yaw"],
            item["pitch"],
            item["roll"],
        )

        # OpenCV camera +Z 就是光轴
        forward = R_wc[:, 2]

        # image right
        right = R_wc[:, 0]

        # image down
        down = R_wc[:, 1]

        item["C"] = C
        item["R_wc"] = R_wc
        item["forward"] = forward
        item["right"] = right
        item["down"] = down

        records.append(item)

    return records, (lat0, lon0, alt0)
