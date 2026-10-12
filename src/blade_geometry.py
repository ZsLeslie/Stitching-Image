"""Shared plane geometry for image-based and laser-based distance estimation."""

import numpy as np


def normalize(v):
    v = np.asarray(v, dtype=np.float64)

    n = np.linalg.norm(v)

    if n < 1e-12:
        raise ValueError("Cannot normalize zero vector")

    return v / n


def build_blade_frame(
    C,
    R_wc
):
    """
    返回：

        C_mean
        n_world : 叶片法向，方向为 camera -> blade
        e_u     : 叶片宽度方向
        e_v     : 叶片长度方向

    e_u x e_v = n_world
    """

    # ------------------------------------------------------------
    # 1. 平均光轴
    # ------------------------------------------------------------

    forwards = R_wc[:, :, 2].copy()

    ref = forwards[0]

    # 防止某些旋转表示出现反向
    for i in range(len(forwards)):
        if np.dot(
            forwards[i],
            ref
        ) < 0:
            forwards[i] *= -1.0

    f_mean = normalize(
        np.mean(
            forwards,
            axis=0
        )
    )

    # ------------------------------------------------------------
    # 2. 相机轨迹 PCA
    # ------------------------------------------------------------

    C_mean = np.mean(
        C,
        axis=0
    )

    X = (
        C
        -
        C_mean
    )

    _, _, Vt = np.linalg.svd(
        X,
        full_matrices=False
    )

    e_v = normalize(
        Vt[0]
    )

    # 保证方向与采集方向一致
    if np.dot(
        e_v,
        C[-1] - C[0]
    ) < 0:
        e_v = -e_v

    # ------------------------------------------------------------
    # 3. 叶片法向应与长度方向正交
    # ------------------------------------------------------------

    n_world = (
        f_mean
        -
        np.dot(
            f_mean,
            e_v
        ) * e_v
    )

    n_world = normalize(
        n_world
    )

    # 保证仍然指向平均光轴方向
    if np.dot(
        n_world,
        f_mean
    ) < 0:
        n_world = -n_world

    # ------------------------------------------------------------
    # 4. 叶片宽度方向
    # ------------------------------------------------------------

    e_u = normalize(
        np.cross(
            e_v,
            n_world
        )
    )

    # 此时:
    #
    # e_u x e_v = n_world

    return (
        C_mean,
        n_world,
        e_u,
        e_v
    )


def plane_from_distance(
    C_mean,
    n_world,
    D
):
    """
    O 是叶片平面上的一点
    """

    O = (
        C_mean
        +
        D * n_world
    )

    return O
