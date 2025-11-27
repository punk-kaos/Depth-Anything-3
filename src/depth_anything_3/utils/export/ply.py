# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import shutil
import trimesh
import numpy as np

from depth_anything_3.specs import Prediction
from depth_anything_3.utils.logger import logger

from .depth_vis import export_to_depth_vis
from .glb import (
    _depths_to_world_mesh,
    _generate_aligned_point_cloud,
    _prepare_conf_and_threshold,
)


def export_to_ply(
    prediction: Prediction,
    export_dir: str,
    num_max_points: int = 1_000_000,
    conf_thresh: float = 1.05,
    filter_black_bg: bool = False,
    filter_white_bg: bool = False,
    conf_thresh_percentile: float = 40.0,
    ensure_thresh_percentile: float = 90.0,
    sky_depth_def: float = 98.0,
    export_depth_vis: bool = True,
    as_points: bool = False,
    as_mesh: bool = False,
) -> str:
    """Export the reconstructed colored point cloud to a PLY file."""
    logger.info(
        f"PLY export mode: {'mesh' if as_mesh else 'points' if as_points else 'vertices'}, "
        f"num_max_points={num_max_points}"
    )
    assert (
        prediction.processed_images is not None
    ), "Export to PLY: prediction.processed_images is required but not available"
    assert prediction.depth is not None, "Export to PLY: prediction.depth is required but not available"
    assert (
        prediction.intrinsics is not None
    ), "Export to PLY: prediction.intrinsics is required but not available"
    assert (
        prediction.extrinsics is not None
    ), "Export to PLY: prediction.extrinsics is required but not available"
    assert prediction.conf is not None, "Export to PLY: prediction.conf is required but not available"

    logger.info(f"Exporting to PLY with num_max_points: {num_max_points}")
    conf_thr = _prepare_conf_and_threshold(
        prediction=prediction,
        conf_thresh=conf_thresh,
        filter_black_bg=filter_black_bg,
        filter_white_bg=filter_white_bg,
        conf_thresh_percentile=conf_thresh_percentile,
        ensure_thresh_percentile=ensure_thresh_percentile,
        sky_depth_def=sky_depth_def,
    )

    points, colors, A = _generate_aligned_point_cloud(
        prediction=prediction,
        num_max_points=num_max_points,
        conf_thresh=conf_thresh,
        filter_black_bg=filter_black_bg,
        filter_white_bg=filter_white_bg,
        conf_thresh_percentile=conf_thresh_percentile,
        ensure_thresh_percentile=ensure_thresh_percentile,
        sky_depth_def=sky_depth_def,
        precomputed_conf_thr=conf_thr,
        skip_preprocessing=True,
    )

    os.makedirs(export_dir, exist_ok=True)
    out_path = os.path.join(export_dir, "scene.ply")

    if points.shape[0] == 0:
        logger.warning("No valid points available; exporting empty PLY.")
        # Still write an empty point cloud so downstream tooling does not break.
        points = points.reshape(-1, 3)

    # Quick viewer alignment hack: rotate about X by +90° and scale by 10x
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    points = (points @ Rx.T) * 10.0

    if as_mesh:
        # Build mesh in aligned frame consistent with GLB (uses A from point cloud path)
        verts, cols, faces = _depths_to_world_mesh(
            prediction, conf_thr, A=A, use_conf=True
        )
        if verts.shape[0] > 0:
            verts = (verts @ Rx.T) * 10.0
        logger.info(
            f"[PLY mesh] verts: {verts.shape[0]} faces: {faces.shape[0] if faces is not None else 0}"
        )
        if faces is None or faces.shape[0] == 0:
            logger.warning("PLY mesh export requested, but no faces were generated.")
        _write_ply_mesh(out_path, verts, cols, faces)
    elif as_points:
        _write_ply_point_cloud(out_path, points, colors, element_name="point")
    else:
        # Legacy vertex-only mesh export (kept for compatibility with older viewers)
        pc = trimesh.points.PointCloud(vertices=points, colors=colors)
        pc.export(out_path)

    if export_depth_vis:
        export_to_depth_vis(prediction, export_dir)
        preview_src = os.path.join(export_dir, "depth_vis", "0000.jpg")
        preview_dst = os.path.join(export_dir, "scene.jpg")
        if os.path.exists(preview_src):
            shutil.copy2(preview_src, preview_dst)

    return out_path


__all__ = ["export_to_ply"]


def _write_ply_point_cloud(path: str, points, colors, element_name: str = "point") -> None:
    """Write a PLY containing only point data (no faces).

    Using ``element point`` makes it explicit to readers that this is a point cloud
    rather than a mesh with disconnected vertices.
    """
    num_points = points.shape[0]
    has_color = colors is not None and colors.shape[0] == num_points and colors.shape[1] == 3

    header_lines = [
        "ply",
        "format ascii 1.0",
        f"element {element_name} {num_points}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_color:
        header_lines.extend(
            [
                "property uchar red",
                "property uchar green",
                "property uchar blue",
            ]
        )
    header_lines.append("end_header")

    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(header_lines))
        f.write("\n")
        if num_points == 0:
            return

        pts = points.astype(float)
        cols = colors.astype(int) if has_color else None
        for i in range(num_points):
            x, y, z = pts[i]
            line = f"{x} {y} {z}"
            if cols is not None:
                r, g, b = cols[i]
                line += f" {r} {g} {b}"
            f.write(line)
            f.write("\n")


def _write_ply_mesh(path: str, verts, colors, faces) -> None:
    """Write a PLY mesh with optional vertex colors."""
    num_verts = verts.shape[0]
    num_faces = faces.shape[0] if faces is not None else 0
    has_color = colors is not None and colors.shape[0] == num_verts and colors.shape[1] == 3

    header_lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {num_verts}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_color:
        header_lines.extend(
            [
                "property uchar red",
                "property uchar green",
                "property uchar blue",
            ]
        )
    header_lines.extend(
        [
            f"element face {num_faces}",
            "property list uchar int vertex_indices",
            "end_header",
        ]
    )

    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(header_lines))
        f.write("\n")

        # Vertices
        if num_verts > 0:
            v = verts.astype(float)
            c = colors.astype(int) if has_color else None
            for i in range(num_verts):
                x, y, z = v[i]
                line = f"{x} {y} {z}"
                if c is not None:
                    r, g, b = c[i]
                    line += f" {r} {g} {b}"
                f.write(line)
                f.write("\n")

        # Faces
        if num_faces > 0:
            fcs = faces.astype(int)
            for tri in fcs:
                a, b, c = tri
                f.write(f"3 {a} {b} {c}\n")
