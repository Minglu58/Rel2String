"""Generate and evaluate SPD motion using predicted relations from Stage 1."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
import inspect
import json
import os
import shutil
import subprocess
import tempfile
import wave
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))

import librosa
import matplotlib
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.lines import Line2D

from audio2pose.data.data import SPDCanonicalPoseDataset
from audio2pose.data.syncviolinist_pose import (
    decode_pose,
    require_same_representation,
    resolve_pose_representation,
)
from audio2pose.data.pretrained_audio import apply_audio_encoder_overrides, disable_cuda_audio_encoder_workers
from audio2pose.metrics.audio_alignment import add_audio_alignment_f1, compute_audio_alignment_metric_sums
from audio2pose.metrics.elgar_playability import (
    ELGAR_BOWING_BRIDGE_REFERENCE,
    compute_contact_diagnostic_sums,
    compute_contact_frame_diagnostics,
)
from audio2pose.metrics.elgar_original import PROTOCOL, ElgarOriginalAccumulator, PitchCache, protocol_metadata
from audio2pose.metrics.reporting import (
    print_metric_summary,
    print_seed_metric_summary,
    write_metric_reports,
    write_seed_metric_reports,
    write_sample_seed_metric_reports,
)
from audio2pose.metrics.syncviolinist import (
    SyncViolinistDTWAccumulator,
    compute_syncviolinist_metric_sums,
)
from audio2pose.models.condition_encoder import (
    infer_dual_condition_dims,
    make_audio_condition,
    make_relation_condition,
)
from audio2pose.pipelines.pipeline_spd_pose import (
    SPDPoseDiffusionPipeline,
    make_pose_padding_mask,
)
from audio2pose.pipelines.pose_context import PredictedPoseContext, continuous_windows
from audio2pose.pipelines.predicted_relations import (
    _plain,
    _stage1_metrics,
    _verify_dynamics_cache,
    _write_json,
    build_predicted_relation_dataset,
    load_mapping_model,
)
from data_process.canonical_string_frame import string_frame_to_world


BODY_EDGES_COCO = [
    (0, 1),
    (0, 2),
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (5, 6),
    (5, 7),
    (6, 8),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

DEFAULT_KNOWN_LOWER_BODY_INDICES = (11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
DEFAULT_BODY_WRIST_INDICES = (9, 10)
DEFAULT_INACTIVE_LOWER_BODY_INDICES = (13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
DEFAULT_FACE_JOINT_INDICES = tuple(range(23, 91))
DEFAULT_INACTIVE_GENERATION_INDICES = (
    DEFAULT_BODY_WRIST_INDICES + DEFAULT_INACTIVE_LOWER_BODY_INDICES + DEFAULT_FACE_JOINT_INDICES
)
POSE_POINT_SIZE = 5.0
INSTRUMENT_POINT_SIZE = 5.0
CONTACT_POINT_SIZE = 5.0
USED_FINGER_POINT_SIZE = 5.0
LEGEND_LINE_WIDTH = 1.2
LEFT_FINGER_CHAINS = (
    (92, 93, 94, 95),
    (96, 97, 98, 99),
    (100, 101, 102, 103),
    (104, 105, 106, 107),
    (108, 109, 110, 111),
)

HAND_LOCAL_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def make_hand_edges(offset: int) -> List[Tuple[int, int]]:
    return [(offset + a, offset + b) for a, b in HAND_LOCAL_EDGES]


def _safe_stem(text: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in text)


def _as_str(value) -> str:
    if isinstance(value, (list, tuple)):
        value = value[0]
    if torch.is_tensor(value):
        if value.ndim == 0:
            return str(value.item())
        return str(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        return str(value.tolist())
    return str(value)


def _as_int(value, default: int = 0) -> int:
    if isinstance(value, (list, tuple)):
        value = value[0]
    if torch.is_tensor(value):
        return int(value.item())
    if isinstance(value, np.ndarray):
        return int(value.item())
    try:
        return int(value)
    except Exception:
        return default


def normalize_index_tuple(indices: Optional[Iterable[int]]) -> Tuple[int, ...]:
    if indices is None:
        return ()
    return tuple(int(index) for index in indices)


def merge_known_pose_indices(
    pred_pose: np.ndarray,
    known_pose: np.ndarray,
    indices: Optional[Iterable[int]],
) -> Tuple[np.ndarray, Tuple[int, ...]]:
    merged = np.array(pred_pose, copy=True)
    joint_count = min(merged.shape[1], known_pose.shape[1])
    valid_indices = tuple(index for index in normalize_index_tuple(indices) if 0 <= index < joint_count)
    if valid_indices:
        merged[:, valid_indices, :] = known_pose[:, valid_indices, :]
    return merged, valid_indices


def build_pred_used_finger_context(
    gt_pose: np.ndarray,
    pred_pose: np.ndarray,
    extra_context: Optional[Dict[str, np.ndarray]],
) -> Optional[np.ndarray]:
    if not extra_context or "used_finger" not in extra_context:
        return None
    used_finger = np.asarray(extra_context["used_finger"], dtype=np.float32)
    if used_finger.ndim != 3 or used_finger.shape[1] != 4 or used_finger.shape[-1] != 3:
        return None

    frame_count = min(gt_pose.shape[0], pred_pose.shape[0], used_finger.shape[0])
    joint_count = min(gt_pose.shape[1], pred_pose.shape[1])
    pred_used = np.full((frame_count, 4, 3), np.nan, dtype=np.float32)
    for frame_idx in range(frame_count):
        if not np.isfinite(used_finger[frame_idx]).all():
            continue
        best_chain = None
        best_dist = float("inf")
        for chain in LEFT_FINGER_CHAINS:
            if max(chain) >= joint_count:
                continue
            gt_chain = gt_pose[frame_idx, list(chain), :]
            if not np.isfinite(gt_chain).all():
                continue
            dist = float(np.linalg.norm(gt_chain - used_finger[frame_idx], axis=-1).mean())
            if dist < best_dist:
                best_dist = dist
                best_chain = chain
        if best_chain is not None:
            pred_used[frame_idx] = pred_pose[frame_idx, list(best_chain), :]
    return pred_used


def make_prediction_extra_context(
    extra_context: Optional[Dict[str, np.ndarray]],
    gt_pose: np.ndarray,
    pred_pose: np.ndarray,
) -> Optional[Dict[str, np.ndarray]]:
    if not extra_context:
        return extra_context
    pred_context = {key: value for key, value in extra_context.items() if key != "used_finger"}
    pred_used = build_pred_used_finger_context(gt_pose, pred_pose, extra_context)
    if pred_used is not None:
        pred_context["used_finger"] = pred_used
    return pred_context


def mask_generation_indices(
    pose_padding_mask: torch.Tensor,
    joint_counts: torch.Tensor,
    indices: Optional[Iterable[int]],
) -> torch.Tensor:
    indices = normalize_index_tuple(indices)
    if not indices:
        return pose_padding_mask
    mask = pose_padding_mask.clone()
    batch_size, channels, frames, height, width = mask.shape
    flat = mask.reshape(batch_size, channels, frames, height * width)
    for batch_idx in range(batch_size):
        joint_count = int(joint_counts[batch_idx].item())
        for index in indices:
            if 0 <= index < joint_count and index < height * width:
                flat[batch_idx, :, :, index] = 0.0
    return flat.reshape(batch_size, channels, frames, height, width)


def _batch_value(batch: Dict, key: str, index: int):
    value = batch[key]
    if torch.is_tensor(value):
        return value[index]
    if isinstance(value, np.ndarray):
        return value[index]
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def save_prediction_npy_outputs(
    out_dir: Path,
    sample_id: str,
    start_frame: int,
    end_frame: int,
    pred_pose: np.ndarray,
    target_pose: np.ndarray,
    pose_valid: np.ndarray,
    string_frame_path: str,
    render_coordinate: str,
    include_instrument_context: bool,
    use_known_lower_body: bool,
    known_lower_body_indices: Tuple[int, ...],
    pose_context_frames: int = 0,
) -> Dict[str, Optional[str]]:
    stem = _safe_stem(f"{sample_id}_{start_frame:06d}_{end_frame:06d}")
    npy_dir = out_dir / "npy_predictions"
    npy_dir.mkdir(parents=True, exist_ok=True)

    raw_path = npy_dir / f"{stem}_pred_pose_string_frame.npy"
    np.save(raw_path, np.asarray(pred_pose, dtype=np.float32))

    extra_context = None
    if include_instrument_context:
        extra_context = load_instrument_context(
            string_frame_path,
            start_frame,
            end_frame,
            target_length=target_pose.shape[0],
        )
    render_target_pose, render_pred_pose, _, _, resolved_coordinate = convert_render_coordinate(
        target_pose,
        pred_pose,
        pose_valid,
        extra_context,
        string_frame_path,
        start_frame,
        end_frame,
        render_coordinate,
    )
    used_known_indices: Tuple[int, ...] = ()
    if use_known_lower_body:
        render_pred_pose, used_known_indices = merge_known_pose_indices(
            render_pred_pose,
            render_target_pose,
            known_lower_body_indices,
        )

    render_path = npy_dir / f"{stem}_pred_pose_{resolved_coordinate}.npy"
    np.save(render_path, np.asarray(render_pred_pose, dtype=np.float32))

    meta_path = npy_dir / f"{stem}_npy_meta.json"
    meta = {
        "sample_id": sample_id,
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "raw_npy_path": str(raw_path),
        "render_npy_path": str(render_path),
        "raw_coordinate": "string_frame",
        "render_coordinate": resolved_coordinate,
        "shape": list(np.asarray(pred_pose).shape),
        "use_known_lower_body": bool(use_known_lower_body),
        "known_lower_body_indices": list(used_known_indices),
        "string_frame_path": string_frame_path,
        "pose_context_frames": int(pose_context_frames),
        "pose_context_source": "previous_prediction_only" if pose_context_frames else "none",
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    return {
        "raw_npy_path": str(raw_path),
        "render_npy_path": str(render_path),
        "npy_meta_path": str(meta_path),
    }


def find_ffmpeg() -> Optional[str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        return ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def write_audio_clip(
    audio_path: str,
    out_path: Path,
    start_frame: int,
    end_frame: int,
    fps: int,
    sr: int,
) -> Optional[Path]:
    audio_file = Path(audio_path)
    if not audio_file.exists():
        return None
    audio, _ = librosa.load(str(audio_file), sr=sr, mono=True)
    start_sample = max(0, int(round(start_frame / fps * sr)))
    end_sample = min(len(audio), int(round(end_frame / fps * sr)))
    clip = audio[start_sample:end_sample]
    if clip.size == 0:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    clip_i16 = (np.clip(clip, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(out_path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(clip_i16.tobytes())
    return out_path


def mux_audio(video_path: Path, audio_path: Optional[Path], output_path: Path) -> Path:
    if audio_path is None:
        return video_path
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        return video_path
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        return video_path
    return output_path


def selected_indices(joint_count: int) -> List[int]:
    selected = []
    selected.extend([i for i in range(17) if i < joint_count and i not in DEFAULT_BODY_WRIST_INDICES])
    selected.extend([i for i in range(91, 112) if i < joint_count])
    selected.extend([i for i in range(112, 133) if i < joint_count])
    selected.extend([i for i in range(133, 135) if i < joint_count])
    return selected


def build_edges(joint_count: int) -> List[Tuple[int, int, str, float]]:
    edge_groups = [
        (BODY_EDGES_COCO, "#2563eb", 0.65),
        (make_hand_edges(91), "#16a34a", 0.45),
        (make_hand_edges(112), "#f97316", 0.45),
        ([(7, 91), (8, 112)], "#2563eb", 0.65),
        ([(133, 134)], "#111827", 0.8),
    ]
    out = []
    for edges, color, linewidth in edge_groups:
        for a, b in edges:
            if a < joint_count and b < joint_count:
                out.append((a, b, color, linewidth))
    return out


def point_color(index: int) -> str:
    if index < 17:
        return "#2563eb"
    if 91 <= index < 112:
        return "#16a34a"
    if 112 <= index < 133:
        return "#f97316"
    if 133 <= index < 135:
        return "#111827"
    return "#94a3b8"


def finite_points(points: np.ndarray) -> np.ndarray:
    return np.isfinite(points).all(axis=-1)


def normalize_vector(vector: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-8:
        if fallback is None:
            fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return np.asarray(fallback, dtype=np.float64)
    return vector / norm


def finite_mean(points: np.ndarray, axis=None):
    points = np.asarray(points, dtype=np.float64)
    valid = np.isfinite(points)
    count = valid.sum(axis=axis)
    total = np.where(valid, points, 0.0).sum(axis=axis)
    return total / np.maximum(count, 1)


def robust_frame_vector(values: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values).all(axis=-1)
    if not np.any(valid):
        return normalize_vector(fallback)
    return normalize_vector(np.median(values[valid], axis=0), fallback=fallback)


def estimate_body_front_view(gt_pose: np.ndarray):
    """
    Build display axes from the GT body:
      x: body left/right axis
      y: body front axis
      z: hip -> shoulder upright axis
    """
    if gt_pose.shape[1] <= 12:
        return np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)

    left_shoulder = gt_pose[:, 5, :]
    right_shoulder = gt_pose[:, 6, :]
    left_hip = gt_pose[:, 11, :]
    right_hip = gt_pose[:, 12, :]
    shoulder_center = 0.5 * (left_shoulder + right_shoulder)
    hip_center = 0.5 * (left_hip + right_hip)

    center_values = 0.5 * (shoulder_center + hip_center)
    center_valid = np.isfinite(center_values).all(axis=-1)
    if np.any(center_valid):
        center = np.median(center_values[center_valid], axis=0)
    else:
        center = np.zeros(3, dtype=np.float64)

    up = robust_frame_vector(shoulder_center - hip_center, np.array([0.0, 0.0, 1.0]))
    lateral_values = 0.5 * ((left_shoulder - right_shoulder) + (left_hip - right_hip))
    lateral = robust_frame_vector(lateral_values, np.array([1.0, 0.0, 0.0]))
    lateral = normalize_vector(lateral - np.dot(lateral, up) * up, np.array([1.0, 0.0, 0.0]))
    front = normalize_vector(np.cross(up, lateral), np.array([0.0, 1.0, 0.0]))

    if gt_pose.shape[1] > 0:
        nose_values = gt_pose[:, 0, :] - shoulder_center
        nose_front = robust_frame_vector(nose_values, front)
        if np.dot(front, nose_front) < 0:
            front *= -1.0
            lateral *= -1.0

    lateral = normalize_vector(np.cross(front, up), lateral)
    front = normalize_vector(np.cross(up, lateral), front)
    basis = np.stack([lateral, front, up], axis=-1)
    return center.astype(np.float64), basis.astype(np.float64)


def transform_points_for_view(points: np.ndarray, center: np.ndarray, basis: np.ndarray, flips: Tuple[bool, bool, bool]):
    points = np.asarray(points, dtype=np.float64)
    transformed = np.einsum("...i,ij->...j", points - center.reshape((1,) * (points.ndim - 1) + (3,)), basis)
    for axis, should_flip in enumerate(flips):
        if should_flip:
            transformed[..., axis] *= -1.0
    return transformed.astype(np.float32)


def transform_extra_context(extra_context: Optional[Dict[str, np.ndarray]], center, basis, flips):
    if not extra_context:
        return None
    return {
        key: transform_points_for_view(value, center, basis, flips)
        for key, value in extra_context.items()
        if value is not None
    }


def apply_visualization_transform(
    gt_pose: np.ndarray,
    pred_pose: np.ndarray,
    extra_context: Optional[Dict[str, np.ndarray]],
    pred_extra_context: Optional[Dict[str, np.ndarray]] = None,
    mode: str = "body_front",
    flip_x: bool = False,
    flip_y: bool = False,
    flip_z: bool = False,
):
    if mode == "none":
        return gt_pose, pred_pose, extra_context, pred_extra_context
    if mode != "body_front":
        raise ValueError("visualization_mode must be 'body_front' or 'none'.")
    center, basis = estimate_body_front_view(gt_pose)
    flips = (bool(flip_x), bool(flip_y), bool(flip_z))
    return (
        transform_points_for_view(gt_pose, center, basis, flips),
        transform_points_for_view(pred_pose, center, basis, flips),
        transform_extra_context(extra_context, center, basis, flips),
        transform_extra_context(pred_extra_context, center, basis, flips),
    )


def _flatten_context_arrays(extra_context: Optional[Dict[str, np.ndarray]]) -> List[np.ndarray]:
    if not extra_context:
        return []
    arrays = []
    for value in extra_context.values():
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.shape[-1] == 3:
            arrays.append(arr.reshape(-1, 3))
    return arrays


def axis_limits(
    gt_pose: np.ndarray,
    pred_pose: np.ndarray,
    selected: List[int],
    extra_context=None,
    pred_extra_context=None,
):
    arrays = [gt_pose[:, selected, :].reshape(-1, 3), pred_pose[:, selected, :].reshape(-1, 3)]
    arrays.extend(_flatten_context_arrays(extra_context))
    arrays.extend(_flatten_context_arrays(pred_extra_context))
    combined = np.concatenate(arrays, axis=0)
    valid = finite_points(combined)
    if not np.any(valid):
        return [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
    flat = combined[valid]
    mins = flat.min(axis=0)
    maxs = flat.max(axis=0)
    center = (mins + maxs) / 2.0
    span = float(np.max(maxs - mins))
    if span < 1e-6:
        span = 1.0
    half = span * 0.62
    return [(float(c - half), float(c + half)) for c in center]


def set_axis(ax, limits, elev: float, azim: float, title: str):
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def resolve_project_path(path_value: str) -> Path:
    path = Path(str(path_value))
    if path.is_absolute() or path.exists():
        return path
    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return project_path
    return path


def pad_time_array(arr: np.ndarray, start_frame: int, end_frame: int, target_length: int) -> np.ndarray:
    sliced = np.asarray(arr)[start_frame:end_frame]
    if sliced.shape[0] >= target_length:
        return sliced[:target_length].astype(np.float32)
    pad = target_length - sliced.shape[0]
    if sliced.shape[0] > 0:
        pad_values = np.repeat(sliced[-1:], pad, axis=0)
        return np.concatenate([sliced, pad_values], axis=0).astype(np.float32)
    return np.zeros((target_length,) + tuple(arr.shape[1:]), dtype=np.float32)


def load_instrument_context(
    string_frame_path: str,
    start_frame: int,
    end_frame: int,
    target_length: int,
) -> Dict[str, np.ndarray]:
    path = resolve_project_path(string_frame_path)
    if not path.exists():
        return {}
    context = {}
    with np.load(path, allow_pickle=False) as z:
        if "instrument_body_string_frame" in z:
            context["instrument_body"] = pad_time_array(
                z["instrument_body_string_frame"], start_frame, end_frame, target_length
            )
        if "strings_string_frame" in z:
            context["strings"] = pad_time_array(z["strings_string_frame"], start_frame, end_frame, target_length)
        if "contact_point_string_frame" in z:
            contact = pad_time_array(z["contact_point_string_frame"], start_frame, end_frame, target_length)
            context["contact_point"] = contact[:, None, :]
        if "used_finger_string_frame" in z:
            context["used_finger"] = pad_time_array(
                z["used_finger_string_frame"], start_frame, end_frame, target_length
            )
    return context


def normalize_render_coordinate(value: str) -> str:
    value = str(value or "string_frame").lower()
    aliases = {
        "string": "string_frame",
        "canonical": "string_frame",
        "canonical_string": "string_frame",
        "local": "string_frame",
        "spd": "world",
        "spd_world": "world",
        "raw": "world",
        "mocap": "world",
    }
    value = aliases.get(value, value)
    if value not in ("string_frame", "world"):
        raise ValueError("render_coordinate must be 'string_frame' or 'world'.")
    return value


def load_string_frame_world_transform(
    string_frame_path: str,
    start_frame: int,
    end_frame: int,
    target_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = resolve_project_path(string_frame_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing string_frame.npz for world rendering: {path}")
    with np.load(path, allow_pickle=False) as z:
        required = [
            "string_frame_origin_world",
            "string_frame_axes_world",
            "string_frame_scale_world",
        ]
        missing = [key for key in required if key not in z]
        if missing:
            raise KeyError(f"{path} is missing world transform keys: {missing}")
        origin = pad_time_array(z["string_frame_origin_world"], start_frame, end_frame, target_length)
        axes = pad_time_array(z["string_frame_axes_world"], start_frame, end_frame, target_length)
        scale = pad_time_array(z["string_frame_scale_world"], start_frame, end_frame, target_length)
    return origin.astype(np.float64), axes.astype(np.float64), scale.astype(np.float64)


def context_string_frame_to_world(
    extra_context: Optional[Dict[str, np.ndarray]],
    origin: np.ndarray,
    axes: np.ndarray,
    scale: np.ndarray,
) -> Optional[Dict[str, np.ndarray]]:
    if not extra_context:
        return extra_context
    return {
        key: string_frame_to_world(value, origin, axes, scale).astype(np.float32)
        for key, value in extra_context.items()
        if value is not None
    }


def convert_render_coordinate(
    target_pose: np.ndarray,
    pred_pose: np.ndarray,
    pose_valid: np.ndarray,
    extra_context: Optional[Dict[str, np.ndarray]],
    string_frame_path: str,
    start_frame: int,
    end_frame: int,
    render_coordinate: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Dict[str, np.ndarray]], str]:
    render_coordinate = normalize_render_coordinate(render_coordinate)
    if render_coordinate == "string_frame":
        return target_pose, pred_pose, pose_valid, extra_context, render_coordinate

    target_length = int(target_pose.shape[0])
    origin, axes, scale = load_string_frame_world_transform(
        string_frame_path,
        start_frame,
        end_frame,
        target_length,
    )
    target_world = string_frame_to_world(target_pose, origin, axes, scale).astype(np.float32)
    pred_world = string_frame_to_world(pred_pose, origin, axes, scale).astype(np.float32)
    context_world = context_string_frame_to_world(extra_context, origin, axes, scale)
    world_valid = pose_valid & np.isfinite(target_world).all(axis=-1)
    return target_world, pred_world, world_valid, context_world, render_coordinate


def instrument_body_edges(n_points: int) -> List[Tuple[int, int]]:
    if n_points == 5:
        return [(0, 1), (0, 2), (1, 3), (2, 4)]
    if n_points >= 7:
        return [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6)]
    return [(i, i + 1) for i in range(max(0, n_points - 1))]


def set_scatter_3d(scatter, points: np.ndarray):
    points = np.asarray(points, dtype=np.float64)
    valid = finite_points(points)
    frame = points.copy()
    frame[~valid] = np.nan
    scatter._offsets3d = (frame[:, 0], frame[:, 1], frame[:, 2])


def create_extra_artists(ax, extra_context: Optional[Dict[str, np.ndarray]], alpha: float = 0.68):
    if not extra_context:
        return {}
    artists = {}
    if "instrument_body" in extra_context:
        n_points = extra_context["instrument_body"].shape[1]
        scatter = ax.scatter([], [], [], s=INSTRUMENT_POINT_SIZE, c="#dc2626", depthshade=False, alpha=alpha)
        lines = []
        for _ in instrument_body_edges(n_points):
            line, = ax.plot([], [], [], color="#dc2626", linewidth=0.55, alpha=alpha)
            lines.append(line)
        artists["instrument_body"] = (scatter, lines, instrument_body_edges(n_points))
    if "strings" in extra_context:
        lines = []
        for _ in range(extra_context["strings"].shape[1]):
            line, = ax.plot([], [], [], color="#7c3aed", linewidth=0.6, alpha=alpha)
            lines.append(line)
        artists["strings"] = lines
    if "contact_point" in extra_context:
        artists["contact_point"] = ax.scatter(
            [], [], [], s=CONTACT_POINT_SIZE, c="#db2777", marker="x", depthshade=False, alpha=alpha
        )
    if "used_finger" in extra_context:
        scatter = ax.scatter([], [], [], s=USED_FINGER_POINT_SIZE, c="#db2777", depthshade=False, alpha=alpha * 0.85)
        line, = ax.plot([], [], [], color="#db2777", linewidth=0.45, alpha=alpha * 0.85)
        artists["used_finger"] = (scatter, line)
    return artists


def update_line_3d(line, points: np.ndarray):
    points = np.asarray(points, dtype=np.float64)
    if finite_points(points).all():
        line.set_data_3d(points[:, 0], points[:, 1], points[:, 2])
    else:
        line.set_data_3d([np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan])


def update_extra_artists(artists: Dict, extra_context: Optional[Dict[str, np.ndarray]], frame_idx: int):
    if not artists or not extra_context:
        return []
    changed = []
    if "instrument_body" in artists and "instrument_body" in extra_context:
        scatter, lines, edges = artists["instrument_body"]
        points = extra_context["instrument_body"][frame_idx]
        set_scatter_3d(scatter, points)
        changed.append(scatter)
        for line, (a, b) in zip(lines, edges):
            update_line_3d(line, points[[a, b]])
            changed.append(line)
    if "strings" in artists and "strings" in extra_context:
        strings = extra_context["strings"][frame_idx]
        for line, segment in zip(artists["strings"], strings):
            update_line_3d(line, segment)
            changed.append(line)
    if "contact_point" in artists and "contact_point" in extra_context:
        scatter = artists["contact_point"]
        set_scatter_3d(scatter, extra_context["contact_point"][frame_idx])
        changed.append(scatter)
    if "used_finger" in artists and "used_finger" in extra_context:
        scatter, line = artists["used_finger"]
        points = extra_context["used_finger"][frame_idx]
        set_scatter_3d(scatter, points)
        update_line_3d(line, points)
        changed.extend([scatter, line])
    return changed


def render_pose_comparison_mp4(
    gt_pose: np.ndarray,
    pred_pose: np.ndarray,
    pose_valid: np.ndarray,
    output_path: Path,
    extra_context: Optional[Dict[str, np.ndarray]] = None,
    pred_extra_context: Optional[Dict[str, np.ndarray]] = None,
    fps: int = 30,
    elev: float = 18.0,
    azim: float = -70.0,
    dpi: int = 150,
    bitrate: int = 2400,
    title: str = "",
    visualization_mode: str = "body_front",
    flip_x: bool = False,
    flip_y: bool = False,
    flip_z: bool = False,
) -> Optional[Path]:
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        return None
    matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg
    if pred_extra_context is None:
        pred_extra_context = extra_context

    gt_pose, pred_pose, extra_context, pred_extra_context = apply_visualization_transform(
        gt_pose,
        pred_pose,
        extra_context,
        pred_extra_context=pred_extra_context,
        mode=visualization_mode,
        flip_x=flip_x,
        flip_y=flip_y,
        flip_z=flip_z,
    )
    joint_count = min(gt_pose.shape[1], pred_pose.shape[1])
    selected = selected_indices(joint_count)
    if not selected:
        return None
    selected_pos = {index: pos for pos, index in enumerate(selected)}
    edges = build_edges(joint_count)
    limits = axis_limits(
        gt_pose,
        pred_pose,
        selected,
        extra_context=extra_context,
        pred_extra_context=pred_extra_context,
    )
    colors = [point_color(index) for index in selected]

    fig = plt.figure(figsize=(12.5, 6.0), dpi=dpi)
    ax_gt = fig.add_subplot(121, projection="3d")
    ax_pr = fig.add_subplot(122, projection="3d")
    fig.patch.set_facecolor("white")
    set_axis(ax_gt, limits, elev, azim, "ground truth")
    set_axis(ax_pr, limits, elev, azim, "prediction")

    legend_items = [
        Line2D([0], [0], color="#2563eb", lw=LEGEND_LINE_WIDTH, label="body"),
        Line2D([0], [0], color="#16a34a", lw=LEGEND_LINE_WIDTH, label="left hand"),
        Line2D([0], [0], color="#f97316", lw=LEGEND_LINE_WIDTH, label="right hand"),
        Line2D([0], [0], color="#111827", lw=LEGEND_LINE_WIDTH, label="bow"),
        Line2D([0], [0], color="#dc2626", lw=LEGEND_LINE_WIDTH, label="instrument body"),
        Line2D([0], [0], color="#7c3aed", lw=LEGEND_LINE_WIDTH, label="strings"),
        Line2D([0], [0], color="#db2777", lw=LEGEND_LINE_WIDTH, label="contact/finger"),
    ]
    ax_gt.legend(handles=legend_items, loc="upper left", fontsize=8)

    gt_selected = gt_pose[:, selected, :]
    pr_selected = pred_pose[:, selected, :]
    valid_selected = pose_valid[:, selected] if pose_valid.shape[1] >= max(selected) + 1 else finite_points(gt_selected)

    def init_scatter(ax, points):
        frame = points[0].copy()
        valid = finite_points(frame)
        frame[~valid] = np.nan
        return ax.scatter(frame[:, 0], frame[:, 1], frame[:, 2], s=POSE_POINT_SIZE, c=colors, depthshade=False)

    scatter_gt = init_scatter(ax_gt, gt_selected)
    scatter_pr = init_scatter(ax_pr, pr_selected)
    gt_lines = []
    pr_lines = []
    for a, b, color, linewidth in edges:
        if a not in selected_pos or b not in selected_pos:
            continue
        line_gt, = ax_gt.plot([], [], [], color=color, linewidth=linewidth, alpha=0.84)
        line_pr, = ax_pr.plot([], [], [], color=color, linewidth=linewidth, alpha=0.84)
        gt_lines.append((a, b, line_gt))
        pr_lines.append((a, b, line_pr))

    gt_extra_artists = create_extra_artists(ax_gt, extra_context, alpha=0.78)
    pr_extra_artists = create_extra_artists(ax_pr, pred_extra_context, alpha=0.58)

    frame_title = fig.suptitle("", fontsize=12)
    per_frame_mpjpe = np.linalg.norm(pred_pose - gt_pose, axis=-1)

    def update_scatter(scatter, points, valid):
        frame = points.copy()
        frame[~valid] = np.nan
        scatter._offsets3d = (frame[:, 0], frame[:, 1], frame[:, 2])

    def update_lines(line_items, pose_frame, valid_full):
        for a, b, line in line_items:
            if valid_full[a] and valid_full[b]:
                pts = pose_frame[[a, b]]
                line.set_data_3d(pts[:, 0], pts[:, 1], pts[:, 2])
            else:
                line.set_data_3d([np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan])

    def update(frame_idx: int):
        valid_full = pose_valid[frame_idx]
        valid_sel = valid_selected[frame_idx] & finite_points(gt_selected[frame_idx])
        update_scatter(scatter_gt, gt_selected[frame_idx], valid_sel)
        update_scatter(scatter_pr, pr_selected[frame_idx], finite_points(pr_selected[frame_idx]))
        update_lines(gt_lines, gt_pose[frame_idx], valid_full)
        update_lines(pr_lines, pred_pose[frame_idx], np.ones(joint_count, dtype=bool))
        changed_extra = []
        changed_extra.extend(update_extra_artists(gt_extra_artists, extra_context, frame_idx))
        changed_extra.extend(update_extra_artists(pr_extra_artists, pred_extra_context, frame_idx))
        valid = pose_valid[frame_idx]
        mpjpe = float(np.mean(per_frame_mpjpe[frame_idx][valid])) if np.any(valid) else float("nan")
        frame_title.set_text(f"{title} | frame {frame_idx + 1}/{gt_pose.shape[0]} | MPJPE {mpjpe:.4f}")
        return (
            [scatter_gt, scatter_pr, frame_title]
            + [x[2] for x in gt_lines]
            + [x[2] for x in pr_lines]
            + changed_extra
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=gt_pose.shape[0],
        interval=1000 / fps,
        blit=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, bitrate=bitrate, metadata={"title": title or "SPD pose comparison"})
    animation.save(str(output_path), writer=writer, dpi=dpi)
    plt.close(fig)
    return output_path


def metric_sum(values: np.ndarray, mask: np.ndarray) -> Tuple[float, int]:
    mask = mask.astype(bool)
    if not np.any(mask):
        return 0.0, 0
    return float(values[mask].sum()), int(mask.sum())


def compute_metric_sums(
    pred_pose: np.ndarray,
    target_pose: np.ndarray,
    valid: np.ndarray,
    ignored_indices: Tuple[int, ...] = DEFAULT_BODY_WRIST_INDICES,
) -> Dict[str, Tuple[float, int]]:
    joint_count = min(pred_pose.shape[1], target_pose.shape[1])
    native_syncviolinist_layout = joint_count == 75
    pred_pose = pred_pose[:, :joint_count]
    target_pose = target_pose[:, :joint_count]
    valid = valid[:, :joint_count].astype(bool)
    # DEFAULT_BODY_WRIST_INDICES belongs to SPD/COCO-WholeBody.  Indices 9 and
    # 10 mean right shoulder/right arm in the released SyncViolinist skeleton,
    # so applying that mask to the native 75-joint layout would be incorrect.
    if not native_syncviolinist_layout:
        for index in ignored_indices:
            if 0 <= index < joint_count:
                valid[:, index] = False

    dist = np.linalg.norm(pred_pose - target_pose, axis=-1)
    metrics = {"mpjpe": metric_sum(dist, valid)}

    if native_syncviolinist_layout:
        metrics["body_mpjpe"] = metric_sum(dist, valid)
        metrics["left_hand_mpjpe"] = metric_sum(dist[:, 39:63], valid[:, 39:63])
        metrics["right_hand_mpjpe"] = metric_sum(dist[:, 12:36], valid[:, 12:36])
        metrics["bow_mpjpe"] = (0.0, 0)
    else:
        body_end = min(133, joint_count)
        metrics["body_mpjpe"] = metric_sum(dist[:, :body_end], valid[:, :body_end])

    if not native_syncviolinist_layout and joint_count > 91:
        left_end = min(112, joint_count)
        metrics["left_hand_mpjpe"] = metric_sum(dist[:, 91:left_end], valid[:, 91:left_end])
    elif not native_syncviolinist_layout:
        metrics["left_hand_mpjpe"] = (0.0, 0)

    if not native_syncviolinist_layout and joint_count > 112:
        right_end = min(133, joint_count)
        metrics["right_hand_mpjpe"] = metric_sum(dist[:, 112:right_end], valid[:, 112:right_end])
    elif not native_syncviolinist_layout:
        metrics["right_hand_mpjpe"] = (0.0, 0)

    if not native_syncviolinist_layout and joint_count > 133:
        metrics["bow_mpjpe"] = metric_sum(dist[:, 133:joint_count], valid[:, 133:joint_count])
    elif not native_syncviolinist_layout:
        metrics["bow_mpjpe"] = (0.0, 0)

    # metrics["pck_0.03"] = (float(((dist < 0.03) & valid).sum()), int(valid.sum()))
    # metrics["pck_0.05"] = (float(((dist < 0.05) & valid).sum()), int(valid.sum()))

    if pred_pose.shape[0] > 1:
        pred_vel = np.diff(pred_pose, axis=0)
        target_vel = np.diff(target_pose, axis=0)
        vel_valid = valid[1:] & valid[:-1]
        vel_dist = np.linalg.norm(pred_vel - target_vel, axis=-1)
        metrics["velocity_mpjpe"] = metric_sum(vel_dist, vel_valid)
    else:
        metrics["velocity_mpjpe"] = (0.0, 0)

    if pred_pose.shape[0] > 2:
        pred_acc = pred_pose[2:] - 2 * pred_pose[1:-1] + pred_pose[:-2]
        target_acc = target_pose[2:] - 2 * target_pose[1:-1] + target_pose[:-2]
        acc_valid = valid[2:] & valid[1:-1] & valid[:-2]
        acc_dist = np.linalg.norm(pred_acc - target_acc, axis=-1)
        metrics["acceleration_mpjpe"] = metric_sum(acc_dist, acc_valid)
    else:
        metrics["acceleration_mpjpe"] = (0.0, 0)

    return metrics


def smooth_bow_endpoints(
    pred_pose: np.ndarray,
    window: int = 1,
    endpoint_slice: Tuple[int, int] = (133, 135),
) -> np.ndarray:
    """Smooth only the two bow endpoints in time for evaluation ablations."""
    window = int(window)
    if window <= 1:
        return np.asarray(pred_pose, dtype=np.float32)
    if window % 2 == 0:
        raise ValueError("bow_endpoint_smoothing_window must be odd, e.g. 1, 3, or 5.")

    output = np.asarray(pred_pose, dtype=np.float32).copy()
    start, end = (int(endpoint_slice[0]), int(endpoint_slice[1]))
    if output.ndim != 3 or output.shape[-1] != 3 or start < 0 or end > output.shape[1]:
        return output
    if output.shape[0] < 2:
        return output

    radius = window // 2
    values = output[:, start:end, :]
    padded = np.pad(values, ((radius, radius), (0, 0), (0, 0)), mode="edge")
    kernel = np.full((window,), 1.0 / float(window), dtype=np.float32)
    smoothed = np.stack(
        [
            np.convolve(padded[:, joint_index, coordinate], kernel, mode="valid")
            for joint_index in range(end - start)
            for coordinate in range(3)
        ],
        axis=-1,
    ).reshape(output.shape[0], end - start, 3)
    output[:, start:end, :] = smoothed
    return output


def pose_to_grid_like(
    pred_grid: torch.Tensor,
    pred_pose: np.ndarray,
    pose_grid_size: Tuple[int, int],
) -> torch.Tensor:
    """Write a postprocessed pose back into a single prediction grid."""
    grid = pred_grid.detach().cpu().numpy().copy()
    height, width = (int(pose_grid_size[0]), int(pose_grid_size[1]))
    flat = grid.transpose(1, 2, 3, 0).reshape(grid.shape[1], height * width, 3)
    joint_count = min(flat.shape[1], pred_pose.shape[1])
    flat[:, :joint_count, :] = pred_pose[:, :joint_count, :]
    return torch.from_numpy(
        flat.reshape(grid.shape[1], height, width, 3).transpose(3, 0, 1, 2)
    )


def postprocess_sample_pose(
    pred_pose, *, smoothing_window, local_refiner=None, relation=None,
    strings=None, device="cpu", relation_condition_mode="normal",
    apply_left_hand_refiner=True, apply_bow_event_refiner=True,
):
    """Apply the shared smoothing, then the optional frozen local correction."""
    pred_pose = smooth_bow_endpoints(pred_pose, window=int(smoothing_window))
    if local_refiner is None:
        return pred_pose
    local_relation = relation.to(device=device, dtype=torch.float32)
    if relation_condition_mode == "zero":
        local_relation = torch.zeros_like(local_relation)
    refined, _ = local_refiner(
        torch.from_numpy(pred_pose[None]).to(device=device, dtype=torch.float32),
        local_relation, strings.to(device=device, dtype=torch.float32),
        apply_left_hand=bool(apply_left_hand_refiner),
        apply_bow_event=bool(apply_bow_event_refiner),
    )
    return refined[0].detach().cpu().numpy()


def finalize_metrics(sums: Dict[str, float], counts: Dict[str, int]) -> Dict[str, Optional[float]]:
    out = {}
    for key in sorted(sums):
        if counts.get(key, 0) <= 0:
            out[key] = None
        else:
            out[key] = float(sums[key] / counts[key])
    return out


UNREPORTED_EVALUATION_METRICS = frozenset({
    "audio_bowing_f1",
    "audio_motion_lag_ms",
    "syncviolinist_l1",
    "syncviolinist_dtw",
    "syncviolinist_motion_jerk",
})


def _reported_metrics(metrics: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    return {key: value for key, value in metrics.items() if key not in UNREPORTED_EVALUATION_METRICS}


def add_metric_sums(total_sums: Dict[str, float], total_counts: Dict[str, int], metrics: Dict[str, Tuple[float, int]]):
    for key, (value_sum, count) in metrics.items():
        total_sums[key] = total_sums.get(key, 0.0) + float(value_sum)
        total_counts[key] = total_counts.get(key, 0) + int(count)


def finalize_sample_metrics(sums_by_sample, counts_by_sample):
    """Pool the window-based metrics for each recording."""
    result = {}
    for sample_id, sums in sorted(sums_by_sample.items()):
        metrics = finalize_metrics(sums, counts_by_sample[sample_id])
        add_audio_alignment_f1(metrics)
        result[sample_id] = _reported_metrics(metrics)
    return result


def _tensor_or_array_to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _masked_dist_stats(dist: np.ndarray, valid: np.ndarray, prefix: str) -> Dict[str, Optional[float]]:
    dist = np.asarray(dist, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(dist)
    if not np.any(valid):
        return {
            f"{prefix}_mean": None,
            f"{prefix}_median": None,
            f"{prefix}_p95": None,
        }
    values = dist[valid]
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
    }


def _metric_tuple_to_value(value: Tuple[float, int]) -> Optional[float]:
    value_sum, count = value
    if count <= 0:
        return None
    return float(value_sum / count)


def summarize_contact_diagnostics(pred_pose: np.ndarray, batch_item: Dict) -> Dict[str, Optional[float]]:
    if not {"strings", "contact_target_point", "contact_valid"}.issubset(batch_item):
        return {}
    distance_scale = _tensor_or_array_to_numpy(batch_item["string_frame_scale"]) if "string_frame_scale" in batch_item else None
    sums = compute_contact_diagnostic_sums(
        pred_pose,
        _tensor_or_array_to_numpy(batch_item["strings"]),
        _tensor_or_array_to_numpy(batch_item["contact_target_point"]),
        _tensor_or_array_to_numpy(batch_item["contact_valid"]),
        distance_scale=distance_scale,
    )
    return {key: _metric_tuple_to_value(value) for key, value in sums.items()}


def write_contact_diagnostics_csv(path: Path, pred_pose: np.ndarray, batch_item: Dict) -> Optional[Path]:
    if not {"strings", "contact_target_point", "contact_valid"}.issubset(batch_item):
        return None
    distance_scale = _tensor_or_array_to_numpy(batch_item["string_frame_scale"]) if "string_frame_scale" in batch_item else None
    diagnostics = compute_contact_frame_diagnostics(
        pred_pose,
        _tensor_or_array_to_numpy(batch_item["strings"]),
        _tensor_or_array_to_numpy(batch_item["contact_target_point"]),
        _tensor_or_array_to_numpy(batch_item["contact_valid"]),
        distance_scale=distance_scale,
    )
    keys = list(diagnostics.keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        frame_count = len(diagnostics["frame"])
        for frame_idx in range(frame_count):
            writer.writerow([diagnostics[key][frame_idx] for key in keys])
    return path


def compute_playability_diagnostics(pred_pose: np.ndarray, target_pose: np.ndarray, batch_item: Dict) -> Dict[str, Optional[float]]:
    diagnostics: Dict[str, Optional[float]] = {}
    pred_pose = np.asarray(pred_pose, dtype=np.float32)
    target_pose = np.asarray(target_pose, dtype=np.float32)
    frame_count = pred_pose.shape[0]

    if "contact_target_point" in batch_item and "contact_valid" in batch_item:
        contact_target = _tensor_or_array_to_numpy(batch_item["contact_target_point"]).astype(np.float32)[:frame_count]
        contact_valid = _tensor_or_array_to_numpy(batch_item["contact_valid"]).astype(bool)[:frame_count]
        tip_indices = [idx for idx in (95, 99, 103, 107, 111) if idx < pred_pose.shape[1]]
        if tip_indices:
            pred_tip_dist = np.linalg.norm(pred_pose[:, tip_indices, :] - contact_target[:, None, :], axis=-1).min(axis=1)
            target_tip_dist = np.linalg.norm(target_pose[:, tip_indices, :] - contact_target[:, None, :], axis=-1).min(axis=1)
            diagnostics.update(_masked_dist_stats(pred_tip_dist, contact_valid, "pred_contact_tip_dist"))
            diagnostics.update(_masked_dist_stats(target_tip_dist, contact_valid, "gt_contact_tip_dist"))

    if "bow_target_point" in batch_item and "bow_valid" in batch_item and pred_pose.shape[1] > 134:
        bow_target = _tensor_or_array_to_numpy(batch_item["bow_target_point"]).astype(np.float32)[:frame_count]
        bow_valid = _tensor_or_array_to_numpy(batch_item["bow_valid"]).astype(bool)[:frame_count]
        if "bow_t_on_bow" in batch_item:
            bow_t = _tensor_or_array_to_numpy(batch_item["bow_t_on_bow"]).astype(np.float32)[:frame_count]
        else:
            bow_t = np.full((frame_count,), 0.5, dtype=np.float32)
        bow_t = np.clip(bow_t, 0.0, 1.0)[:, None]
        pred_bow_point = pred_pose[:, 133, :] + bow_t * (pred_pose[:, 134, :] - pred_pose[:, 133, :])
        target_bow_point = target_pose[:, 133, :] + bow_t * (target_pose[:, 134, :] - target_pose[:, 133, :])
        pred_bow_dist = np.linalg.norm(pred_bow_point - bow_target, axis=-1)
        target_bow_dist = np.linalg.norm(target_bow_point - bow_target, axis=-1)
        diagnostics.update(_masked_dist_stats(pred_bow_dist, bow_valid, "pred_bow_contact_target_dist"))
        diagnostics.update(_masked_dist_stats(target_bow_dist, bow_valid, "gt_bow_contact_target_dist"))

        if "bow_target_direction" in batch_item:
            bow_dir_target = _tensor_or_array_to_numpy(batch_item["bow_target_direction"]).astype(np.float32)[:frame_count]
            pred_dir = pred_pose[:, 134, :] - pred_pose[:, 133, :]
            pred_dir = pred_dir / np.maximum(np.linalg.norm(pred_dir, axis=-1, keepdims=True), 1e-8)
            target_dir = bow_dir_target / np.maximum(np.linalg.norm(bow_dir_target, axis=-1, keepdims=True), 1e-8)
            direction_error = 1.0 - np.abs(np.sum(pred_dir * target_dir, axis=-1))
            diagnostics.update(_masked_dist_stats(direction_error, bow_valid, "pred_bow_direction_error"))

    return diagnostics


def unpack_pose_grid(pose_grid: torch.Tensor, joint_count: int, pose_grid_size: Iterable[int]) -> np.ndarray:
    pose = SPDCanonicalPoseDataset.unpack_pose_grid(
        pose_grid.detach().cpu(),
        int(joint_count),
        pose_grid_size=tuple(int(x) for x in pose_grid_size),
    )
    if torch.is_tensor(pose):
        pose = pose.detach().cpu().numpy()
    return np.asarray(pose, dtype=np.float32)


def make_generator(device: str, seed: Optional[int]):
    if seed is None:
        return None
    try:
        return torch.Generator(device=device).manual_seed(int(seed))
    except Exception:
        return torch.Generator().manual_seed(int(seed))


def resolve_sampling_seeds(
    seed: Optional[int],
    sampling_seeds: Optional[Iterable[int]],
    num_sampling_seeds: int,
) -> Tuple[int, ...]:
    if sampling_seeds is not None:
        seeds = tuple(int(value) for value in sampling_seeds)
    else:
        if seed is None:
            raise ValueError("seed must be set when sampling_seeds is not provided.")
        count = int(num_sampling_seeds)
        if count <= 0:
            raise ValueError("num_sampling_seeds must be greater than zero.")
        seeds = tuple(int(seed) + offset for offset in range(count))
    if not seeds:
        raise ValueError("sampling_seeds must contain at least one seed.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("sampling_seeds must not contain duplicate values.")
    return seeds


def parse_torch_dtype(value: str, device: str):
    value = str(value or "float32").lower()
    if str(device).startswith("cpu"):
        return torch.float32
    if value in ("fp16", "float16", "half"):
        return torch.float16
    if value in ("bf16", "bfloat16"):
        return torch.bfloat16
    if value in ("fp32", "float32", "full"):
        return torch.float32
    raise ValueError(f"Unknown torch_dtype: {value}")


def save_sample_outputs(
    out_dir: Path,
    sample_id: str,
    start_frame: int,
    end_frame: int,
    audio_path: str,
    string_frame_path: str,
    pred_pose: np.ndarray,
    target_pose: np.ndarray,
    pose_valid: np.ndarray,
    pred_grid: torch.Tensor,
    target_grid: torch.Tensor,
    batch_item: Dict,
    sample_metrics: Dict[str, Optional[float]],
    fps: int,
    audio_sr: int,
    render_fps: int,
    elev: float,
    azim: float,
    save_video: bool,
    include_instrument_context: bool = True,
    render_coordinate: str = "string_frame",
    use_known_lower_body: bool = False,
    known_lower_body_indices: Tuple[int, ...] = DEFAULT_KNOWN_LOWER_BODY_INDICES,
    visualization_mode: str = "body_front",
    flip_x: bool = False,
    flip_y: bool = False,
    flip_z: bool = False,
) -> Dict[str, Optional[str]]:
    stem = _safe_stem(f"{sample_id}_{start_frame:06d}_{end_frame:06d}")
    sample_dir = out_dir / stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    extra_context = None
    if include_instrument_context:
        extra_context = load_instrument_context(
            string_frame_path,
            start_frame,
            end_frame,
            target_length=target_pose.shape[0],
        )
    render_target_pose, render_pred_pose, render_pose_valid, render_extra_context, render_coordinate = (
        convert_render_coordinate(
            target_pose,
            pred_pose,
            pose_valid,
            extra_context,
            string_frame_path,
            start_frame,
            end_frame,
            render_coordinate,
        )
    )
    render_pred_pose_raw = np.array(render_pred_pose, copy=True)
    used_known_indices: Tuple[int, ...] = ()
    if use_known_lower_body:
        render_pred_pose, used_known_indices = merge_known_pose_indices(
            render_pred_pose,
            render_target_pose,
            known_lower_body_indices,
        )
    render_pred_extra_context = make_prediction_extra_context(
        render_extra_context,
        render_target_pose,
        render_pred_pose,
    )
    render_pred_used_finger = None
    if render_pred_extra_context and "used_finger" in render_pred_extra_context:
        render_pred_used_finger = render_pred_extra_context["used_finger"]

    npz_path = sample_dir / f"{stem}_poses.npz"
    arrays_to_save = dict(
        pred_pose=pred_pose,
        target_pose=target_pose,
        pose_valid=pose_valid,
        render_pred_pose_raw=render_pred_pose_raw,
        render_pred_pose=render_pred_pose,
        render_target_pose=render_target_pose,
        render_pose_valid=render_pose_valid,
        render_coordinate=np.array(render_coordinate),
        use_known_lower_body=np.array(bool(use_known_lower_body)),
        known_lower_body_indices=np.asarray(used_known_indices, dtype=np.int64),
        pred_pose_grid=pred_grid.detach().cpu().numpy(),
        target_pose_grid=target_grid.detach().cpu().numpy(),
        condition=batch_item["condition"].detach().cpu().numpy(),
        relation=batch_item["relation"].detach().cpu().numpy(),
        relation_mask=batch_item["relation_mask"].detach().cpu().numpy(),
        contact_valid=_tensor_or_array_to_numpy(batch_item["contact_valid"]),
        bow_valid=_tensor_or_array_to_numpy(batch_item["bow_valid"]),
        contact_target_point=_tensor_or_array_to_numpy(batch_item["contact_target_point"]),
        bow_target_point=_tensor_or_array_to_numpy(batch_item["bow_target_point"]),
        bow_target_direction=_tensor_or_array_to_numpy(batch_item["bow_target_direction"]),
        bow_t_on_bow=_tensor_or_array_to_numpy(batch_item["bow_t_on_bow"]),
        sample_id=np.array(sample_id),
        audio_path=np.array(audio_path),
        string_frame_path=np.array(string_frame_path),
        start_frame=np.array(start_frame),
        end_frame=np.array(end_frame),
    )
    if "string_frame_scale" in batch_item:
        arrays_to_save["string_frame_scale"] = _tensor_or_array_to_numpy(batch_item["string_frame_scale"])
    if "strings" in batch_item:
        arrays_to_save["strings"] = _tensor_or_array_to_numpy(batch_item["strings"])
    if render_pred_used_finger is not None:
        arrays_to_save["render_pred_used_finger"] = render_pred_used_finger
    np.savez_compressed(npz_path, **arrays_to_save)
    raw_npy_path = sample_dir / f"{stem}_pred_pose_string_frame.npy"
    render_npy_path = sample_dir / f"{stem}_pred_pose_{render_coordinate}.npy"
    np.save(raw_npy_path, np.asarray(pred_pose, dtype=np.float32))
    np.save(render_npy_path, np.asarray(render_pred_pose, dtype=np.float32))
    contact_diag_path = write_contact_diagnostics_csv(
        sample_dir / f"{stem}_contact_diagnostics.csv",
        pred_pose,
        batch_item,
    )

    audio_clip = write_audio_clip(
        audio_path,
        sample_dir / f"{stem}_audio.wav",
        start_frame,
        end_frame,
        fps=fps,
        sr=audio_sr,
    )

    video_path = None
    silent_path = None
    if save_video:
        silent_path = sample_dir / f"{stem}_silent.mp4"
        rendered = render_pose_comparison_mp4(
            render_target_pose,
            render_pred_pose,
            render_pose_valid,
            silent_path,
            extra_context=render_extra_context,
            pred_extra_context=render_pred_extra_context,
            fps=render_fps,
            elev=elev,
            azim=azim,
            title=f"{stem} | {render_coordinate}" + (" | known lower body" if use_known_lower_body else ""),
            visualization_mode=visualization_mode,
            flip_x=flip_x,
            flip_y=flip_y,
            flip_z=flip_z,
        )
        if rendered is not None:
            video_path = mux_audio(rendered, audio_clip, sample_dir / f"{stem}_with_audio.mp4")

    report = {
        "sample_id": sample_id,
        "audio_path": audio_path,
        "string_frame_path": string_frame_path,
        "audio_clip_path": str(audio_clip) if audio_clip else None,
        "npz_path": str(npz_path),
        "raw_npy_path": str(raw_npy_path),
        "render_npy_path": str(render_npy_path),
        "contact_diagnostics_csv": str(contact_diag_path) if contact_diag_path else None,
        "silent_video_path": str(silent_path) if silent_path else None,
        "video_path": str(video_path) if video_path else None,
        "render_coordinate": render_coordinate,
        "visualization_mode": visualization_mode,
        "use_known_lower_body": bool(use_known_lower_body),
        "known_lower_body_indices": list(used_known_indices),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_time_sec": start_frame / fps,
        "end_time_sec": end_frame / fps,
        "metrics": sample_metrics,
    }
    with (sample_dir / f"{stem}_report.json").open("w", encoding="utf-8") as f:
        playability = compute_playability_diagnostics(pred_pose, target_pose, batch_item)
        if playability:
            report["playability"] = playability
        contact_diagnostics = summarize_contact_diagnostics(pred_pose, batch_item)
        if contact_diagnostics:
            report["contact_diagnostics"] = contact_diagnostics
        json.dump(report, f, indent=2, sort_keys=True)
    return report


def _run_single_seed(
    checkpoint_dir: str,
    test_data: Dict,
    output_dir: Optional[str] = None,
    train_config: Optional[str] = None,
    pose_representation_type: Optional[str] = None,
    test_batch_size: int = 1,
    num_inference_steps: int = 50,
    scheduler_type: str = "ddim",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: Optional[int] = 2026,
    max_batches: Optional[int] = None,
    num_visualizations: int = 8,
    save_video: bool = True,
    save_npy: bool = True,
    render_after_inference: bool = True,
    torch_dtype: str = "float16",
    latent_dtype: Optional[str] = None,
    initial_noise_mode: str = "random",
    audio_condition_mode: str = "normal",
    relation_condition_mode: str = "normal",
    relation_temporal_features: Optional[Tuple[str, ...]] = None,
    relation_auxiliary_features: Optional[Tuple[str, ...]] = None,
    bow_t_velocity_temperature: float = 0.01,
    bow_endpoint_smoothing_window: int = 1,
    use_autocast: bool = True,
    audio_encoder_type: Optional[str] = None,
    audio_encoder_model_name: Optional[str] = None,
    audio_encoder_processor_name: Optional[str] = None,
    audio_encoder_layer: int = -1,
    audio_encoder_device: str = "cpu",
    audio_encoder_dtype: str = "float32",
    audio_encoder_max_seconds: Optional[float] = None,
    audio_encoder_normalize: bool = True,
    audio_encoder_local_files_only: bool = False,
    audio_encoder_trust_remote_code: bool = True,
    audio_feature_cache_dir: Optional[str] = None,
    audio_feature_cache_required: bool = False,
    render_fps: Optional[int] = None,
    camera_elev: float = 18.0,
    camera_azim: float = -70.0,
    include_instrument_context: bool = True,
    render_coordinate: str = "string_frame",
    use_known_lower_body: bool = False,
    known_lower_body_indices: Tuple[int, ...] = DEFAULT_KNOWN_LOWER_BODY_INDICES,
    mask_inactive_generation_joints: bool = True,
    inactive_generation_indices: Tuple[int, ...] = DEFAULT_INACTIVE_GENERATION_INDICES,
    visualization_mode: str = "body_front",
    visualization_flip_x: bool = False,
    visualization_flip_y: bool = False,
    visualization_flip_z: bool = False,
    report_title: str = "SPD audio+relations -> pose inference",
    syncviolinist_full_sequence_metrics: bool = False,
    continuous_prefix_frames: int = 0,
    apply_local_refiner: bool = True,
    apply_left_hand_refiner: bool = True,
    apply_bow_event_refiner: bool = True,
    elgar_pitch_cache_dir: str = "datasets/SPD_canonical/elgar_pitch_cache",
    elgar_pitch_device: str = "cuda:0",
    elgar_pitch_cache_required: bool = False,
):
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        device = "cpu"

    out_dir = Path(output_dir or Path(checkpoint_dir) / "pose_inference")
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = int(test_data.get("fps", 30))
    if fps != 30:
        raise ValueError("ELGAR original metrics require 30 fps.")
    elgar_accumulator = ElgarOriginalAccumulator()
    elgar_pitch_cache = PitchCache(elgar_pitch_cache_dir, elgar_pitch_device, elgar_pitch_cache_required)
    audio_sr = int(test_data.get("sr", 16000))
    render_fps = int(render_fps or fps)
    dtype = parse_torch_dtype(torch_dtype, device)
    sampling_dtype = dtype if latent_dtype is None else parse_torch_dtype(latent_dtype, device)
    audio_condition_mode = str(audio_condition_mode).lower()
    if audio_condition_mode not in ("normal", "zero"):
        raise ValueError("audio_condition_mode must be 'normal' or 'zero'.")
    relation_condition_mode = str(relation_condition_mode).lower()
    if relation_condition_mode not in ("normal", "zero"):
        raise ValueError("relation_condition_mode must be 'normal' or 'zero'.")
    render_coordinate = normalize_render_coordinate(render_coordinate)
    known_lower_body_indices = normalize_index_tuple(known_lower_body_indices)
    inactive_generation_indices = normalize_index_tuple(inactive_generation_indices)
    test_data = apply_audio_encoder_overrides(
        test_data,
        {
            "audio_encoder_type": audio_encoder_type,
            "audio_encoder_model_name": audio_encoder_model_name,
            "audio_encoder_processor_name": audio_encoder_processor_name,
            "audio_encoder_layer": audio_encoder_layer,
            "audio_encoder_device": audio_encoder_device,
            "audio_encoder_dtype": audio_encoder_dtype,
            "audio_encoder_max_seconds": audio_encoder_max_seconds,
            "audio_encoder_normalize": audio_encoder_normalize,
            "audio_encoder_local_files_only": audio_encoder_local_files_only,
            "audio_encoder_trust_remote_code": audio_encoder_trust_remote_code,
            "audio_feature_cache_dir": audio_feature_cache_dir,
            "audio_feature_cache_required": audio_feature_cache_required,
        },
    )
    test_data = disable_cuda_audio_encoder_workers(test_data)
    if str(device).startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        print(f"Using CUDA device: {torch.cuda.get_device_name(torch.device(device).index or 0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(torch.device(device).index or 0)}")
    print(
        f"Inference device: {device}, model_dtype: {dtype}, latent_dtype: {sampling_dtype}, "
        f"batch_size: {test_batch_size}, initial_noise: {initial_noise_mode}, "
        f"audio_condition: {audio_condition_mode}, relation_condition: {relation_condition_mode}"
    )
    print(f"Render coordinate: {render_coordinate}, visualization_mode: {visualization_mode}")
    print(f"Bow endpoint smoothing window: {int(bow_endpoint_smoothing_window)}")
    print(f"Use known lower body: {bool(use_known_lower_body)}, indices: {list(known_lower_body_indices)}")
    print(
        f"Mask inactive generation joints: {bool(mask_inactive_generation_joints)}, "
        f"indices: {list(inactive_generation_indices)}"
    )

    # The checkpoint's embedded train-only transform is authoritative. It must
    # remain usable even if the preparation JSON is absent on this machine.
    saved_config_path = Path(train_config) if train_config else Path(checkpoint_dir) / "config.yaml"
    if not saved_config_path.exists() and train_config is None:
        saved_config_path = Path(checkpoint_dir).parent / "config.yaml"
    if not saved_config_path.is_file():
        if train_config is not None:
            raise FileNotFoundError(f"Stage-2 training config not found: {saved_config_path}")
        raise FileNotFoundError(
            f"Missing the saved Stage-2 config.yaml for {checkpoint_dir}. "
            "Place it in the checkpoint directory or its parent, or pass train_config=/path/to/config.yaml."
        )
    saved_config = OmegaConf.to_container(OmegaConf.load(saved_config_path), resolve=True)
    saved_data = resolve_pose_representation(saved_config.get("train_data", {}))
    if pose_representation_type is not None:
        actual_type = (saved_data.get("pose_representation") or {}).get("type")
        if actual_type != pose_representation_type:
            raise ValueError("Requested test representation does not match the checkpoint.")
    if test_data.get("pose_representation") is not None or (
        test_data.get("pose_representation_path") and Path(test_data["pose_representation_path"]).is_file()
    ):
        require_same_representation(saved_data, test_data)
    elif test_data.get("pose_representation_path") and saved_data.get("pose_representation") is None:
        raise ValueError("A root/wrist test config cannot be used with a native checkpoint.")
    test_data = dict(test_data)
    test_data["pose_representation"] = saved_data.get("pose_representation")
    dataset = SPDCanonicalPoseDataset(**test_data)
    if dataset.pose_keys != ["coco_wholebody_string_frame", "bow_string_frame"]:
        raise ValueError("ELGAR original requires native SPD human + bow pose keys.")
    # Prepare the pitch cache before sampling. It is shared across seeds.
    for record in dataset.records:
        elgar_pitch_cache.get(str(record["id"]), record["_audio_path"], record["instrument"])
    if continuous_prefix_frames:
        if int(test_batch_size) != 1 or use_known_lower_body or mask_inactive_generation_joints:
            raise ValueError("Continuous native inference requires batch_size=1 and no GT/inactive-joint masking.")
        if dataset.pose_keys != ["syncviolinist_pose"]:
            raise ValueError("Continuous inference currently supports native SyncViolinist only.")
        dataset.windows = continuous_windows(dataset.records, dataset.sequence_length, int(continuous_prefix_frames))
    dataloader = DataLoader(
        dataset,
        batch_size=int(test_batch_size),
        shuffle=False,
        num_workers=int(test_data.get("num_workers", 4)),
        pin_memory=True,
        persistent_workers=int(test_data.get("num_workers", 4)) > 0,
    )
    sample = dataset[0]
    condition_dim = int(sample["condition"].shape[-1])
    dual_relation_include_mask = bool(test_data.get("include_relation_mask_in_condition", True))
    audio_condition_dim, relation_condition_dim = infer_dual_condition_dims(
        sample,
        include_relation_mask=dual_relation_include_mask,
        temporal_features=relation_temporal_features,
        auxiliary_features=relation_auxiliary_features,
        bow_t_velocity_temperature=bow_t_velocity_temperature,
    )
    pose_channels = int(sample["pose_grid"].shape[0])
    pose_grid_size = tuple(int(x) for x in sample["pose_grid"].shape[-2:])

    pipe = SPDPoseDiffusionPipeline.from_pretrained(
        checkpoint_dir=checkpoint_dir,
        condition_dim=condition_dim,
        audio_condition_dim=audio_condition_dim,
        relation_condition_dim=relation_condition_dim,
        config_path=str(saved_config_path),
        scheduler_type=scheduler_type,
        device=device,
        pose_channels=pose_channels,
        pose_size=pose_grid_size[-1],
        torch_dtype=dtype,
    )
    require_same_representation(pipe.config.get("train_data", {}), test_data)
    if continuous_prefix_frames and pipe.local_bow_refiner is not None:
        raise ValueError("Continuous inference does not support the SPD-only local bow refiner.")
    (out_dir / "continuous_inference.json").write_text(json.dumps({
        "prefix_frames": int(continuous_prefix_frames),
        "stride": dataset.sequence_length - int(continuous_prefix_frames),
        "context_source": "previous_prediction_only" if continuous_prefix_frames else "none",
        "clean_prefix": bool(pipe.config.get("pose_context_clean_prefix", False)),
        "evaluation_split": test_data.get("mode"),
    }, indent=2), encoding="utf-8")
    if dataset.pose_representation is not None:
        (out_dir / "pose_representation.json").write_text(
            json.dumps(dataset.pose_representation, indent=2), encoding="utf-8"
        )
    print(f"Loaded condition encoder class: {pipe.condition_encoder.__class__.__name__}")
    print(f"Loaded local bow refiner: {pipe.local_bow_refiner is not None}")
    print(f"Apply local bow refiner: {bool(apply_local_refiner) and pipe.local_bow_refiner is not None}")
    print(f"Local refinement branches: G={bool(apply_local_refiner)}, "
          f"L={bool(apply_local_refiner and apply_left_hand_refiner)}, "
          f"A={bool(apply_local_refiner and apply_bow_event_refiner)} (requested)")
    checkpoint_metrics_path = Path(checkpoint_dir) / "best_metrics.json"
    if checkpoint_metrics_path.exists():
        with checkpoint_metrics_path.open("r", encoding="utf-8") as file:
            checkpoint_metrics = json.load(file)
        print(
            "Loaded best checkpoint: "
            f"step={checkpoint_metrics.get('global_step')}, "
            f"metric={checkpoint_metrics.get('best_checkpoint_metric')}, "
            f"value={checkpoint_metrics.get('best_metric_value')}"
        )
    relation_gate = getattr(pipe.condition_encoder, "relation_gate", None)
    if relation_gate is not None:
        print(f"Learned relation gate: {float(relation_gate.detach().float().cpu()):.6f}")

    generator = make_generator(device, seed)
    pose_context = PredictedPoseContext(continuous_prefix_frames) if continuous_prefix_frames else None

    total_sums = {}
    total_counts = {}
    sums_by_sample = {}
    counts_by_sample = {}
    windows_by_sample = {}
    syncviolinist_sequence_metrics = SyncViolinistDTWAccumulator()
    sample_reports = []
    npy_reports = []
    pending_visualizations = []
    visualized = 0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inferring pose")):
            if max_batches is not None and batch_idx >= int(max_batches):
                break

            condition = batch["condition"].to(device=device, dtype=dtype, non_blocking=True)
            audio_condition = make_audio_condition(batch).to(device=device, dtype=dtype, non_blocking=True)
            if audio_condition_mode == "zero":
                audio_condition = torch.zeros_like(audio_condition)
            relation_condition = make_relation_condition(
                batch,
                include_relation_mask=dual_relation_include_mask,
                temporal_features=relation_temporal_features,
                auxiliary_features=relation_auxiliary_features,
                bow_t_velocity_temperature=bow_t_velocity_temperature,
            ).to(device=device, dtype=dtype, non_blocking=True)
            if relation_condition_mode == "zero":
                relation_condition = torch.zeros_like(relation_condition)
            target_grid = batch["pose_grid"].to(device=device, dtype=dtype, non_blocking=True)
            joint_counts = batch["joint_count"].to(device=device, non_blocking=True)
            padding_mask = make_pose_padding_mask(
                batch_size=target_grid.shape[0],
                channels=target_grid.shape[1],
                frames=target_grid.shape[2],
                height=target_grid.shape[3],
                width=target_grid.shape[4],
                joint_counts=joint_counts,
                device=torch.device(device),
                dtype=torch.float32,
            )
            if mask_inactive_generation_joints:
                padding_mask = mask_generation_indices(
                    padding_mask,
                    joint_counts=joint_counts,
                    indices=inactive_generation_indices,
                )
            known_pose, known_mask = (None, None)
            if pose_context is not None:
                known_pose, known_mask = pose_context.get(
                    str(batch["sample_id"][0]), int(batch["start_frame"][0]),
                    tuple(target_grid.shape), torch.device(device), sampling_dtype)
            pred_grid = pipe(
                condition=condition,
                pose_shape=tuple(target_grid.shape),
                audio_condition=audio_condition,
                relation_condition=relation_condition,
                num_inference_steps=int(num_inference_steps),
                generator=generator,
                pose_padding_mask=padding_mask,
                use_autocast=bool(use_autocast),
                latent_dtype=sampling_dtype,
                initial_noise_mode=initial_noise_mode,
                refiner_strings=batch["strings"],
                # The local correction is applied AFTER the requested fixed
                # smoothing below, preserving the corresponding base bow line.
                apply_local_refiner=False,
                known_pose_grid=known_pose,
                known_pose_mask=known_mask,
            ).pose_grid
            if pose_context is not None:
                pose_context.update(str(batch["sample_id"][0]), int(batch["start_frame"][0]),
                                    int(batch["end_frame"][0]), pred_grid)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()

            for i in range(pred_grid.shape[0]):
                joint_count = int(batch["joint_count"][i].item())
                pred_pose = unpack_pose_grid(pred_grid[i], joint_count, pose_grid_size)
                pred_pose = decode_pose(pred_pose, dataset.pose_representation)
                pred_pose = postprocess_sample_pose(
                    pred_pose, smoothing_window=bow_endpoint_smoothing_window,
                    local_refiner=pipe.local_bow_refiner if apply_local_refiner else None,
                    relation=batch["relation"][i:i+1], strings=batch["strings"][i:i+1],
                    device=device, relation_condition_mode=relation_condition_mode,
                    apply_left_hand_refiner=apply_left_hand_refiner,
                    apply_bow_event_refiner=apply_bow_event_refiner,
                )
                output_pred_grid = pose_to_grid_like(
                    pred_grid[i],
                    pred_pose,
                    pose_grid_size,
                )
                target_pose = batch["pose"][i].detach().cpu().numpy().astype(np.float32)
                pose_valid = batch["pose_valid"][i].detach().cpu().numpy().astype(bool)
                sample_id = _as_str(_batch_value(batch, "sample_id", i))
                start_frame = _as_int(_batch_value(batch, "start_frame", i))
                end_frame = _as_int(_batch_value(batch, "end_frame", i), start_frame + target_pose.shape[0])
                metrics = compute_metric_sums(pred_pose, target_pose, pose_valid)
                if not syncviolinist_full_sequence_metrics:
                    metrics.update(
                        compute_syncviolinist_metric_sums(
                            pred_pose,
                            target_pose,
                            pose_valid,
                            include_dtw=False,
                        )
                    )
                syncviolinist_sequence_metrics.add_window(
                    sample_id,
                    start_frame,
                    end_frame,
                    pred_pose,
                    target_pose,
                    pose_valid,
                )
                metric_item = {
                    "strings": batch["strings"][i].detach().cpu(),
                    "contact_target_point": batch["contact_target_point"][i].detach().cpu(),
                    "contact_valid": batch["contact_valid"][i].detach().cpu(),
                    "bow_valid": batch["bow_valid"][i].detach().cpu(),
                    "string_frame_scale": batch["string_frame_scale"][i].detach().cpu(),
                }
                metrics.update(
                    compute_audio_alignment_metric_sums(
                        pred_pose,
                        batch["onset"][i].detach().cpu(),
                        batch["pitch"][i].detach().cpu(),
                        strings=metric_item["strings"],
                        bow_valid=metric_item["bow_valid"],
                        pose_valid=pose_valid,
                        fps=fps,
                    )
                )
                metrics.update(
                    compute_contact_diagnostic_sums(
                        pred_pose,
                        metric_item["strings"],
                        metric_item["contact_target_point"],
                        metric_item["contact_valid"],
                        distance_scale=metric_item["string_frame_scale"],
                    )
                )
                add_metric_sums(total_sums, total_counts, metrics)
                add_metric_sums(
                    sums_by_sample.setdefault(sample_id, {}),
                    counts_by_sample.setdefault(sample_id, {}), metrics,
                )
                windows_by_sample[sample_id] = windows_by_sample.get(sample_id, 0) + 1
                sample_metrics = finalize_metrics(
                    {k: v[0] for k, v in metrics.items()},
                    {k: v[1] for k, v in metrics.items()},
                )
                add_audio_alignment_f1(sample_metrics)
                sample_metrics = _reported_metrics(sample_metrics)

                audio_path = _as_str(_batch_value(batch, "audio_path", i))
                string_frame_path = _as_str(_batch_value(batch, "string_frame_path", i))
                instrument = {0: "cello", 1: "violin"}[int(batch["instrument_id"][i])]
                elgar_accumulator.add_window(
                    sample_id, start_frame, end_frame, pred_pose, string_frame_path, audio_path, instrument
                )

                if save_npy:
                    npy_reports.append(
                        {
                            "sample_id": sample_id,
                            "start_frame": int(start_frame),
                            "end_frame": int(end_frame),
                            **save_prediction_npy_outputs(
                                out_dir=out_dir,
                                sample_id=sample_id,
                                start_frame=start_frame,
                                end_frame=end_frame,
                                pred_pose=pred_pose,
                                target_pose=target_pose,
                                pose_valid=pose_valid,
                                string_frame_path=string_frame_path,
                                render_coordinate=render_coordinate,
                                include_instrument_context=bool(include_instrument_context),
                                use_known_lower_body=bool(use_known_lower_body),
                                known_lower_body_indices=known_lower_body_indices,
                                pose_context_frames=0 if known_mask is None else int(known_mask.sum()),
                            ),
                        }
                    )

                if visualized < int(num_visualizations):
                    batch_item = {
                        "condition": batch["condition"][i].detach().cpu(),
                        "relation": batch["relation"][i].detach().cpu(),
                        "relation_mask": batch["relation_mask"][i].detach().cpu(),
                        "contact_valid": batch["contact_valid"][i].detach().cpu(),
                        "bow_valid": batch["bow_valid"][i].detach().cpu(),
                        "contact_target_point": batch["contact_target_point"][i].detach().cpu(),
                        "bow_target_point": batch["bow_target_point"][i].detach().cpu(),
                        "bow_target_direction": batch["bow_target_direction"][i].detach().cpu(),
                        "bow_t_on_bow": batch["bow_t_on_bow"][i].detach().cpu(),
                        "strings": batch["strings"][i].detach().cpu(),
                        "string_frame_scale": batch["string_frame_scale"][i].detach().cpu(),
                        "string_frame_path": string_frame_path,
                    }
                    payload = dict(
                        out_dir=out_dir,
                        sample_id=sample_id,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        audio_path=audio_path,
                        string_frame_path=string_frame_path,
                        pred_pose=pred_pose,
                        target_pose=target_pose,
                        pose_valid=pose_valid,
                        pred_grid=output_pred_grid,
                        target_grid=target_grid[i].detach().cpu().float(),
                        batch_item=batch_item,
                        sample_metrics=sample_metrics,
                        fps=fps,
                        audio_sr=audio_sr,
                        render_fps=render_fps,
                        elev=float(camera_elev),
                        azim=float(camera_azim),
                        save_video=bool(save_video),
                        include_instrument_context=bool(include_instrument_context),
                        render_coordinate=render_coordinate,
                        use_known_lower_body=bool(use_known_lower_body),
                        known_lower_body_indices=known_lower_body_indices,
                        visualization_mode=str(visualization_mode),
                        flip_x=bool(visualization_flip_x),
                        flip_y=bool(visualization_flip_y),
                        flip_z=bool(visualization_flip_z),
                    )
                    if render_after_inference:
                        pending_visualizations.append(payload)
                    else:
                        sample_reports.append(save_sample_outputs(**payload))
                    visualized += 1

    if pending_visualizations:
        print(f"Rendering {len(pending_visualizations)} visualizations on CPU after GPU inference...")
    for payload in pending_visualizations:
        sample_reports.append(save_sample_outputs(**payload))

    if syncviolinist_full_sequence_metrics:
        syncviolinist_metric_sums, syncviolinist_metrics_by_sample = (
            syncviolinist_sequence_metrics.compute_metric_sums(
                include_l1=True,
                include_dtw=True,
                include_jerk=True,
            )
        )
    else:
        # Preserve the original SPD evaluation behavior: L1/jerk are aggregated
        # per inference window above, while only official 128-frame DTW is
        # reconstructed over complete samples here.
        syncviolinist_metric_sums, syncviolinist_metrics_by_sample = (
            syncviolinist_sequence_metrics.compute_metric_sums()
        )
    add_metric_sums(total_sums, total_counts, syncviolinist_metric_sums)
    metrics_final = finalize_metrics(total_sums, total_counts)
    add_audio_alignment_f1(metrics_final)
    per_sample_metrics = finalize_sample_metrics(sums_by_sample, counts_by_sample)
    official_metrics, official_samples = elgar_accumulator.compute(elgar_pitch_cache)
    metrics_final.update(official_metrics)
    for sample_id, details in official_samples.items():
        per_sample_metrics[sample_id].update(details["metrics"])
    (out_dir / "elgar_original_by_sample.json").write_text(
        json.dumps(official_samples, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    metrics_final = _reported_metrics(metrics_final)
    instruments_by_sample = {str(record["id"]): record.get("instrument", "") for record in dataset.records}
    with (out_dir / "metrics_by_sample.json").open("w", encoding="utf-8") as f:
        json.dump({
            sample_id: {
                "instrument": instruments_by_sample.get(sample_id, ""),
                "num_windows": windows_by_sample[sample_id],
                "metrics": metrics,
            } for sample_id, metrics in per_sample_metrics.items()
        }, f, indent=2, sort_keys=True)
    with (out_dir / "metrics_protocol.json").open("w", encoding="utf-8") as f:
        json.dump({
            **protocol_metadata(),
            "bow_endpoint_smoothing_window": int(bow_endpoint_smoothing_window),
            "audio_condition_mode": audio_condition_mode,
            "relation_condition_mode": relation_condition_mode,
            "apply_local_refiner": bool(apply_local_refiner),
            "apply_left_hand_refiner": bool(apply_left_hand_refiner),
            "apply_bow_event_refiner": bool(apply_bow_event_refiner),
            "local_refiner_loaded": pipe.local_bow_refiner is not None,
            "local_refiner_applied": bool(apply_local_refiner) and pipe.local_bow_refiner is not None,
            "local_refiner_left_hand_applied": bool(apply_local_refiner and apply_left_hand_refiner)
                and hasattr(pipe.local_bow_refiner, "left_hand_head"),
            "local_refiner_bow_event_applied": bool(apply_local_refiner and apply_bow_event_refiner)
                and hasattr(pipe.local_bow_refiner, "bow_event_head"),
            "per_sample_aggregation": "ELGAR: full records, overlap removed; other metrics: window sums",
            "elgar_aggregation": "full records, overlap removed",
            "elgar_pitch_cache_dir": str(elgar_pitch_cache_dir),
        }, f, indent=2, sort_keys=True)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_final, f, indent=2, sort_keys=True)
    with (out_dir / "sample_reports.json").open("w", encoding="utf-8") as f:
        json.dump(sample_reports, f, indent=2, sort_keys=True)
    if save_npy:
        with (out_dir / "npy_reports.json").open("w", encoding="utf-8") as f:
            json.dump(npy_reports, f, indent=2, sort_keys=True)
    syncviolinist_metrics_by_sample = {
        sample_id: _reported_metrics(metrics) for sample_id, metrics in syncviolinist_metrics_by_sample.items()
    }
    with (out_dir / "syncviolinist_dtw_by_sample.json").open("w", encoding="utf-8") as f:
        json.dump(syncviolinist_metrics_by_sample, f, indent=2, sort_keys=True)
    with (out_dir / "syncviolinist_metrics_by_sample.json").open("w", encoding="utf-8") as f:
        json.dump(syncviolinist_metrics_by_sample, f, indent=2, sort_keys=True)

    with (out_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key in sorted(metrics_final):
            writer.writerow([key, metrics_final[key]])

    report_paths = write_metric_reports(
        out_dir,
        metrics_final,
        title=str(report_title),
        metadata={"checkpoint_dir": checkpoint_dir, "output_dir": out_dir,
                  "bowing_bridge_reference": ELGAR_BOWING_BRIDGE_REFERENCE,
                  "bow_endpoint_smoothing_window": int(bow_endpoint_smoothing_window)},
    )
    print_metric_summary(
        metrics_final,
        str(report_title),
        metadata={"checkpoint_dir": checkpoint_dir, "output_dir": out_dir,
                  "bowing_bridge_reference": ELGAR_BOWING_BRIDGE_REFERENCE,
                  "bow_endpoint_smoothing_window": int(bow_endpoint_smoothing_window)},
        report_path=report_paths["metrics_report_md"],
    )
    return metrics_final


def evaluate_stage2(
    checkpoint_dir: str,
    test_data: Dict,
    output_dir: Optional[str] = None,
    seed: Optional[int] = 2026,
    sampling_seeds: Optional[Iterable[int]] = None,
    num_sampling_seeds: int = 5,
    group_output_by_checkpoint: bool = True,
    **test_kwargs,
):
    # Accept the redundant setting in existing release configurations.
    protocol = test_kwargs.pop("elgar_metric_protocol", PROTOCOL)
    if protocol != PROTOCOL:
        raise ValueError(f"Stage-2 evaluation uses {PROTOCOL}; remove elgar_metric_protocol from the config.")
    seeds = resolve_sampling_seeds(seed, sampling_seeds, num_sampling_seeds)
    report_title = str(test_kwargs.get("report_title", "SPD audio+relations -> pose inference"))
    multi_seed_title = f"{report_title} | multi-seed"
    checkpoint_path = Path(checkpoint_dir)
    out_dir = Path(output_dir or checkpoint_path / "pose_inference")
    if output_dir is not None and bool(group_output_by_checkpoint) and out_dir.name != checkpoint_path.name:
        out_dir = out_dir / checkpoint_path.name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_seed: Dict[int, Dict[str, Optional[float]]] = {}

    for index, sampling_seed in enumerate(seeds, start=1):
        seed_out_dir = out_dir / f"seed-{sampling_seed}"
        print("=" * 72)
        print(f"Sampling seed {index}/{len(seeds)}: {sampling_seed}")
        print(f"seed_output_dir: {seed_out_dir}")
        print("=" * 72)
        metrics_by_seed[sampling_seed] = _run_single_seed(
            checkpoint_dir=checkpoint_dir,
            test_data=test_data,
            output_dir=str(seed_out_dir),
            seed=sampling_seed,
            **test_kwargs,
        )

    metadata = {
        "elgar_metric_protocol": PROTOCOL,
        "checkpoint_dir": checkpoint_dir,
        "output_dir": out_dir,
        "sampling_seeds": ", ".join(str(value) for value in seeds),
        "num_sampling_seeds": len(seeds),
        "audio_condition_mode": test_kwargs.get("audio_condition_mode", "normal"),
        "relation_condition_mode": test_kwargs.get("relation_condition_mode", "normal"),
        "bowing_bridge_reference": ELGAR_BOWING_BRIDGE_REFERENCE,
        "bow_endpoint_smoothing_window": int(test_kwargs.get("bow_endpoint_smoothing_window", 1)),
        "apply_local_refiner": bool(test_kwargs.get("apply_local_refiner", True)),
        "apply_left_hand_refiner": bool(test_kwargs.get("apply_left_hand_refiner", True)),
        "apply_bow_event_refiner": bool(test_kwargs.get("apply_bow_event_refiner", True)),
    }
    report_paths = write_seed_metric_reports(
        out_dir,
        metrics_by_seed,
        title=multi_seed_title,
        metadata=metadata,
    )
    write_sample_seed_metric_reports(out_dir, {
        sampling_seed: json.loads((out_dir / f"seed-{sampling_seed}" / "metrics_by_sample.json").read_text(encoding="utf-8"))
        for sampling_seed in seeds
    })
    print_seed_metric_summary(
        metrics_by_seed,
        multi_seed_title,
        metadata=metadata,
        report_path=report_paths["metrics_report_md"],
    )
    return metrics_by_seed


# End-to-end evaluation with frozen Stage-1 predictions.

def _write_stage1_metrics(output_dir: Path, metrics: Dict[str, float]) -> None:
    _write_json(output_dir / "metrics.json", metrics)
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key in sorted(metrics):
            writer.writerow([key, metrics[key]])


def _resolve_stage2_runner(backend: str):
    backend = str(backend).lower()
    if backend == "overlap_export":
        from scripts.export_test_motion import run_stage2_export
        return run_stage2_export, "overlap_export"
    if backend in {"unet", "diffusion", "test_audio2pose"}:
        return evaluate_stage2, "unet"
    if backend in {"transformer", "dit", "test_audio2pose_transformer"}:
        from scripts.test_audio2pose_transformer import main as transformer_main

        return transformer_main, "transformer"
    raise ValueError("stage2_backend must be 'unet', 'transformer' or 'overlap_export'.")


def main(
    checkpoint_dir: str,
    mapping_checkpoint_path: str,
    test_data: Dict[str, Any],
    output_dir: Optional[str] = None,
    mapping_latent_dim: Optional[int] = None,
    mapping_batch_size: Optional[int] = None,
    mapping_num_workers: Optional[int] = None,
    mapping_sequence_length: Optional[int] = None,
    mapping_window_stride: Optional[int] = None,
    mapping_stitching_mode: str = "mean",
    mapping_stitching_min_weight: float = 0.1,
    mapping_categorical_mode: str = "one_hot",
    mapping_clamp_continuous: bool = True,
    predicted_relation_mask_mode: str = "ones",
    predicted_validity_threshold: float = 0.5,
    predicted_relation_slices: Optional[Sequence[Sequence[int]]] = None,
    relation_dropped_slices: Optional[Sequence[Sequence[int]]] = None,
    hybrid_relation_mask_source: str = "matched",
    predicted_relation_smoothing_window: int = 0,
    predicted_relation_smoothing_slices: Optional[Sequence[Sequence[int]]] = None,
    predicted_scalar_key: Optional[str] = None,
    predicted_scalar_valid_key: Optional[str] = None,
    predicted_scalar_transform: str = "none",
    predicted_scalar_transform_window: int = 31,
    stage2_backend: str = "unet",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    stage2_output_dir: Optional[str] = None,
    predicted_relation_dataset_dir: Optional[str] = None,
    audio_event_validation_options: Optional[Dict[str, Any]] = None,
    trajectory_validation_options: Optional[Dict[str, Any]] = None,
    **stage2_kwargs: Any,
):
    """Run stage 1, materialise predictions, then run the unchanged stage 2."""
    if str(stage2_backend) == "cached_audio_event_trajectory_test":
        if audio_event_validation_options is not None:
            raise ValueError("Use the completed frozen test event report")
        from audio2pose.pipelines.audio_event_trajectory_test import run_cached_test
        return run_cached_test(checkpoint_dir,_plain(test_data),
            output_dir or str(Path(checkpoint_dir).parent/"cached_trajectory_test"),
            _plain(trajectory_validation_options),smoothing_window=stage2_kwargs.get("bow_endpoint_smoothing_window",5))
    if str(stage2_backend) == "cached_audio_event_trajectory":
        if audio_event_validation_options is not None:
            raise ValueError("Use the frozen completed val event report for cached trajectory validation")
        from audio2pose.pipelines.audio_event_trajectory import run_cached_validation
        return run_cached_validation(checkpoint_dir,_plain(test_data),
            output_dir or str(Path(checkpoint_dir).parent/"cached_trajectory_validation"),
            _plain(trajectory_validation_options),
            smoothing_window=stage2_kwargs.get("bow_endpoint_smoothing_window",5),device=device)
    if trajectory_validation_options is not None:
        raise ValueError("trajectory_validation_options requires stage2_backend: cached_audio_event_trajectory")
    if str(stage2_backend) == "audio_bow_events":
        from audio2pose.training.audio_bow_events import evaluate_checkpoint
        print("Audio-event proxy evaluation; this backend does not generate poses or report end-to-end BF1.")
        return evaluate_checkpoint(checkpoint_dir,_plain(test_data),
            output_dir or str(Path(checkpoint_dir).parent/"audio_event_validation"),device,
            batch_size=int(stage2_kwargs.get("test_batch_size",32)),reference_dir=predicted_relation_dataset_dir,
            fusion_options=_plain(audio_event_validation_options))
    if audio_event_validation_options is not None:
        raise ValueError("audio_event_validation_options requires stage2_backend: audio_bow_events")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available; stage 1 will run on CPU.")
        device = "cpu"
    device_obj = torch.device(device)
    test_data = _plain(test_data)
    if not isinstance(test_data, dict):
        raise TypeError("test_data must be a mapping.")

    root_output = Path(output_dir or Path(checkpoint_dir).parent / "end_to_end_inference")
    root_output.mkdir(parents=True, exist_ok=True)
    stage1_output = root_output / "stage1_relations"
    shadow_root = (Path(predicted_relation_dataset_dir) if predicted_relation_dataset_dir
                   else root_output / "predicted_relation_dataset")
    stage2_output = Path(stage2_output_dir) if stage2_output_dir else root_output / "stage2_motion"

    mapping_meta = torch.load(str(Path(mapping_checkpoint_path)), map_location="cpu")
    mapping_pretrained_audio_dim = int(mapping_meta.get("pretrained_audio_dim", 0))
    mapping_onset_dim = int(mapping_meta.get("onset_dim", 1))

    # SPDCanonicalPoseDataset applies the exact same instrument/exclusion and
    # validity filters as the existing stage-2 evaluator.  Old mapping
    # checkpoints only consume mel/onset/pitch, so pretrained-audio options are
    # stripped for speed.  New MERT-aware checkpoints keep those options.
    stage1_data = dict(test_data)
    if mapping_onset_dim > 1:
        # Dynamics-aware mapping checkpoints consume 1+4 onset channels
        # (impulse + onset_strength/spectral_flux/log_rms/hf_ratio). Enable
        # them for the Stage-1 source dataset only; stage 2's own dataset
        # keeps the original 1-channel onset so the UNet condition width is
        # identical to previous runs.
        stage1_data["use_bow_dynamics"] = True
    if mapping_sequence_length is not None:
        stage1_data["sequence_length"] = int(mapping_sequence_length)
    if mapping_window_stride is not None:
        stage1_data["window_stride"] = int(mapping_window_stride)
    pretrained_audio_keys = (
        "audio_encoder_type",
        "audio_encoder_model_name",
        "audio_encoder_processor_name",
        "audio_encoder_layer",
        "audio_encoder_device",
        "audio_encoder_dtype",
        "audio_encoder_max_seconds",
        "audio_encoder_normalize",
        "audio_encoder_local_files_only",
        "audio_encoder_trust_remote_code",
        "audio_feature_cache_dir",
        "audio_feature_cache_required",
    )
    if mapping_pretrained_audio_dim > 0:
        for key in pretrained_audio_keys:
            if key in stage2_kwargs and key not in stage1_data:
                stage1_data[key] = stage2_kwargs[key]
        if "audio_encoder_type" not in stage1_data:
            raise ValueError(
                "The mapping checkpoint was trained with pretrained_audio_dim > 0, "
                "but the end-to-end test_data does not define audio_encoder_type. "
                "Add audio_encoder_type and audio_feature_cache_dir to test_data."
            )
    else:
        for key in pretrained_audio_keys:
            stage1_data.pop(key, None)
    # The scalar auxiliary is predicted by Stage 1, not consumed as one of
    # its 23-D targets. It is appended only when materialising the Stage-2
    # predicted-relation cache below.
    stage1_data.pop("relation_extra_keys", None)
    stage1_data["include_relation_mask_in_condition"] = False
    stage1_data["num_workers"] = int(
        mapping_num_workers if mapping_num_workers is not None else stage1_data.get("num_workers", 4)
    )
    source_dataset = SPDCanonicalPoseDataset(**stage1_data)
    if len(source_dataset) == 0:
        raise RuntimeError("The end-to-end test dataset contains no valid pose/relation windows.")
    _verify_dynamics_cache(source_dataset, str(stage1_data.get("mode", "test")))
    mapping_loader = DataLoader(
        source_dataset,
        batch_size=int(mapping_batch_size or stage2_kwargs.get("test_batch_size", 1)),
        shuffle=False,
        num_workers=int(stage1_data["num_workers"]),
        pin_memory=device_obj.type == "cuda",
        persistent_workers=int(stage1_data["num_workers"]) > 0,
    )
    sample = source_dataset[0]
    mapping_model = load_mapping_model(
        checkpoint_path=mapping_checkpoint_path,
        mel_dim=int(sample["mel"].shape[-1]),
        relation_dim=int(sample["relation"].shape[-1]),
        latent_dim=mapping_latent_dim,
        device=device_obj,
    )
    relation_key = str(test_data.get("relation_key", "relation_feature_zero_filled"))
    relation_mask_key = str(test_data.get("relation_mask_key", "relation_feature_valid_mask"))
    stage1_metrics, accumulators = _stage1_metrics(
        mapping_model,
        mapping_loader,
        device_obj,
        relation_key=relation_key,
        mapping_stitching_mode=mapping_stitching_mode,
        mapping_stitching_min_weight=float(mapping_stitching_min_weight),
    )
    stage1_output.mkdir(parents=True, exist_ok=True)
    _write_stage1_metrics(stage1_output, stage1_metrics)
    print("Stage 1 metrics:", json.dumps(stage1_metrics, indent=2, sort_keys=True))

    build_predicted_relation_dataset(
        source_dataset=source_dataset,
        accumulators=accumulators,
        output_dir=shadow_root,
        relation_key=relation_key,
        relation_mask_key=relation_mask_key,
        categorical_mode=mapping_categorical_mode,
        clamp_continuous=bool(mapping_clamp_continuous),
        relation_mask_mode=predicted_relation_mask_mode,
        mapping_checkpoint_path=mapping_checkpoint_path,
        validity_threshold=predicted_validity_threshold,
        predicted_relation_slices=predicted_relation_slices,
        relation_dropped_slices=relation_dropped_slices,
        hybrid_relation_mask_source=hybrid_relation_mask_source,
        mapping_sequence_length=stage1_data.get("sequence_length"),
        mapping_window_stride=stage1_data.get("window_stride"),
        mapping_stitching_mode=mapping_stitching_mode,
        mapping_stitching_min_weight=float(mapping_stitching_min_weight),
        predicted_relation_smoothing_window=int(predicted_relation_smoothing_window),
        predicted_relation_smoothing_slices=predicted_relation_smoothing_slices,
        predicted_scalar_key=predicted_scalar_key,
        predicted_scalar_valid_key=predicted_scalar_valid_key,
        predicted_scalar_transform=str(predicted_scalar_transform),
        predicted_scalar_transform_window=int(predicted_scalar_transform_window),
    )

    stage2_data = dict(test_data)
    stage2_data["data_path"] = str(shadow_root)
    stage2_data["mode"] = "end_to_end"
    # The predicted cache already contains the appended scalar channel.
    stage2_data.pop("relation_extra_keys", None)
    # Keep the original evaluator's output structure but place it below this
    # run's root directory.  All other stage-2 arguments (seeds, rendering,
    # scheduler, dtype, etc.) pass through untouched.
    stage2_runner, resolved_backend = _resolve_stage2_runner(stage2_backend)
    stage2_kwargs = dict(stage2_kwargs)
    stage2_kwargs.pop("output_dir", None)
    stage2_kwargs["group_output_by_checkpoint"] = False
    # The transformer evaluator has a narrower signature than the UNet
    # evaluator (for example it has no MERT override or multi-seed options).
    # Drop only options it cannot consume when that backend is selected.
    if resolved_backend == "transformer":
        accepted = set(inspect.signature(stage2_runner).parameters)
        stage2_kwargs = {key: value for key, value in stage2_kwargs.items() if key in accepted}
    stage2_metrics = stage2_runner(
        checkpoint_dir=checkpoint_dir,
        test_data=stage2_data,
        output_dir=str(stage2_output),
        device=device,
        **stage2_kwargs,
    )

    result = {
        "mapping_checkpoint_path": str(Path(mapping_checkpoint_path).resolve()),
        "pose_checkpoint_dir": str(Path(checkpoint_dir).resolve()),
        "stage1_metrics": stage1_metrics,
        "stage1_output_dir": str(stage1_output.resolve()),
        "predicted_relation_dataset": str(shadow_root.resolve()),
        "stage2_output_dir": str(stage2_output.resolve()),
        "stage2_metrics": stage2_metrics,
        "relation_mask_mode": str(predicted_relation_mask_mode),
        "hybrid_relation_mask_source": str(hybrid_relation_mask_source),
        "predicted_relation_slices": _plain(predicted_relation_slices),
        "relation_dropped_slices": _plain(relation_dropped_slices),
        "validity_threshold": float(predicted_validity_threshold),
        "categorical_mode": str(mapping_categorical_mode),
        "mapping_sequence_length": int(stage1_data["sequence_length"]),
        "mapping_window_stride": int(stage1_data["window_stride"]),
        "mapping_stitching_mode": str(mapping_stitching_mode),
        "mapping_stitching_min_weight": float(mapping_stitching_min_weight),
        "predicted_relation_smoothing_window": int(predicted_relation_smoothing_window),
        "predicted_relation_smoothing_slices": _plain(predicted_relation_smoothing_slices),
        "stage2_backend": resolved_backend,
        "test_instruments": test_data.get("instruments"),
        "evaluated_samples": [
            {"sample_id": record["id"], "instrument": record.get("instrument", "")}
            for record in source_dataset.records
        ],
        "apply_local_refiner": bool(stage2_kwargs.get("apply_local_refiner", True)),
        "apply_left_hand_refiner": bool(stage2_kwargs.get("apply_left_hand_refiner", True)),
        "apply_bow_event_refiner": bool(stage2_kwargs.get("apply_bow_event_refiner", True)),
    }
    _write_json(root_output / "combined_results.json", result)
    print(f"End-to-end results written to: {root_output}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML configuration containing both mapping_checkpoint_path and stage-2 settings.",
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="OmegaConf dotlist overrides, e.g. apply_local_refiner=false output_dir=results/violin_no_refiner.",
    )
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))
    main(**config)
