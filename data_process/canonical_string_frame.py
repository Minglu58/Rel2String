#!/usr/bin/env python3
"""
Build a canonical string frame from 3D mocap instrument landmarks.

The frame is instrument-centric and intended to work for both violin and cello:

  origin: default nut center
  +x:     nut center -> bridge center
  +y:     low-pitched string side -> high-pitched string side
  +z:     fingerboard/string plane normal, chosen so x cross y = z
  scale:  default median nut-to-bridge distance

Input is an .npz file with at least one array shaped (T, N, 3), plus a JSON
config telling the script which point indices or names define the landmarks.
All configured point sets with shape (T, ..., 3) are transformed into this
canonical frame.

Example:
  python canonical_string_frame.py \
    --input sample_spd.npz \
    --config example_string_frame_config.json \
    --output sample_spd_string_frame.npz

Sanity check without data:
  python canonical_string_frame.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-8


class ConfigError(ValueError):
    """Raised when the frame config cannot be resolved against the data."""


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "landmarks" not in cfg:
        raise ConfigError("Config must contain a 'landmarks' object.")
    return cfg


def names_from_data(data: Mapping[str, np.ndarray], cfg: Mapping[str, Any]) -> Optional[List[str]]:
    if "names" in cfg:
        return [_as_text(x) for x in cfg["names"]]

    names_key = cfg.get("names_key")
    if names_key:
        if names_key not in data:
            raise ConfigError(f"names_key '{names_key}' is not present in input data.")
        names_arr = np.asarray(data[names_key]).reshape(-1)
        return [_as_text(x) for x in names_arr]

    return None


def choose_frames_key(data: Mapping[str, np.ndarray], cfg: Mapping[str, Any]) -> str:
    key = cfg.get("frames_key")
    if key:
        if key not in data:
            raise ConfigError(f"frames_key '{key}' is not present in input data.")
        return key

    candidates: List[str] = []
    for k, arr in data.items():
        a = np.asarray(arr)
        if a.ndim == 3 and a.shape[-1] == 3:
            candidates.append(k)

    if not candidates:
        raise ConfigError("Could not auto-detect a (T, N, 3) frame array. Set frames_key.")

    def score(k: str) -> Tuple[int, str]:
        lk = k.lower()
        preferred = any(s in lk for s in ("instrument", "violin", "cello", "guitar"))
        return (0 if preferred else 1, k)

    return sorted(candidates, key=score)[0]


def resolve_one_selector(selector: Any, names: Optional[Sequence[str]]) -> List[int]:
    if isinstance(selector, (int, np.integer)):
        return [int(selector)]

    if isinstance(selector, str):
        if selector.isdigit() or (selector.startswith("-") and selector[1:].isdigit()):
            return [int(selector)]
        if names is None:
            raise ConfigError(
                f"Landmark selector '{selector}' is a name, but no names/names_key were supplied."
            )
        matches = [i for i, name in enumerate(names) if name == selector]
        if not matches:
            raise ConfigError(f"Landmark name '{selector}' was not found in names.")
        return matches

    if isinstance(selector, Mapping):
        if "index" in selector:
            return resolve_one_selector(selector["index"], names)
        if "name" in selector:
            return resolve_one_selector(selector["name"], names)
        raise ConfigError(f"Unsupported selector object: {selector}")

    raise ConfigError(f"Unsupported landmark selector: {selector!r}")


def resolve_indices(selector: Any, names: Optional[Sequence[str]]) -> List[int]:
    if isinstance(selector, list):
        out: List[int] = []
        for item in selector:
            out.extend(resolve_indices(item, names))
        return out
    return resolve_one_selector(selector, names)


def nanmean_points(points: np.ndarray, axis: int) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanmean(points, axis=axis)


def group_center(
    frames: np.ndarray,
    landmarks: Mapping[str, Any],
    group_name: str,
    names: Optional[Sequence[str]],
) -> Tuple[np.ndarray, List[int]]:
    if group_name not in landmarks:
        raise ConfigError(f"landmarks.{group_name} is required.")
    idx = resolve_indices(landmarks[group_name], names)
    if not idx:
        raise ConfigError(f"landmarks.{group_name} resolved to no points.")
    return nanmean_points(frames[:, idx, :], axis=1), idx


def optional_group_points(
    frames: np.ndarray,
    landmarks: Mapping[str, Any],
    group_name: str,
    names: Optional[Sequence[str]],
) -> Tuple[Optional[np.ndarray], List[int]]:
    if group_name not in landmarks:
        return None, []
    idx = resolve_indices(landmarks[group_name], names)
    if not idx:
        return None, []
    return frames[:, idx, :], idx


def normalize(v: np.ndarray, eps: float = EPS) -> Tuple[np.ndarray, np.ndarray]:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    valid = np.isfinite(n[..., 0]) & (n[..., 0] > eps)
    out = np.zeros_like(v, dtype=np.float64)
    np.divide(v, n, out=out, where=n > eps)
    return out, valid


def fit_plane_normals(points: np.ndarray, fallback: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit per-frame plane normals by SVD and orient them toward fallback."""
    t_count = points.shape[0]
    normals = np.array(fallback, dtype=np.float64, copy=True)
    valid = np.zeros((t_count,), dtype=bool)

    for t in range(t_count):
        pts = np.asarray(points[t], dtype=np.float64)
        finite = np.isfinite(pts).all(axis=-1)
        pts = pts[finite]
        if pts.shape[0] < 3:
            continue
        centered = pts - pts.mean(axis=0, keepdims=True)
        if np.linalg.matrix_rank(centered) < 2:
            continue
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        fb = fallback[t]
        if np.isfinite(fb).all() and np.dot(normal, fb) < 0:
            normal = -normal
        normals[t] = normal
        valid[t] = True

    normals, valid_norm = normalize(normals)
    return normals, valid & valid_norm


def fill_invalid_frames(
    axes: np.ndarray,
    origin: np.ndarray,
    scale: np.ndarray,
    valid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use nearest previous valid frame, then nearest next valid frame, for continuity."""
    if valid.any():
        first_valid = int(np.flatnonzero(valid)[0])
    else:
        raise ConfigError("No valid frames could be built from the supplied landmarks.")

    for t in range(axes.shape[0]):
        if valid[t]:
            continue
        if t > 0:
            axes[t] = axes[t - 1]
            origin[t] = origin[t - 1]
            scale[t] = scale[t - 1]
        else:
            axes[t] = axes[first_valid]
            origin[t] = origin[first_valid]
            scale[t] = scale[first_valid]

    for t in range(axes.shape[0] - 2, -1, -1):
        if np.isfinite(axes[t]).all() and np.isfinite(origin[t]).all() and np.isfinite(scale[t]):
            continue
        axes[t] = axes[t + 1]
        origin[t] = origin[t + 1]
        scale[t] = scale[t + 1]

    return axes, origin, scale


def choose_origin(
    mode: Any,
    nut_center: np.ndarray,
    bridge_center: np.ndarray,
    frames: np.ndarray,
    landmarks: Mapping[str, Any],
    names: Optional[Sequence[str]],
) -> np.ndarray:
    if mode in (None, "nut"):
        return np.array(nut_center, copy=True)
    if mode == "bridge":
        return np.array(bridge_center, copy=True)
    if mode == "midpoint":
        return 0.5 * (nut_center + bridge_center)
    if isinstance(mode, Mapping):
        idx = resolve_indices(mode, names)
        return nanmean_points(frames[:, idx, :], axis=1)
    if isinstance(mode, str) and mode in landmarks:
        idx = resolve_indices(landmarks[mode], names)
        return nanmean_points(frames[:, idx, :], axis=1)
    raise ConfigError(
        "origin must be 'nut', 'bridge', 'midpoint', a landmark group name, or a selector object."
    )


def choose_scale(mode: Any, nut_to_bridge: np.ndarray, valid: np.ndarray) -> np.ndarray:
    t_count = nut_to_bridge.shape[0]
    if mode in (None, "median_nut_to_bridge"):
        if not valid.any():
            raise ConfigError("Cannot compute median scale because there are no valid frames.")
        med = float(np.nanmedian(nut_to_bridge[valid]))
        if not math.isfinite(med) or med <= EPS:
            raise ConfigError("Median nut-to-bridge scale is invalid.")
        return np.full((t_count,), med, dtype=np.float64)
    if mode == "nut_to_bridge":
        scale = np.array(nut_to_bridge, dtype=np.float64, copy=True)
        scale[~valid] = np.nan
        return scale
    if mode == "none":
        return np.ones((t_count,), dtype=np.float64)
    if isinstance(mode, (int, float)):
        if float(mode) <= EPS:
            raise ConfigError("Numeric scale must be positive.")
        return np.full((t_count,), float(mode), dtype=np.float64)
    raise ConfigError("scale must be 'median_nut_to_bridge', 'nut_to_bridge', 'none', or a number.")


def build_string_frame(
    frames: np.ndarray,
    cfg: Mapping[str, Any],
    names: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    if frames.ndim != 3 or frames.shape[-1] != 3:
        raise ConfigError(f"frames must be shaped (T, N, 3), got {frames.shape}.")
    frames = np.asarray(frames, dtype=np.float64)
    landmarks = cfg["landmarks"]

    nut_center, nut_idx = group_center(frames, landmarks, "nut", names)
    bridge_center, bridge_idx = group_center(frames, landmarks, "bridge", names)

    y_neg, y_neg_idx = group_center(frames, landmarks, "y_negative", names)
    y_pos, y_pos_idx = group_center(frames, landmarks, "y_positive", names)

    x_raw = bridge_center - nut_center
    y_hint_raw = y_pos - y_neg
    x_axis, valid_x = normalize(x_raw)
    y_hint, valid_y = normalize(y_hint_raw)

    z_hint_raw = np.cross(x_axis, y_hint)
    z_hint, valid_z_hint = normalize(z_hint_raw)

    plane_points, plane_idx = optional_group_points(frames, landmarks, "plane", names)
    if plane_points is not None:
        z_raw, valid_plane = fit_plane_normals(plane_points, z_hint)
        # Force z to be perpendicular to x while staying as close as possible to the fitted plane normal.
        z_projected = z_raw - np.sum(z_raw * x_axis, axis=-1, keepdims=True) * x_axis
        z_axis, valid_z = normalize(z_projected)
        valid_z = valid_z & valid_plane
    else:
        z_axis = z_hint
        valid_z = valid_z_hint

    y_axis_raw = np.cross(z_axis, x_axis)
    y_axis, valid_y_axis = normalize(y_axis_raw)

    # Keep +y aligned with the configured low-to-high string direction.
    flip = np.sum(y_axis * y_hint, axis=-1) < 0
    y_axis[flip] *= -1.0
    z_axis[flip] *= -1.0

    # Recompute z so the basis is exactly right-handed after all projections/flips.
    z_axis, valid_z_recomputed = normalize(np.cross(x_axis, y_axis))

    axes = np.stack([x_axis, y_axis, z_axis], axis=-1)  # columns are world-space axes
    nut_to_bridge = np.linalg.norm(x_raw, axis=-1)

    valid = (
        valid_x
        & valid_y
        & valid_z
        & valid_y_axis
        & valid_z_recomputed
        & np.isfinite(axes).all(axis=(1, 2))
        & np.isfinite(nut_center).all(axis=-1)
        & np.isfinite(bridge_center).all(axis=-1)
        & (nut_to_bridge > EPS)
    )

    origin = choose_origin(
        cfg.get("origin", "nut"),
        nut_center,
        bridge_center,
        frames,
        landmarks,
        names,
    )
    scale = choose_scale(cfg.get("scale", "median_nut_to_bridge"), nut_to_bridge, valid)

    axes, origin, scale = fill_invalid_frames(axes, origin, scale, valid)

    return {
        "origin": origin,
        "axes": axes,
        "scale": scale,
        "valid_frame": valid,
        "nut_center": nut_center,
        "bridge_center": bridge_center,
        "nut_to_bridge": nut_to_bridge,
        "landmark_indices_json": np.array(
            json.dumps(
                {
                    "nut": nut_idx,
                    "bridge": bridge_idx,
                    "y_negative": y_neg_idx,
                    "y_positive": y_pos_idx,
                    "plane": plane_idx,
                },
                sort_keys=True,
            )
        ),
    }


def world_to_string_frame(points: np.ndarray, origin: np.ndarray, axes: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Transform points shaped (T, ..., 3) from world/mocap coordinates to string frame."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] != origin.shape[0] or points.shape[-1] != 3:
        raise ValueError(f"points must be shaped (T, ..., 3); got {points.shape}.")
    broadcast_shape = (origin.shape[0],) + (1,) * (points.ndim - 2) + (3,)
    centered = points - origin.reshape(broadcast_shape)
    local = np.einsum("t...i,tij->t...j", centered, axes)
    scale_shape = (scale.shape[0],) + (1,) * (points.ndim - 1)
    return local / scale.reshape(scale_shape)


def string_frame_to_world(local: np.ndarray, origin: np.ndarray, axes: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Inverse transform for points shaped (T, ..., 3)."""
    local = np.asarray(local, dtype=np.float64)
    if local.shape[0] != origin.shape[0] or local.shape[-1] != 3:
        raise ValueError(f"local must be shaped (T, ..., 3); got {local.shape}.")
    scale_shape = (scale.shape[0],) + (1,) * (local.ndim - 1)
    scaled = local * scale.reshape(scale_shape)
    world = np.einsum("t...j,tij->t...i", scaled, axes)
    broadcast_shape = (origin.shape[0],) + (1,) * (local.ndim - 2) + (3,)
    return world + origin.reshape(broadcast_shape)


def point_sets_from_config(
    data: Mapping[str, np.ndarray],
    cfg: Mapping[str, Any],
    t_count: int,
) -> Dict[str, str]:
    if "point_sets" in cfg:
        return {str(out_name): str(in_key) for out_name, in_key in cfg["point_sets"].items()}

    point_sets: Dict[str, str] = {}
    for key, arr in data.items():
        a = np.asarray(arr)
        if a.ndim >= 3 and a.shape[0] == t_count and a.shape[-1] == 3:
            point_sets[key] = key
    return point_sets


def run_npz(input_path: Path, config_path: Path, output_path: Path, print_summary: bool = False) -> None:
    cfg = load_config(config_path)
    with np.load(input_path, allow_pickle=False) as npz:
        data = {k: npz[k] for k in npz.files}

    frames_key = choose_frames_key(data, cfg)
    names = names_from_data(data, cfg)
    frame_info = build_string_frame(data[frames_key], cfg, names)
    t_count = frame_info["origin"].shape[0]

    out: Dict[str, np.ndarray] = {}
    for key, value in frame_info.items():
        out[f"string_frame_{key}"] = value

    for out_name, in_key in point_sets_from_config(data, cfg, t_count).items():
        if in_key not in data:
            raise ConfigError(f"point_sets.{out_name} refers to missing key '{in_key}'.")
        points = np.asarray(data[in_key])
        out[f"{out_name}_string_frame"] = world_to_string_frame(
            points,
            frame_info["origin"],
            frame_info["axes"],
            frame_info["scale"],
        )

    metadata = {
        "input": str(input_path),
        "config": str(config_path),
        "frames_key": frames_key,
        "origin": cfg.get("origin", "nut"),
        "scale": cfg.get("scale", "median_nut_to_bridge"),
        "point_sets": point_sets_from_config(data, cfg, t_count),
        "axis_convention": {
            "x": "nut_center_to_bridge_center",
            "y": "landmarks.y_negative_to_landmarks.y_positive",
            "z": "right_handed_x_cross_y",
        },
    }
    out["string_frame_metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **out)

    if print_summary:
        valid = frame_info["valid_frame"]
        print(f"frames_key: {frames_key}")
        print(f"frames: {t_count}")
        print(f"valid frames: {int(valid.sum())}/{t_count}")
        print(f"output: {output_path}")
        for out_name, in_key in point_sets_from_config(data, cfg, t_count).items():
            print(f"transformed: {in_key} -> {out_name}_string_frame")


def rotation_matrix_z(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def run_self_test() -> None:
    names = [
        "nut_low",
        "nut_high",
        "bridge_low",
        "bridge_high",
        "fingerboard_low_mid",
        "fingerboard_high_mid",
    ]
    canonical = np.array(
        [
            [0.0, -0.1, 0.0],
            [0.0, 0.1, 0.0],
            [1.0, -0.1, 0.0],
            [1.0, 0.1, 0.0],
            [0.45, -0.08, 0.0],
            [0.45, 0.08, 0.0],
        ],
        dtype=np.float64,
    )

    t_count = 8
    frames = []
    for t in range(t_count):
        r = rotation_matrix_z(0.1 * t)
        s = 0.72
        tr = np.array([1.0 + 0.02 * t, -0.4, 0.25], dtype=np.float64)
        frames.append(canonical @ r.T * s + tr)
    frames = np.stack(frames, axis=0)

    cfg = {
        "names": names,
        "origin": "nut",
        "scale": "median_nut_to_bridge",
        "landmarks": {
            "nut": ["nut_low", "nut_high"],
            "bridge": ["bridge_low", "bridge_high"],
            "y_negative": ["nut_low", "bridge_low"],
            "y_positive": ["nut_high", "bridge_high"],
            "plane": names,
        },
    }
    info = build_string_frame(frames, cfg, names)
    local = world_to_string_frame(frames, info["origin"], info["axes"], info["scale"])
    restored = string_frame_to_world(local, info["origin"], info["axes"], info["scale"])

    nut_center = local[:, [0, 1], :].mean(axis=1)
    bridge_center = local[:, [2, 3], :].mean(axis=1)
    max_restore_err = np.max(np.abs(restored - frames))

    assert np.allclose(nut_center, np.array([0.0, 0.0, 0.0]), atol=1e-7)
    assert np.allclose(bridge_center[:, 0], 1.0, atol=1e-7)
    assert np.all(bridge_center[:, 1] < 1e-7)
    assert np.all(local[:, 1, 1] > local[:, 0, 1])
    assert max_restore_err < 1e-9

    print("self-test passed")
    print(f"max inverse-transform error: {max_restore_err:.3e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", type=Path, help="Input .npz mocap data.")
    parser.add_argument("--config", "-c", type=Path, help="JSON config with landmark groups.")
    parser.add_argument("--output", "-o", type=Path, help="Output .npz path.")
    parser.add_argument("--print-summary", action="store_true", help="Print frame-building summary.")
    parser.add_argument("--self-test", action="store_true", help="Run a synthetic geometry test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    if args.input is None or args.config is None:
        raise SystemExit("--input and --config are required unless --self-test is used.")

    output = args.output
    if output is None:
        output = args.input.with_name(args.input.stem + "_string_frame.npz")

    run_npz(args.input, args.config, output, print_summary=args.print_summary)


if __name__ == "__main__":
    main()
