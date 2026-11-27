#!/usr/bin/env python3
"""
Build a unified canonical string frame for raw SPD cello/violin mocap files.

This script expects one raw SPD .npy array shaped:
  cello:  (T, 155, 3)
  violin: (T, 153, 3)

It does not use cello body/endpin indices as the frame definition. Instead, it
uses the explicit four string endpoints available in both instruments after the
violin index shift:

  +x: nut/top side -> bridge/bottom side
  +y: string 4 low-pitched side -> string 1 high-pitched side
  +z: right-handed normal to the string/fingerboard plane

The output local coordinates are dimensionless if scale is enabled.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from canonical_string_frame import build_string_frame, world_to_string_frame  # noqa: E402


EPS = 1e-8


@dataclass(frozen=True)
class SpdIndexMap:
    instrument: str
    n_keypoints: int
    coco_wholebody: Tuple[int, int]
    instrument_body: Tuple[int, int]
    bow: Tuple[int, int]
    strings_low_to_high: Mapping[int, Tuple[int, int]]
    contact_point: int
    used_finger: Tuple[int, int, int, int]


CELLO = SpdIndexMap(
    instrument="cello",
    n_keypoints=155,
    coco_wholebody=(0, 133),
    instrument_body=(133, 140),
    bow=(140, 142),
    strings_low_to_high={
        4: (142, 143),
        3: (144, 145),
        2: (146, 147),
        1: (148, 149),
    },
    contact_point=150,
    used_finger=(151, 152, 153, 154),
)


VIOLIN = SpdIndexMap(
    instrument="violin",
    n_keypoints=153,
    coco_wholebody=(0, 133),
    instrument_body=(133, 138),
    bow=(138, 140),
    strings_low_to_high={
        4: (140, 141),
        3: (142, 143),
        2: (144, 145),
        1: (146, 147),
    },
    contact_point=148,
    used_finger=(149, 150, 151, 152),
)


def get_index_map(instrument: str, n_keypoints: int) -> SpdIndexMap:
    if instrument == "auto":
        if n_keypoints == CELLO.n_keypoints:
            return CELLO
        if n_keypoints == VIOLIN.n_keypoints:
            return VIOLIN
        raise ValueError(f"Cannot infer instrument from {n_keypoints} keypoints.")

    if instrument == "cello":
        idx = CELLO
    elif instrument == "violin":
        idx = VIOLIN
    else:
        raise ValueError("instrument must be auto, cello, or violin.")

    if n_keypoints != idx.n_keypoints:
        raise ValueError(
            f"{instrument} expects {idx.n_keypoints} keypoints, got {n_keypoints}. "
            "Use --instrument auto or check that the raw file is correct."
        )
    return idx


def string_endpoint_indices_low_to_high(idx: SpdIndexMap) -> np.ndarray:
    pairs = [idx.strings_low_to_high[s] for s in (4, 3, 2, 1)]
    return np.array(pairs, dtype=np.int64)


def frame_config(idx: SpdIndexMap, origin: str, scale: str) -> Dict[str, object]:
    pairs = string_endpoint_indices_low_to_high(idx)
    top = pairs[:, 0].tolist()
    bottom = pairs[:, 1].tolist()
    string4 = list(idx.strings_low_to_high[4])
    string1 = list(idx.strings_low_to_high[1])
    all_string_points = pairs.reshape(-1).tolist()

    return {
        "origin": origin,
        "scale": scale,
        "landmarks": {
            "nut": top,
            "bridge": bottom,
            "y_negative": string4,
            "y_positive": string1,
            "plane": all_string_points,
        },
    }


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    out = np.zeros_like(vectors, dtype=np.float64)
    np.divide(vectors, norm, out=out, where=norm > EPS)
    out[~np.isfinite(out)] = np.nan
    return out


def contact_features(
    contact_local: np.ndarray,
    strings_local: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute nearest string and nut-to-bridge ratio.

    strings_local is shaped (T, 4, 2, 3), ordered low to high:
      0 string_4, 1 string_3, 2 string_2, 3 string_1.
    """
    t_count = strings_local.shape[0]
    active_low_to_high = np.full((t_count,), -1, dtype=np.int64)
    active_elgar = np.full((t_count,), -1, dtype=np.int64)
    u_nut_to_bridge = np.full((t_count,), np.nan, dtype=np.float64)
    vibrating_length = np.full((t_count,), np.nan, dtype=np.float64)
    distance = np.full((t_count,), np.nan, dtype=np.float64)

    starts = strings_local[:, :, 0, :]
    ends = strings_local[:, :, 1, :]
    line = ends - starts
    line_len2 = np.sum(line * line, axis=-1)

    for t in range(t_count):
        cp = contact_local[t]
        if not np.isfinite(cp).all():
            continue
        valid = np.isfinite(starts[t]).all(axis=-1) & np.isfinite(ends[t]).all(axis=-1)
        valid &= line_len2[t] > EPS
        if not valid.any():
            continue

        point_vec = cp[None, :] - starts[t]
        u = np.sum(point_vec * line[t], axis=-1) / line_len2[t]
        u = np.clip(u, 0.0, 1.0)
        projection = starts[t] + u[:, None] * line[t]
        dist = np.linalg.norm(cp[None, :] - projection, axis=-1)
        dist[~valid] = np.inf

        low_to_high_idx = int(np.argmin(dist))
        active_low_to_high[t] = low_to_high_idx
        active_elgar[t] = 3 - low_to_high_idx
        u_nut_to_bridge[t] = u[low_to_high_idx]
        vibrating_length[t] = 1.0 - u[low_to_high_idx]
        distance[t] = dist[low_to_high_idx]

    return {
        "contact_string_low_to_high": active_low_to_high,
        "contact_string_elgar": active_elgar,
        "contact_u_nut_to_bridge": u_nut_to_bridge,
        "contact_vibrating_length": vibrating_length,
        "contact_to_string_distance": distance,
    }


def build_output(
    data: np.ndarray,
    idx: SpdIndexMap,
    origin: str,
    scale: str,
    input_path: Path,
) -> Dict[str, np.ndarray]:
    cfg = frame_config(idx, origin=origin, scale=scale)
    info = build_string_frame(data, cfg)

    pairs = string_endpoint_indices_low_to_high(idx)
    coco_start, coco_end = idx.coco_wholebody
    inst_start, inst_end = idx.instrument_body
    bow_start, bow_end = idx.bow

    all_local = world_to_string_frame(data, info["origin"], info["axes"], info["scale"])
    strings_world = data[:, pairs.reshape(-1), :].reshape(data.shape[0], 4, 2, 3)
    strings_local = world_to_string_frame(strings_world, info["origin"], info["axes"], info["scale"])
    contact_world = data[:, idx.contact_point, :]
    contact_local = world_to_string_frame(contact_world, info["origin"], info["axes"], info["scale"])
    bow_world = data[:, bow_start:bow_end, :]
    bow_local = world_to_string_frame(bow_world, info["origin"], info["axes"], info["scale"])
    bow_direction = normalize_vectors(bow_local[:, 1, :] - bow_local[:, 0, :])

    out: Dict[str, np.ndarray] = {
        "keypoints_string_frame": all_local,
        "coco_wholebody_string_frame": all_local[:, coco_start:coco_end, :],
        "instrument_body_string_frame": all_local[:, inst_start:inst_end, :],
        "bow_string_frame": bow_local,
        "bow_direction_string_frame": bow_direction,
        "strings_string_frame": strings_local,
        "contact_point_string_frame": contact_local,
        "used_finger_string_frame": all_local[:, list(idx.used_finger), :],
        "string_frame_origin_world": info["origin"],
        "string_frame_axes_world": info["axes"],
        "string_frame_scale_world": info["scale"],
        "string_frame_valid_frame": info["valid_frame"],
        "string_frame_nut_center_world": info["nut_center"],
        "string_frame_bridge_center_world": info["bridge_center"],
        "string_frame_nut_to_bridge_world": info["nut_to_bridge"],
    }
    out.update(contact_features(contact_local, strings_local))

    metadata = {
        "input": str(input_path),
        "instrument": idx.instrument,
        "input_shape": list(data.shape),
        "origin": origin,
        "scale": scale,
        "unit": "local coordinates are divided by string_frame_scale_world",
        "string_order": "strings_string_frame axis 1 is [string_4, string_3, string_2, string_1]",
        "endpoint_order": "strings_string_frame axis 2 is [top_or_nut, bottom_or_bridge]",
        "axis_convention": {
            "x": "mean string top/nut endpoints -> mean string bottom/bridge endpoints",
            "y": "string_4 low-pitched side -> string_1 high-pitched side",
            "z": "right-handed x cross y, fitted with all string endpoints as plane hint",
        },
        "indices": {
            "coco_wholebody": list(idx.coco_wholebody),
            "instrument_body": list(idx.instrument_body),
            "bow": list(idx.bow),
            "strings_low_to_high": {str(k): list(v) for k, v in idx.strings_low_to_high.items()},
            "contact_point": idx.contact_point,
            "used_finger": list(idx.used_finger),
        },
    }
    out["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    return out


def print_summary(out: Mapping[str, np.ndarray]) -> None:
    valid = out["string_frame_valid_frame"]
    strings = out["strings_string_frame"]
    contact = out["contact_point_string_frame"]
    bow_dir = out["bow_direction_string_frame"]

    top_x = np.nanmean(strings[:, :, 0, 0])
    bottom_x = np.nanmean(strings[:, :, 1, 0])
    z_abs = np.nanmean(np.abs(strings[:, :, :, 2]))
    contact_valid = np.isfinite(contact).all(axis=-1)
    bow_valid = np.isfinite(bow_dir).all(axis=-1)

    print(f"valid frames: {int(valid.sum())}/{len(valid)}")
    print(f"mean string top x: {top_x:.6f}")
    print(f"mean string bottom x: {bottom_x:.6f}")
    print(f"mean abs string z: {z_abs:.6f}")
    print(f"finite contact frames: {int(contact_valid.sum())}/{len(contact_valid)}")
    print(f"finite bow direction frames: {int(bow_valid.sum())}/{len(bow_valid)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", type=Path, required=True, help="Raw SPD .npy file.")
    parser.add_argument(
        "--instrument",
        choices=("auto", "cello", "violin"),
        default="auto",
        help="Use auto to infer from keypoint count.",
    )
    parser.add_argument(
        "--origin",
        choices=("nut", "bridge", "midpoint"),
        default="nut",
        help="Frame origin.",
    )
    parser.add_argument(
        "--scale",
        choices=("median_nut_to_bridge", "nut_to_bridge", "none"),
        default="median_nut_to_bridge",
        help="Coordinate scale normalization.",
    )
    parser.add_argument("--output", "-o", type=Path, help="Output .npz path.")
    parser.add_argument("--print-summary", action="store_true", help="Print geometric diagnostics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.input, allow_pickle=False)
    if data.ndim != 3 or data.shape[-1] != 3:
        raise ValueError(f"Expected raw SPD data shaped (T, N, 3), got {data.shape}.")

    idx = get_index_map(args.instrument, data.shape[1])
    out = build_output(data, idx, origin=args.origin, scale=args.scale, input_path=args.input)

    output = args.output
    if output is None:
        output = args.input.with_name(args.input.stem + f"_{idx.instrument}_string_frame.npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **out)

    print(f"instrument: {idx.instrument}")
    print(f"input shape: {data.shape}")
    print(f"output: {output}")
    if args.print_summary:
        print_summary(out)


if __name__ == "__main__":
    main()
