#!/usr/bin/env python3
"""
Extract cross-instrument playable-relation features from SPD string-frame data.

Input:
  The .npz written by spd_string_frame.py.

Output:
  A compact .npz with features such as:
    - left-hand contact string and normalized position
    - bow nearest string and normalized position
    - bow-to-string distance
    - bow/string angle
    - whether contact and bow select the same string

String order is always low to high:
  [string_4, string_3, string_2, string_1]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np


EPS = 1e-8


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    out = np.full_like(v, np.nan, dtype=np.float64)
    np.divide(v, norm, out=out, where=norm > EPS)
    return out


def strict_segment_midpoint(segment: np.ndarray) -> np.ndarray:
    """Return a midpoint only when both endpoints are finite.

    ``np.nanmean`` silently turns a one-endpoint segment into a seemingly
    valid midpoint.  That creates artificial jumps in midpoint velocity and
    lets incomplete bow annotations enter the relation target.  A midpoint is
    only geometrically defined when the full segment is present.
    """
    segment = np.asarray(segment)
    if segment.ndim != 3 or segment.shape[1:] != (2, 3):
        raise ValueError(
            f"Expected segment shape (T, 2, 3), got {segment.shape}."
        )
    valid = np.isfinite(segment).all(axis=(1, 2))
    midpoint = np.full((segment.shape[0], 3), np.nan, dtype=segment.dtype)
    midpoint[valid] = segment[valid].mean(axis=1)
    return midpoint


def bridge_anchor_from_strings(strings: np.ndarray) -> np.ndarray:
    """Keep the mean bridge reference used by trained relation scalars.

    Evaluation uses ELGAR's string-1 bridge endpoint instead. Changing this
    feature definition would invalidate existing mapping targets and caches.
    """
    strings = np.asarray(strings)
    if strings.ndim != 4 or strings.shape[2:] != (2, 3):
        raise ValueError(
            f"Expected strings shape (T, S, 2, 3), got {strings.shape}."
        )
    bridge_points = strings[:, :, 1, :]
    valid = np.isfinite(bridge_points).all(axis=-1)
    bridge_sum = np.where(valid[:, :, None], bridge_points, 0.0).sum(axis=1)
    bridge_count = valid.sum(axis=1, keepdims=True)
    return np.divide(
        bridge_sum,
        np.maximum(bridge_count, 1),
        out=np.full((strings.shape[0], 3), np.nan, dtype=np.float64),
        where=bridge_count > 0,
    )


def one_hot(index: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((index.shape[0], n_classes), dtype=np.float64)
    valid = (index >= 0) & (index < n_classes)
    rows = np.flatnonzero(valid)
    out[rows, index[rows]] = 1.0
    return out


def point_to_segments(point: np.ndarray, segments: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Nearest segment features.

    point:    (T, 3)
    segments: (T, S, 2, 3)

    Returns nearest segment index, segment parameter u, closest point, and distance.
    u=0 at top/nut, u=1 at bottom/bridge.
    """
    starts = segments[:, :, 0, :]
    ends = segments[:, :, 1, :]
    line = ends - starts
    line_len2 = np.sum(line * line, axis=-1)

    point_vec = point[:, None, :] - starts
    u = np.sum(point_vec * line, axis=-1) / np.where(line_len2 > EPS, line_len2, np.nan)
    u = np.clip(u, 0.0, 1.0)
    closest = starts + u[:, :, None] * line
    distance_all = np.linalg.norm(point[:, None, :] - closest, axis=-1)

    valid_point = np.isfinite(point).all(axis=-1)
    valid_segment = np.isfinite(segments).all(axis=(2, 3)) & (line_len2 > EPS)
    valid = valid_point[:, None] & valid_segment
    distance_for_min = np.where(valid, distance_all, np.inf)

    nearest = np.argmin(distance_for_min, axis=1).astype(np.int64)
    has_match = np.isfinite(distance_for_min[np.arange(len(point)), nearest])
    nearest[~has_match] = -1

    u_nearest = np.full((len(point),), np.nan, dtype=np.float64)
    dist_nearest = np.full((len(point),), np.nan, dtype=np.float64)
    closest_nearest = np.full((len(point), 3), np.nan, dtype=np.float64)
    rows = np.flatnonzero(has_match)
    u_nearest[rows] = u[rows, nearest[rows]]
    dist_nearest[rows] = distance_all[rows, nearest[rows]]
    closest_nearest[rows] = closest[rows, nearest[rows]]

    return {
        "nearest_index": nearest,
        "u": u_nearest,
        "distance": dist_nearest,
        "closest_point": closest_nearest,
        "distance_all": np.where(valid, distance_all, np.nan),
        "u_all": np.where(valid, u, np.nan),
    }


def _clip01(x: float) -> float:
    return min(1.0, max(0.0, float(x)))


def closest_segment_pair(p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> Tuple[float, float, float]:
    """
    Return (s, t, distance) for closest points on p0->p1 and q0->q1.

    s is the bow parameter, t is the string parameter.
    """
    if not (np.isfinite(p0).all() and np.isfinite(p1).all() and np.isfinite(q0).all() and np.isfinite(q1).all()):
        return np.nan, np.nan, np.nan

    u = p1 - p0
    v = q1 - q0
    a = float(np.dot(u, u))
    c = float(np.dot(v, v))
    if a <= EPS or c <= EPS:
        return np.nan, np.nan, np.nan

    candidates: List[Tuple[float, float]] = []

    # Interior candidate for the two infinite lines, kept only if it lies on both segments.
    w = p0 - q0
    b = float(np.dot(u, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    denom = a * c - b * b
    if abs(denom) > EPS:
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom
        if 0.0 <= s <= 1.0 and 0.0 <= t <= 1.0:
            candidates.append((float(s), float(t)))

    # Segment-endpoint projections. These make the routine robust after clamping.
    for s_fixed in (0.0, 1.0):
        p = p0 + s_fixed * u
        t = np.dot(p - q0, v) / c
        candidates.append((s_fixed, _clip01(t)))

    for t_fixed in (0.0, 1.0):
        q = q0 + t_fixed * v
        s = np.dot(q - p0, u) / a
        candidates.append((_clip01(s), t_fixed))

    best_s = np.nan
    best_t = np.nan
    best_distance = np.inf
    for s, t in candidates:
        cp = p0 + s * u
        cq = q0 + t * v
        dist = float(np.linalg.norm(cp - cq))
        if dist < best_distance:
            best_s = s
            best_t = t
            best_distance = dist

    return best_s, best_t, best_distance


def bow_to_strings(bow: np.ndarray, strings: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Closest relation between a bow segment and all string segments.

    bow:     (T, 2, 3)
    strings: (T, 4, 2, 3)
    """
    t_count = bow.shape[0]
    n_strings = strings.shape[1]
    bow_t_all = np.full((t_count, n_strings), np.nan, dtype=np.float64)
    string_u_all = np.full((t_count, n_strings), np.nan, dtype=np.float64)
    distance_all = np.full((t_count, n_strings), np.nan, dtype=np.float64)

    for t in range(t_count):
        p0, p1 = bow[t]
        for s_idx in range(n_strings):
            q0, q1 = strings[t, s_idx]
            bow_t, string_u, dist = closest_segment_pair(p0, p1, q0, q1)
            bow_t_all[t, s_idx] = bow_t
            string_u_all[t, s_idx] = string_u
            distance_all[t, s_idx] = dist

    nearest = np.full((t_count,), -1, dtype=np.int64)
    bow_t = np.full((t_count,), np.nan, dtype=np.float64)
    string_u = np.full((t_count,), np.nan, dtype=np.float64)
    distance = np.full((t_count,), np.nan, dtype=np.float64)
    closest_on_bow = np.full((t_count, 3), np.nan, dtype=np.float64)
    closest_on_string = np.full((t_count, 3), np.nan, dtype=np.float64)

    for t in range(t_count):
        if np.all(np.isnan(distance_all[t])):
            continue
        s_idx = int(np.nanargmin(distance_all[t]))
        nearest[t] = s_idx
        bow_t[t] = bow_t_all[t, s_idx]
        string_u[t] = string_u_all[t, s_idx]
        distance[t] = distance_all[t, s_idx]
        closest_on_bow[t] = bow[t, 0] + bow_t[t] * (bow[t, 1] - bow[t, 0])
        closest_on_string[t] = strings[t, s_idx, 0] + string_u[t] * (strings[t, s_idx, 1] - strings[t, s_idx, 0])

    bow_direction = normalize(bow[:, 1, :] - bow[:, 0, :])
    string_direction_all = normalize(strings[:, :, 1, :] - strings[:, :, 0, :])
    parallel_abs_cos_all = np.abs(np.sum(bow_direction[:, None, :] * string_direction_all, axis=-1))
    parallel_abs_cos = np.full((t_count,), np.nan, dtype=np.float64)
    rows = np.flatnonzero(nearest >= 0)
    parallel_abs_cos[rows] = parallel_abs_cos_all[rows, nearest[rows]]

    return {
        "nearest_index": nearest,
        "bow_t": bow_t,
        "string_u": string_u,
        "distance": distance,
        "closest_on_bow": closest_on_bow,
        "closest_on_string": closest_on_string,
        "distance_all": distance_all,
        "bow_t_all": bow_t_all,
        "string_u_all": string_u_all,
        "parallel_abs_cos": parallel_abs_cos,
        "parallel_abs_cos_all": parallel_abs_cos_all,
        "bow_direction": bow_direction,
        "bow_midpoint": strict_segment_midpoint(bow),
    }


def build_relation_features(data: Mapping[str, np.ndarray], input_path: Path) -> Dict[str, np.ndarray]:
    strings = np.asarray(data["strings_string_frame"], dtype=np.float64)
    contact = np.asarray(data["contact_point_string_frame"], dtype=np.float64)
    bow = np.asarray(data["bow_string_frame"], dtype=np.float64)
    used_finger = np.asarray(data["used_finger_string_frame"], dtype=np.float64)

    contact_rel = point_to_segments(contact, strings)
    bow_rel = bow_to_strings(bow, strings)
    bridge_anchor = bridge_anchor_from_strings(strings)
    bow_root_bridge_distance = np.linalg.norm(bow[:, 0, :] - bridge_anchor, axis=-1)
    bow_root_bridge_valid = (
        np.isfinite(bow[:, 0, :]).all(axis=-1)
        & np.isfinite(bridge_anchor).all(axis=-1)
    )
    bow_root_bridge_distance[~bow_root_bridge_valid] = np.nan
    bow_half_length = 0.5 * np.linalg.norm(bow[:, 1, :] - bow[:, 0, :], axis=-1)
    bow_root_bridge_signed_offset = bow_root_bridge_distance - bow_half_length
    bow_root_bridge_signed_offset[~bow_root_bridge_valid] = np.nan
    finger_tip_rel = point_to_segments(used_finger[:, 3, :], strings)

    contact_idx = contact_rel["nearest_index"]
    bow_idx = bow_rel["nearest_index"]
    finger_tip_idx = finger_tip_rel["nearest_index"]
    contact_valid = contact_idx >= 0
    bow_valid = bow_idx >= 0
    finger_tip_valid = finger_tip_idx >= 0
    same_string = (contact_idx == bow_idx) & contact_valid & bow_valid

    finger_tip_to_contact = np.linalg.norm(used_finger[:, 3, :] - contact, axis=-1)
    finger_tip_to_contact[~(np.isfinite(used_finger[:, 3, :]).all(axis=-1) & np.isfinite(contact).all(axis=-1))] = np.nan

    contact_onehot = one_hot(contact_idx, 4)
    bow_onehot = one_hot(bow_idx, 4)

    feature_names = [
        "contact_string_4",
        "contact_string_3",
        "contact_string_2",
        "contact_string_1",
        "contact_u_nut_to_bridge",
        "contact_to_string_distance",
        "bow_string_4",
        "bow_string_3",
        "bow_string_2",
        "bow_string_1",
        "bow_u_nut_to_bridge",
        "bow_t_on_bow",
        "bow_to_string_distance",
        "bow_parallel_abs_cos",
        "bow_dir_x",
        "bow_dir_y",
        "bow_dir_z",
        "bow_mid_x",
        "bow_mid_y",
        "bow_mid_z",
        "contact_bow_same_string",
        "finger_tip_to_contact_distance",
        "finger_tip_to_string_distance",
    ]
    relation_feature = np.concatenate(
        [
            contact_onehot,
            contact_rel["u"][:, None],
            contact_rel["distance"][:, None],
            bow_onehot,
            bow_rel["string_u"][:, None],
            bow_rel["bow_t"][:, None],
            bow_rel["distance"][:, None],
            bow_rel["parallel_abs_cos"][:, None],
            bow_rel["bow_direction"],
            bow_rel["bow_midpoint"],
            same_string.astype(np.float64)[:, None],
            finger_tip_to_contact[:, None],
            finger_tip_rel["distance"][:, None],
        ],
        axis=1,
    )

    feature_valid_mask = np.isfinite(relation_feature)
    relation_feature_zero_filled = np.where(feature_valid_mask, relation_feature, 0.0)

    out: Dict[str, np.ndarray] = {
        "relation_feature": relation_feature,
        "relation_feature_zero_filled": relation_feature_zero_filled,
        "relation_feature_valid_mask": feature_valid_mask,
        "contact_string_low_to_high": contact_idx,
        "contact_string_onehot_low_to_high": contact_onehot,
        "contact_u_nut_to_bridge": contact_rel["u"],
        "contact_vibrating_length": 1.0 - contact_rel["u"],
        "contact_to_string_distance": contact_rel["distance"],
        "contact_to_all_string_distance": contact_rel["distance_all"],
        "contact_closest_point_on_string": contact_rel["closest_point"],
        "bow_string_low_to_high": bow_idx,
        "bow_string_onehot_low_to_high": bow_onehot,
        "bow_u_nut_to_bridge": bow_rel["string_u"],
        "bow_t_on_bow": bow_rel["bow_t"],
        "bow_to_string_distance": bow_rel["distance"],
        "bow_to_all_string_distance": bow_rel["distance_all"],
        "bow_closest_point_on_bow": bow_rel["closest_on_bow"],
        "bow_closest_point_on_string": bow_rel["closest_on_string"],
        "bow_parallel_abs_cos": bow_rel["parallel_abs_cos"],
        "bow_perpendicular_score": 1.0 - bow_rel["parallel_abs_cos"],
        "bow_direction_string_frame": bow_rel["bow_direction"],
        "bow_midpoint_string_frame": bow_rel["bow_midpoint"],
        "bridge_anchor_string_frame": bridge_anchor,
        "bow_root_bridge_distance": bow_root_bridge_distance,
        "bow_root_bridge_distance_valid": bow_root_bridge_valid,
        "bow_root_bridge_signed_offset": bow_root_bridge_signed_offset,
        "bow_root_bridge_signed_offset_valid": bow_root_bridge_valid,
        "contact_bow_same_string": same_string,
        "contact_valid": contact_valid,
        "bow_valid": bow_valid,
        "finger_tip_string_low_to_high": finger_tip_idx,
        "finger_tip_u_nut_to_bridge": finger_tip_rel["u"],
        "finger_tip_to_string_distance": finger_tip_rel["distance"],
        "finger_tip_to_contact_distance": finger_tip_to_contact,
        "finger_tip_valid": finger_tip_valid,
        "strings_string_frame": strings,
    }

    source_meta = {}
    if "metadata_json" in data:
        try:
            source_meta = json.loads(str(np.asarray(data["metadata_json"]).item()))
        except Exception:
            source_meta = {"metadata_json_parse_error": True}

    metadata = {
        "input": str(input_path),
        "feature_names": feature_names,
        "string_order": "low_to_high: [string_4, string_3, string_2, string_1]",
        "u_convention": "u=0 at top/nut/scroll side, u=1 at bottom/bridge/body side",
        "bow_t_convention": "t=0 at bow point 0, t=1 at bow point 1",
        "relation_feature_note": "Use relation_feature_zero_filled with relation_feature_valid_mask for ML models that cannot consume NaN.",
        "source_metadata": source_meta,
    }
    out["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    return out


def print_summary(out: Mapping[str, np.ndarray]) -> None:
    contact_valid = out["contact_valid"]
    bow_valid = out["bow_valid"]
    same = out["contact_bow_same_string"]
    print(f"frames: {len(contact_valid)}")
    print(f"contact valid: {int(contact_valid.sum())}/{len(contact_valid)}")
    print(f"bow valid: {int(bow_valid.sum())}/{len(bow_valid)}")
    if np.any(contact_valid & bow_valid):
        denom = int((contact_valid & bow_valid).sum())
        print(f"contact/bow same string: {int(same.sum())}/{denom}")
    print(f"mean contact-to-string distance: {np.nanmean(out['contact_to_string_distance']):.6f}")
    print(f"mean bow-to-string distance: {np.nanmean(out['bow_to_string_distance']):.6f}")
    print(f"mean bow perpendicular score: {np.nanmean(out['bow_perpendicular_score']):.6f}")
    print(f"relation_feature shape: {out['relation_feature'].shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input string-frame .npz.")
    parser.add_argument("--output", "-o", type=Path, help="Output relation .npz.")
    parser.add_argument("--print-summary", action="store_true", help="Print relation diagnostics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.input, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}

    required = ["strings_string_frame", "contact_point_string_frame", "bow_string_frame", "used_finger_string_frame"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing required arrays in {args.input}: {missing}")

    out = build_relation_features(data, args.input)
    output = args.output
    if output is None:
        output = args.input.with_name(args.input.stem + "_relations.npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **out)
    print(f"output: {output}")
    if args.print_summary:
        print_summary(out)


if __name__ == "__main__":
    main()
