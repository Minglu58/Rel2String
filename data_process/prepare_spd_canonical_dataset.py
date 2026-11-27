#!/usr/bin/env python3
"""
Batch-preprocess SPD cello/violin raw mocap into canonical string-frame data.

Expected raw layout:
  datasets/SPD/
    cello/
      kp3d/*.npy
      audio/*
    violin/
      kp3d/*.npy
      audio/*

Output layout:
  datasets/SPD_canonical/
    manifest.json
    manifest.csv
    README.md
    splits/
      train.json
      val.json
      test.json
      train.txt
      val.txt
      test.txt
    samples/
      cello/cello01/
        audio.wav
        string_frame.npz
        playable_relations.npz
        metadata.json
      violin/violin01/
        ...

Splits are piece-level and instrument-balanced. Do not split temporal windows
before this step, otherwise train/test leakage is very likely.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from spd_playable_relations import build_relation_features  # noqa: E402
from spd_string_frame import build_output as build_string_frame_output  # noqa: E402
from spd_string_frame import get_index_map  # noqa: E402


AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg")
INSTRUMENTS = ("cello", "violin")


@dataclass
class SampleRecord:
    id: str
    instrument: str
    piece: str
    split: str
    n_frames: int
    n_keypoints: int
    fps: float
    raw_kp3d: str
    raw_audio: Optional[str]
    sample_dir: str
    audio: Optional[str]
    string_frame: str
    playable_relations: str
    relation_feature_dim: int
    pose_target: str


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def relative_to(path: Optional[Path], root: Path) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def find_kp3d_files(spd_root: Path, instrument: str) -> List[Path]:
    kp3d_dir = spd_root / instrument / "kp3d"
    if not kp3d_dir.exists():
        return []
    return sorted(p for p in kp3d_dir.glob("*.npy") if not p.name.startswith("."))


def index_audio_files(audio_dirs: Sequence[Path]) -> Dict[str, Path]:
    audio_by_stem: Dict[str, Path] = {}
    for audio_dir in audio_dirs:
        if not audio_dir.exists():
            continue
        for path in audio_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in AUDIO_EXTS and not path.name.startswith("."):
                audio_by_stem.setdefault(path.stem, path)
    return audio_by_stem


def split_ids(piece_ids: Sequence[str], val_ratio: float, test_ratio: float, seed: int) -> Dict[str, List[str]]:
    ids = list(piece_ids)
    Random(seed).shuffle(ids)
    n_total = len(ids)
    if n_total == 0:
        return {"train": [], "val": [], "test": []}

    n_test = int(round(n_total * test_ratio))
    n_val = int(round(n_total * val_ratio))

    if n_total >= 3:
        if test_ratio > 0:
            n_test = max(1, n_test)
        if val_ratio > 0:
            n_val = max(1, n_val)

    if n_val + n_test >= n_total:
        overflow = n_val + n_test - (n_total - 1)
        while overflow > 0 and n_val > 0:
            n_val -= 1
            overflow -= 1
        while overflow > 0 and n_test > 0:
            n_test -= 1
            overflow -= 1

    test = ids[:n_test]
    val = ids[n_test : n_test + n_val]
    train = ids[n_test + n_val :]
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def build_splits(
    pieces_by_instrument: Dict[str, List[str]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, str]:
    split_by_sample: Dict[str, str] = {}
    for offset, instrument in enumerate(INSTRUMENTS):
        splits = split_ids(
            pieces_by_instrument.get(instrument, []),
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed + offset * 997,
        )
        for split, pieces in splits.items():
            for piece in pieces:
                split_by_sample[f"{instrument}_{piece}"] = split
    return split_by_sample


def load_fixed_splits(split_dir: Path) -> Dict[str, str]:
    split_by_sample = {}
    for split in ("train", "val", "test"):
        for line in (split_dir / f"{split}.txt").read_text(encoding="utf-8").splitlines():
            sample_id = line.strip()
            if not sample_id:
                continue
            if sample_id in split_by_sample:
                raise ValueError(f"Duplicate SPD sample in {split_dir}: {sample_id}")
            split_by_sample[sample_id] = split
    return split_by_sample


def link_or_copy_audio(src: Optional[Path], dst: Path, mode: str, overwrite: bool) -> Optional[Path]:
    if src is None or mode == "none":
        return None

    suffix = src.suffix.lower()
    dst_audio = dst / f"audio{suffix}"
    if dst_audio.exists() or dst_audio.is_symlink():
        if not overwrite:
            return dst_audio
        dst_audio.unlink()

    if mode == "copy":
        shutil.copy2(src, dst_audio)
    elif mode == "symlink":
        os.symlink(src.resolve(), dst_audio)
    else:
        raise ValueError("audio_mode must be symlink, copy, or none.")
    return dst_audio


def write_npz_if_needed(path: Path, arrays: Dict[str, np.ndarray], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    np.savez_compressed(path, **arrays)


def process_sample(
    kp3d_path: Path,
    audio_path: Optional[Path],
    instrument: str,
    split: str,
    output_root: Path,
    project_root: Path,
    audio_mode: str,
    fps: float,
    overwrite: bool,
    origin: str,
    scale: str,
    dry_run: bool,
) -> SampleRecord:
    piece = kp3d_path.stem
    sample_id = f"{instrument}_{piece}"
    sample_dir = output_root / "samples" / instrument / piece
    string_frame_path = sample_dir / "string_frame.npz"
    relations_path = sample_dir / "playable_relations.npz"
    metadata_path = sample_dir / "metadata.json"

    data = np.load(kp3d_path, allow_pickle=False)
    if data.ndim != 3 or data.shape[-1] != 3:
        raise ValueError(f"{kp3d_path} must be shaped (T, N, 3), got {data.shape}.")

    idx = get_index_map(instrument, data.shape[1])
    relation_feature_dim = 23
    audio_dst: Optional[Path] = None

    if not dry_run:
        sample_dir.mkdir(parents=True, exist_ok=True)
        string_frame = build_string_frame_output(data, idx, origin=origin, scale=scale, input_path=kp3d_path)
        write_npz_if_needed(string_frame_path, string_frame, overwrite=overwrite)
        relations = build_relation_features(string_frame, string_frame_path)
        write_npz_if_needed(relations_path, relations, overwrite=overwrite)
        relation_feature_dim = int(relations["relation_feature"].shape[1])
        audio_dst = link_or_copy_audio(audio_path, sample_dir, mode=audio_mode, overwrite=overwrite)

    record = SampleRecord(
        id=sample_id,
        instrument=instrument,
        piece=piece,
        split=split,
        n_frames=int(data.shape[0]),
        n_keypoints=int(data.shape[1]),
        fps=float(fps),
        raw_kp3d=relative_to(kp3d_path, project_root) or str(kp3d_path),
        raw_audio=relative_to(audio_path, project_root),
        sample_dir=relative_to(sample_dir, project_root) or str(sample_dir),
        audio=relative_to(audio_dst, project_root),
        string_frame=relative_to(string_frame_path, project_root) or str(string_frame_path),
        playable_relations=relative_to(relations_path, project_root) or str(relations_path),
        relation_feature_dim=relation_feature_dim,
        pose_target="coco_wholebody_string_frame + bow_string_frame",
    )

    if not dry_run:
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(record), f, indent=2, sort_keys=True)

    return record


def write_manifests(records: Sequence[SampleRecord], output_root: Path, dry_run: bool) -> None:
    if dry_run:
        return

    output_root.mkdir(parents=True, exist_ok=True)
    splits_dir = output_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    records_dict = [asdict(r) for r in records]
    with (output_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(records_dict, f, indent=2, sort_keys=True)

    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records_dict[0].keys()) if records_dict else [])
        if records_dict:
            writer.writeheader()
            writer.writerows(records_dict)

    for split in ("train", "val", "test"):
        split_records = [r for r in records if r.split == split]
        split_dicts = [asdict(r) for r in split_records]
        with (splits_dir / f"{split}.json").open("w", encoding="utf-8") as f:
            json.dump(split_dicts, f, indent=2, sort_keys=True)
        with (splits_dir / f"{split}.txt").open("w", encoding="utf-8") as f:
            for r in split_records:
                f.write(r.id + "\n")

    write_readme(output_root, records)


def write_readme(output_root: Path, records: Sequence[SampleRecord]) -> None:
    counts = {
        split: sum(1 for r in records if r.split == split)
        for split in ("train", "val", "test")
    }
    by_instrument = {
        instrument: sum(1 for r in records if r.instrument == instrument)
        for instrument in INSTRUMENTS
    }
    text = f"""# SPD Canonical

This directory was generated by `data_process/prepare_spd_canonical_dataset.py`.

Each sample contains:

- `audio.*`: synchronized audio, copied or symlinked from raw SPD.
- `string_frame.npz`: raw 3D keypoints transformed into the canonical string frame.
- `playable_relations.npz`: compact playable-relation features extracted from `string_frame.npz`.
- `metadata.json`: per-sample metadata.

Splits are piece-level and instrument-balanced. Temporal training windows should be sampled only after reading these split files.

Counts:

- train: {counts["train"]}
- val: {counts["val"]}
- test: {counts["test"]}
- cello: {by_instrument["cello"]}
- violin: {by_instrument["violin"]}

Recommended first pose target:

`coco_wholebody_string_frame + bow_string_frame`

Recommended relation input/target:

`playable_relations.npz::relation_feature_zero_filled` with
`playable_relations.npz::relation_feature_valid_mask`.
"""
    with (output_root / "README.md").open("w", encoding="utf-8") as f:
        f.write(text)


def collect_dataset(
    spd_root: Path,
    extra_audio_root: Optional[Path],
    max_files_per_instrument: Optional[int],
) -> Tuple[Dict[str, List[Path]], Dict[str, Dict[str, Path]]]:
    kp3d_by_instrument: Dict[str, List[Path]] = {}
    audio_by_instrument: Dict[str, Dict[str, Path]] = {}

    for instrument in INSTRUMENTS:
        kp3d_files = find_kp3d_files(spd_root, instrument)
        if max_files_per_instrument is not None:
            kp3d_files = kp3d_files[:max_files_per_instrument]
        kp3d_by_instrument[instrument] = kp3d_files

        audio_dirs = [spd_root / instrument / "audio"]
        if extra_audio_root is not None:
            audio_dirs.extend(
                [
                    extra_audio_root / instrument / "audio",
                    extra_audio_root / instrument,
                ]
            )
        audio_by_instrument[instrument] = index_audio_files(audio_dirs)

    return kp3d_by_instrument, audio_by_instrument


def parse_args() -> argparse.Namespace:
    project_root = repo_root_from_script()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spd-root", type=Path, default=project_root / "datasets" / "SPD")
    parser.add_argument("--output-root", type=Path, default=project_root / "datasets" / "SPD_canonical")
    parser.add_argument(
        "--extra-audio-root",
        type=Path,
        help="Optional extra tree to search for audio files by stem.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--split-dir", type=Path,
        help="Use the exact sample IDs in train.txt, val.txt and test.txt.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--origin", choices=("nut", "bridge", "midpoint"), default="nut")
    parser.add_argument(
        "--scale",
        choices=("median_nut_to_bridge", "nut_to_bridge", "none"),
        default="median_nut_to_bridge",
    )
    parser.add_argument("--audio-mode", choices=("symlink", "copy", "none"), default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-missing-audio", action="store_true")
    parser.add_argument("--max-files-per-instrument", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = repo_root_from_script()
    spd_root = args.spd_root.resolve()
    output_root = args.output_root.resolve()

    kp3d_by_instrument, audio_by_instrument = collect_dataset(
        spd_root,
        extra_audio_root=args.extra_audio_root.resolve() if args.extra_audio_root else None,
        max_files_per_instrument=args.max_files_per_instrument,
    )

    pieces_by_instrument = {
        instrument: [p.stem for p in paths]
        for instrument, paths in kp3d_by_instrument.items()
    }
    if args.split_dir:
        split_by_sample = load_fixed_splits(args.split_dir)
        available_ids = {
            f"{instrument}_{piece}"
            for instrument, pieces in pieces_by_instrument.items()
            for piece in pieces
        }
        missing_ids = sorted(split_by_sample.keys() - available_ids)
        if missing_ids:
            raise SystemExit(f"Missing 3D annotations for the fixed split: {', '.join(missing_ids)}")
        if not args.allow_missing_audio:
            missing_audio_ids = [
                sample_id for sample_id in split_by_sample
                if sample_id.partition("_")[2] not in audio_by_instrument[sample_id.partition("_")[0]]
            ]
            if missing_audio_ids:
                raise SystemExit(f"Missing audio for the fixed split: {', '.join(sorted(missing_audio_ids))}")
    else:
        split_by_sample = build_splits(
            pieces_by_instrument,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    records: List[SampleRecord] = []
    missing_audio: List[str] = []
    for instrument in INSTRUMENTS:
        kp3d_files = kp3d_by_instrument[instrument]
        audio_by_stem = audio_by_instrument[instrument]
        print(f"{instrument}: {len(kp3d_files)} kp3d files, {len(audio_by_stem)} indexed audio files")

        for kp3d_path in kp3d_files:
            piece = kp3d_path.stem
            sample_id = f"{instrument}_{piece}"
            if sample_id not in split_by_sample:
                continue
            split = split_by_sample[sample_id]
            audio_path = audio_by_stem.get(piece)
            if audio_path is None:
                missing_audio.append(sample_id)
                if not args.allow_missing_audio:
                    continue

            record = process_sample(
                kp3d_path=kp3d_path,
                audio_path=audio_path,
                instrument=instrument,
                split=split,
                output_root=output_root,
                project_root=project_root,
                audio_mode=args.audio_mode,
                fps=args.fps,
                overwrite=args.overwrite,
                origin=args.origin,
                scale=args.scale,
                dry_run=args.dry_run,
            )
            records.append(record)
            print(f"[{split}] {record.id}: frames={record.n_frames}, audio={'yes' if audio_path else 'no'}")

    if missing_audio:
        print(f"missing audio: {len(missing_audio)} samples")
        print(", ".join(missing_audio[:20]) + (" ..." if len(missing_audio) > 20 else ""))
        if not args.allow_missing_audio:
            raise SystemExit(
                "Some samples are missing matching audio. Re-run with --allow-missing-audio, "
                "--extra-audio-root, or fix datasets/SPD/{instrument}/audio."
            )

    write_manifests(records, output_root, dry_run=args.dry_run)

    print(f"processed samples: {len(records)}")
    print(f"output root: {output_root}")
    if args.dry_run:
        print("dry run: no files were written")


if __name__ == "__main__":
    main()
