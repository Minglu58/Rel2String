#!/usr/bin/env python3
"""Re-evaluate saved native SPD predictions without loading a motion model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from audio2pose.metrics.elgar_original import ElgarOriginalAccumulator, PitchCache, protocol_metadata
from audio2pose.metrics.reporting import write_metric_reports, write_seed_metric_reports


def resolve_existing(value, *bases):
    path = Path(value)
    candidates = [path] + [Path(base) / path for base in bases]
    # Saved reports often contain paths from the training server. Match only
    # known project-relative suffixes; never silently choose another record.
    for marker in ("results", "datasets"):
        if marker in path.parts:
            suffix = Path(*path.parts[path.parts.index(marker):])
            candidates.append(ROOT / suffix)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot locate {value}")


def load_record_context(manifest):
    manifest = Path(manifest).resolve()
    records = json.loads(manifest.read_text())
    if not isinstance(records, list):
        raise ValueError("Expected a list of SPD manifest records.")
    result = {}
    for record in records:
        sample_dir = manifest.parent / "samples" / record["instrument"] / record["piece"]
        # Canonical copies are portable, whereas raw_audio may point to SPD
        # folders no longer present on the machine.
        frame = sample_dir / "string_frame.npz"
        audio = sample_dir / "audio.wav"
        if not frame.exists():
            frame = resolve_existing(record["string_frame"], ROOT, manifest.parent)
        if not audio.exists():
            audio = resolve_existing(record["audio"], ROOT, manifest.parent)
        if float(record.get("fps", 30)) != 30.:
            raise ValueError("ELGAR original protocol requires 30 fps.")
        result[record["id"]] = {"string_frame_path": frame, "audio_path": audio, "instrument": record["instrument"]}
    return result


def evaluate_directory(input_dir, contexts, pitch_cache, output_dir, check_only=False):
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if input_dir == output_dir or input_dir in output_dir.parents:
        raise ValueError("Use a separate output directory outside the original seed directory.")
    reports = json.loads((input_dir / "npy_reports.json").read_text())
    if not reports:
        raise ValueError(f"Empty npy_reports.json: {input_dir}")
    accumulator = ElgarOriginalAccumulator()
    for report in reports:
        sample_id = report["sample_id"]
        if sample_id not in contexts:
            raise ValueError(f"{sample_id} is absent from the supplied manifest.")
        # A relocated seed directory can also be addressed by its local file.
        # Verify through the raw NPY metadata when available.
        local = input_dir / "npy_predictions" / Path(report["raw_npy_path"]).name
        path = local if local.exists() else resolve_existing(report["raw_npy_path"], input_dir, ROOT)
        meta_path = input_dir / "npy_predictions" / Path(report.get("npy_meta_path", "unused")).name
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("raw_coordinate", "string_frame") != "string_frame":
                raise ValueError(f"Expected raw string-frame predictions: {path}")
        accumulator.add_window(sample_id, report["start_frame"], report["end_frame"],
                               np.load(path, allow_pickle=False), **contexts[sample_id])
    source_protocol = input_dir / "metrics_protocol.json"
    source = json.loads(source_protocol.read_text()) if source_protocol.exists() else {}
    protocol = protocol_metadata()
    protocol.update({"input_dir": str(input_dir), "source_postprocessing": source,
                     "pitch_cache_dir": str(pitch_cache.directory.resolve())})
    if check_only:
        print(f"Checked {len(reports)} windows / {len(accumulator.records)} records in {input_dir}")
        return None
    metrics, by_sample = accumulator.compute(pitch_cache)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (("metrics.json", metrics), ("metrics_by_sample.json", by_sample), ("metrics_protocol.json", protocol)):
        (output_dir / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    write_metric_reports(output_dir, metrics, title="ELGAR original protocol", metadata={"protocol": protocol["protocol"]})
    print(f"{input_dir.name}: FCD={metrics['elgar_left_hand_contact_deviation_mm']}, "
          f"BSD={metrics['elgar_bow_string_deviation_mm']}, BF1={metrics['elgar_bowing_f1']}, "
          f"BCS={metrics['elgar_bowing_cosine_similarity']}", flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="A seed directory or a parent with seed-* directories")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; original metrics are kept intact")
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/SPD_canonical/manifest.json")
    parser.add_argument("--pitch-cache-dir", type=Path, default=ROOT / "datasets/SPD_canonical/elgar_pitch_cache")
    parser.add_argument("--pitch-device", default="cuda:0", help="torchcrepe device, e.g. cuda:0 or cpu")
    parser.add_argument("--pitch-cache-required", action="store_true", help="Fail instead of running pitch extraction")
    parser.add_argument("--check-only", action="store_true", help="Check files/shapes without extracting pitch or writing outputs")
    args = parser.parse_args()
    input_path, output_path = args.input_dir.resolve(), args.output_dir.resolve()
    if input_path == output_path or input_path in output_path.parents or output_path in input_path.parents:
        raise ValueError("--output-dir must be separate from the existing input tree.")
    contexts = load_record_context(args.manifest)
    cache = PitchCache(args.pitch_cache_dir, args.pitch_device, args.pitch_cache_required)
    if (args.input_dir / "npy_reports.json").exists():
        evaluate_directory(args.input_dir, contexts, cache, args.output_dir, args.check_only)
        return
    seeds = sorted(p for p in args.input_dir.glob("seed-*") if (p / "npy_reports.json").exists())
    if not seeds:
        raise FileNotFoundError(f"No saved predictions found in {args.input_dir}")
    by_seed = {}
    for seed_dir in seeds:
        seed = int(seed_dir.name[len("seed-"):])
        metrics = evaluate_directory(seed_dir, contexts, cache, args.output_dir / seed_dir.name, args.check_only)
        if metrics is not None:
            by_seed[seed] = metrics
    if by_seed:
        write_seed_metric_reports(args.output_dir, by_seed, title="ELGAR original protocol | sampling means",
                                  metadata={"protocol": "elgar_original_v1", "input_dir": str(args.input_dir)})
        (args.output_dir / "metrics_protocol.json").write_text(json.dumps(protocol_metadata(), indent=2) + "\n")


if __name__ == "__main__":
    main()
