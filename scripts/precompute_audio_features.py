import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import argparse
from typing import Dict, Iterable, Optional, Tuple


def _plain_config(config) -> Dict:
    from omegaconf import OmegaConf

    return OmegaConf.to_container(config, resolve=True)


def _top_level_audio_overrides(config: Dict) -> Dict:
    keys = (
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
    return {key: config.get(key) for key in keys if key in config}


def _default_cache_dir(data_config: Dict) -> str:
    from audio2pose.data.pretrained_audio import normalize_audio_encoder_type

    audio_encoder_type = normalize_audio_encoder_type(data_config.get("audio_encoder_type"))
    feature_name = audio_encoder_type or "mel_onset_pitch"
    return str(Path(data_config["data_path"]) / "audio_feature_cache" / feature_name)


def _selected_data_keys(config: Dict, requested: Optional[Iterable[str]]) -> Tuple[str, ...]:
    if requested:
        return tuple(requested)
    preferred = ("train_data", "validation_data", "test_data")
    keys = [key for key in preferred if isinstance(config.get(key), dict)]
    if keys:
        return tuple(keys)
    return tuple(key for key, value in config.items() if key.endswith("_data") and isinstance(value, dict))


def _record_frame_count(record: Dict, relation_key: str) -> int:
    import numpy as np

    n_frames = int(record.get("n_frames", 0) or 0)
    if n_frames > 0:
        return n_frames
    with np.load(record["_relation_path"], allow_pickle=False) as z:
        return int(z[relation_key].shape[0])


def _extractor_key(data_config: Dict):
    from audio2pose.data.pretrained_audio import normalize_audio_encoder_type

    audio_encoder_type = normalize_audio_encoder_type(data_config.get("audio_encoder_type"))
    if audio_encoder_type is None:
        return None
    return (
        audio_encoder_type,
        data_config.get("audio_encoder_model_name"),
        data_config.get("audio_encoder_processor_name"),
        int(data_config.get("audio_encoder_layer", -1)),
        str(data_config.get("audio_encoder_device", "cpu")),
        str(data_config.get("audio_encoder_dtype", "float32")),
        data_config.get("audio_encoder_max_seconds"),
        bool(data_config.get("audio_encoder_normalize", True)),
        bool(data_config.get("audio_encoder_local_files_only", False)),
        bool(data_config.get("audio_encoder_trust_remote_code", True)),
    )


def _get_extractor(data_config: Dict, extractor_cache: Dict):
    from audio2pose.data.pretrained_audio import PretrainedAudioFeatureExtractor

    key = _extractor_key(data_config)
    if key is None:
        return None
    if key not in extractor_cache:
        (
            audio_encoder_type,
            model_name,
            processor_name,
            layer,
            device,
            dtype,
            max_seconds,
            normalize,
            local_files_only,
            trust_remote_code,
        ) = key
        extractor_cache[key] = PretrainedAudioFeatureExtractor(
            audio_encoder_type=audio_encoder_type,
            model_name=model_name,
            processor_name=processor_name,
            layer=layer,
            device=device,
            dtype=dtype,
            max_seconds=max_seconds,
            normalize=normalize,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
    return extractor_cache[key]


def precompute_audio_features(
    config_path: str,
    data_keys: Optional[Iterable[str]] = None,
    cache_dir: Optional[str] = None,
    overwrite: bool = False,
    compressed: bool = False,
    limit: Optional[int] = None,
) -> None:
    from omegaconf import OmegaConf
    from tqdm.auto import tqdm

    from audio2pose.data.audio_feature_cache import (
        audio_feature_cache_path,
        compute_frame_aligned_audio_features,
        save_audio_feature_cache,
    )
    from audio2pose.data.data import SPDCanonicalRelationsDataset
    from audio2pose.data.pretrained_audio import (
        apply_audio_encoder_overrides,
        normalize_audio_encoder_type,
    )

    config = _plain_config(OmegaConf.load(config_path))
    top_overrides = _top_level_audio_overrides(config)
    selected_keys = _selected_data_keys(config, data_keys)
    extractor_cache = {}
    seen_paths = set()
    total_written = 0
    total_skipped = 0

    for data_key in selected_keys:
        if data_key not in config or not isinstance(config[data_key], dict):
            raise KeyError(f"Config does not contain a data section named '{data_key}'.")

        data_config = apply_audio_encoder_overrides(config[data_key], top_overrides)
        data_config["audio_feature_cache_dir"] = cache_dir or data_config.get("audio_feature_cache_dir") or _default_cache_dir(data_config)
        data_config["audio_feature_cache_required"] = False
        dataset = SPDCanonicalRelationsDataset(**data_config)
        cache_root = Path(data_config["audio_feature_cache_dir"]).expanduser()
        relation_key = data_config.get("relation_key", "relation_feature_zero_filled")

        records = dataset.records
        if limit is not None:
            records = records[: int(limit)]
        progress = tqdm(records, desc=f"{data_key}: precompute audio", dynamic_ncols=True)
        for record in progress:
            cache_path = audio_feature_cache_path(cache_root, record, record["_audio_path"])
            if cache_path in seen_paths:
                total_skipped += 1
                continue
            seen_paths.add(cache_path)

            if cache_path.exists() and not overwrite:
                total_skipped += 1
                continue

            n_frames = _record_frame_count(record, relation_key)
            extractor = _get_extractor(data_config, extractor_cache)
            audio_encoder_type = normalize_audio_encoder_type(data_config.get("audio_encoder_type"))
            meta = {
                "config_path": str(config_path),
                "data_key": data_key,
                "record_id": record.get("id", ""),
                "instrument": record.get("instrument", ""),
                "audio_encoder_type": audio_encoder_type or "none",
                "audio_encoder_model_name": data_config.get("audio_encoder_model_name"),
                "audio_encoder_processor_name": data_config.get("audio_encoder_processor_name"),
                "audio_encoder_layer": int(data_config.get("audio_encoder_layer", -1)),
                "audio_encoder_normalize": bool(data_config.get("audio_encoder_normalize", True)),
            }
            cache = compute_frame_aligned_audio_features(
                audio_path=record["_audio_path"],
                n_frames=n_frames,
                sr=int(data_config.get("sr", 16000)),
                fps=int(data_config.get("fps", 30)),
                n_mels=int(data_config.get("n_mels", 80)),
                mel_hop_length=int(data_config.get("mel_hop_length", 256)),
                fmin_note=str(data_config.get("fmin_note", "C2")),
                fmax_note=str(data_config.get("fmax_note", "E7")),
                pretrained_extractor=extractor,
                meta=meta,
            )
            save_audio_feature_cache(cache_path, cache, compressed=compressed)
            total_written += 1
            progress.set_postfix(written=total_written, skipped=total_skipped)

    print(f"Done. Wrote {total_written} cache file(s), skipped {total_skipped}.")


def upgrade_cache_with_dynamics(
    config_path: str,
    data_keys: Optional[Iterable[str]] = None,
    cache_dir: Optional[str] = None,
    limit: Optional[int] = None,
) -> None:
    """Add bow-dynamics channels to existing cache files in place.

    Only the cheap DSP is redone from the raw audio (onset strength, spectral
    flux, log-RMS, HF ratio); mel/onset/pitch and expensive pretrained
    features (MERT/Music2Vec) are copied through unchanged. Missing cache
    files are reported, not created - run the full precompute for those.
    """
    import json

    import librosa
    import numpy as np
    from omegaconf import OmegaConf
    from tqdm.auto import tqdm

    from audio2pose.data.audio_feature_cache import (
        BOW_DYNAMICS_CHANNEL_NAMES,
        audio_feature_cache_path,
        compute_bow_dynamics_channels,
    )
    from audio2pose.data.data import SPDCanonicalRelationsDataset
    from audio2pose.data.pretrained_audio import apply_audio_encoder_overrides

    config = _plain_config(OmegaConf.load(config_path))
    top_overrides = _top_level_audio_overrides(config)
    selected_keys = _selected_data_keys(config, data_keys)
    seen_paths = set()
    upgraded = skipped = missing = already = 0

    for data_key in selected_keys:
        if data_key not in config or not isinstance(config[data_key], dict):
            raise KeyError(f"Config does not contain a data section named '{data_key}'.")
        data_config = apply_audio_encoder_overrides(config[data_key], top_overrides)
        data_config["audio_feature_cache_dir"] = cache_dir or data_config.get("audio_feature_cache_dir") or _default_cache_dir(data_config)
        data_config["audio_feature_cache_required"] = False
        dataset = SPDCanonicalRelationsDataset(**data_config)
        cache_root = Path(data_config["audio_feature_cache_dir"]).expanduser()
        relation_key = data_config.get("relation_key", "relation_feature_zero_filled")

        records = dataset.records
        if limit is not None:
            records = records[: int(limit)]
        progress = tqdm(records, desc=f"{data_key}: dynamics upgrade", dynamic_ncols=True)
        for record in progress:
            cache_path = audio_feature_cache_path(cache_root, record, record["_audio_path"])
            if cache_path in seen_paths:
                continue
            seen_paths.add(cache_path)
            if not cache_path.exists():
                missing += 1
                continue
            with np.load(cache_path, allow_pickle=False) as z:
                if "dynamics" in z:
                    already += 1
                    continue
                arrays = {key: z[key] for key in z.files if key != "meta_json"}
                meta = json.loads(str(z["meta_json"].item())) if "meta_json" in z.files else {}

            n_frames = _record_frame_count(record, relation_key)
            sr = int(data_config.get("sr", 16000))
            fps = int(data_config.get("fps", 30))
            audio, _ = librosa.load(record["_audio_path"], sr=sr)
            dynamics = compute_bow_dynamics_channels(
                audio,
                sr=sr,
                n_frames=n_frames,
                fps=fps,
                n_mels=int(data_config.get("n_mels", 80)),
                mel_hop_length=int(data_config.get("mel_hop_length", 256)),
            )
            if "mel" in arrays and arrays["mel"].shape[0] != n_frames:
                raise ValueError(
                    f"{cache_path}: cached mel has {arrays['mel'].shape[0]} frames but the "
                    f"relation file says {n_frames}; regenerate this cache from scratch."
                )
            arrays["dynamics"] = dynamics.astype(np.float32)
            meta["dynamics_dim"] = int(dynamics.shape[-1])
            meta["dynamics_channels"] = list(BOW_DYNAMICS_CHANNEL_NAMES)
            arrays["meta_json"] = np.asarray(json.dumps(meta, sort_keys=True), dtype=np.str_)
            np.savez(cache_path, **arrays)
            upgraded += 1
            progress.set_postfix(upgraded=upgraded, already=already, missing=missing)

    print(
        f"Dynamics upgrade done. upgraded={upgraded}, already_had_dynamics={already}, "
        f"missing_cache={missing}, skipped_duplicates={len(seen_paths) - upgraded - already - missing}."
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute SPD mel/onset/pitch/MERT/Music2Vec features to .npz files.")
    parser.add_argument("--config", required=True, help="Training or test yaml config.")
    parser.add_argument(
        "--data-key",
        action="append",
        default=None,
        help="Data section to precompute, e.g. train_data. Can be repeated. Defaults to train/validation/test sections.",
    )
    parser.add_argument("--cache-dir", default=None, help="Override audio_feature_cache_dir from the config.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing cache files.")
    parser.add_argument("--compressed", action="store_true", help="Use np.savez_compressed instead of faster uncompressed npz.")
    parser.add_argument("--limit", type=int, default=None, help="Only precompute the first N records per data section.")
    parser.add_argument(
        "--dynamics-only",
        action="store_true",
        help=(
            "Upgrade existing cache files in place by adding the bow-dynamics channels; "
            "mel/onset/pitch and pretrained (MERT) features are preserved."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.dynamics_only:
        if args.overwrite:
            raise SystemExit("--dynamics-only cannot be combined with --overwrite.")
        upgrade_cache_with_dynamics(
            config_path=args.config,
            data_keys=args.data_key,
            cache_dir=args.cache_dir,
            limit=args.limit,
        )
    else:
        precompute_audio_features(
            config_path=args.config,
            data_keys=args.data_key,
            cache_dir=args.cache_dir,
            overwrite=args.overwrite,
            compressed=args.compressed,
            limit=args.limit,
        )
