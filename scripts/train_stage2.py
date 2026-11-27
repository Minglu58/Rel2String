"""Train Stage 2 with frozen audio-to-relation predictions from Stage 1."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
import logging
import math
import os
import re
import shutil
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import diffusers
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from audio2pose.data.data import SPDCanonicalPoseDataset
from audio2pose.data.syncviolinist_pose import (
    decode_pose_grid,
    require_same_representation,
    resolve_pose_representation,
)
from audio2pose.data.validation import (
    average_piece_metric_sums,
    evenly_spaced_indices,
    select_validation_windows,
)
from audio2pose.data.pretrained_audio import apply_audio_encoder_overrides, disable_cuda_audio_encoder_workers
from audio2pose.metrics.audio_alignment import add_audio_alignment_f1, compute_audio_alignment_metric_sums
from audio2pose.metrics.elgar_original import (
    add_elgar_bowing_f1,
    compute_contact_diagnostic_sums,
    compute_elgar_metric_sums,
    metric_sum,
)
from audio2pose.metrics.reporting import compact_metric_log, write_grouped_metrics_text
from audio2pose.models.condition_encoder import (
    AdditiveDualPoseConditionEncoder,
    DualPoseConditionEncoder,
    GatedFusionPoseConditionEncoder,
    PoseConditionEncoder,
    infer_dual_condition_dims,
    make_audio_condition,
    make_relation_condition,
)
from audio2pose.models.unet import UNet3DConditionModel
from audio2pose.pipelines.pose_context import training_context
from audio2pose.losses.bow_radial_velocity import bow_radial_velocity_loss
from audio2pose.training import resolve_stage2_config as _merge_stage2_config
from audio2pose.pipelines.predicted_relations import (
    _dropped_relation_channel_mask,
    _predicted_relation_channel_mask,
    _stage1_metrics,
    _verify_dynamics_cache,
    build_predicted_relation_dataset,
    load_mapping_model,
)


logger = get_logger(__name__, log_level="INFO")


def _joint_mapping_relation(
    raw_relation: torch.Tensor,
    *,
    categorical_temperature: float = 1.0,
    hard_categorical: bool = True,
    relation_dropped_slices: Optional[Sequence[Sequence[int]]] = None,
) -> torch.Tensor:
    """Convert differentiable Stage-1 outputs to the Stage-2 relation format."""
    # Do not write transformed slices back into ``relation`` in-place.  The
    # categorical probabilities and direction normalization below are computed
    # from views of the same tensor; in-place assignment increments its
    # autograd version counter and breaks J1 backward with a version-mismatch
    # error.  Build the output from independent functional pieces instead.
    raw = torch.nan_to_num(raw_relation.float(), nan=0.0, posinf=0.0, neginf=0.0)
    temperature = max(float(categorical_temperature), 1e-4)
    def categorical(start: int, end: int) -> torch.Tensor:
        probabilities = torch.softmax(raw[..., start:end] / temperature, dim=-1)
        if hard_categorical:
            indices = probabilities.argmax(dim=-1)
            hard = F.one_hot(indices, num_classes=end - start).to(probabilities.dtype)
            probabilities = hard - probabilities.detach() + probabilities
        return probabilities

    direction = raw[..., 14:17]
    direction_norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    direction = torch.where(
        direction_norm > 1e-6,
        direction / direction_norm.clamp_min(1e-6),
        direction,
    )
    relation = torch.cat(
        (
            categorical(0, 4),
            raw[..., 4:5].clamp(0.0, 1.0),
            raw[..., 5:6].clamp_min(0.0),
            categorical(6, 10),
            raw[..., 10:11].clamp(0.0, 1.0),
            raw[..., 11:12].clamp(0.0, 1.0),
            raw[..., 12:13].clamp_min(0.0),
            raw[..., 13:14].clamp(0.0, 1.0),
            direction,
            raw[..., 17:20],
            raw[..., 20:21].clamp(0.0, 1.0),
            raw[..., 21:23].clamp_min(0.0),
        ),
        dim=-1,
    )

    dropped_mask = torch.ones(
        relation.shape[-1], device=relation.device, dtype=relation.dtype
    )
    for values in relation_dropped_slices or ():
        start, end = int(values[0]), int(values[1])
        dropped_mask[start:end] = 0.0
    return relation * dropped_mask.view(*([1] * (relation.ndim - 1)), -1)


def _joint_mapping_forward(
    batch: Dict[str, Any],
    mapping_model: torch.nn.Module,
    *,
    device: torch.device,
    categorical_temperature: float,
    hard_categorical: bool,
    relation_dropped_slices: Optional[Sequence[Sequence[int]]],
    stage1_loss_kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run Stage 1 in the Stage-2 graph and return relation plus preservation loss."""
    pretrained_audio = batch.get("pretrained_audio")
    if pretrained_audio is not None:
        pretrained_audio = pretrained_audio.to(device=device, dtype=torch.float32, non_blocking=True)
    instrument_id = batch.get("instrument_id")
    if instrument_id is not None:
        instrument_id = instrument_id.to(device=device, dtype=torch.long, non_blocking=True)

    mel = batch["mel"].to(device=device, dtype=torch.float32, non_blocking=True)
    onset = batch["onset"].to(device=device, dtype=torch.float32, non_blocking=True)
    pitch = batch["pitch"].to(device=device, dtype=torch.float32, non_blocking=True)
    if hasattr(mapping_model, "forward_with_validity"):
        raw_relation, validity_logits = mapping_model.forward_with_validity(
            mel, onset, pitch, pretrained_audio, instrument_id
        )
    else:
        raw_relation = mapping_model(mel, onset, pitch, pretrained_audio, instrument_id)
        validity_logits = None

    target = batch["relation"].to(device=device, dtype=torch.float32, non_blocking=True)
    relation_mask = batch["relation_mask"].to(device=device, dtype=torch.float32, non_blocking=True)
    contact_valid = batch["contact_valid"].to(device=device, dtype=torch.bool, non_blocking=True)
    bow_valid = batch["bow_valid"].to(device=device, dtype=torch.bool, non_blocking=True)

    # Import lazily to keep the regular Stage-2 entry point independent from
    # the Stage-1 training module's optional dependencies.
    from scripts.train_stage1 import relation_prediction_loss

    preservation_loss, _ = relation_prediction_loss(
        raw_relation,
        target,
        relation_mask,
        contact_valid,
        bow_valid,
        validity_logits=validity_logits,
        **stage1_loss_kwargs,
    )
    relation = _joint_mapping_relation(
        raw_relation,
        categorical_temperature=categorical_temperature,
        hard_categorical=hard_categorical,
        relation_dropped_slices=relation_dropped_slices,
    )
    return relation, preservation_loss


def _set_joint_mapping_train_mode(mapping_model: torch.nn.Module) -> None:
    """Keep frozen Stage-1 modules deterministic while training selected heads."""
    base_model = getattr(mapping_model, "module", mapping_model)
    base_model.eval()
    for name, module in base_model.named_modules():
        if name and any(parameter.requires_grad for parameter in module.parameters()):
            module.train()


DEFAULT_FACE_JOINT_INDICES = tuple(range(23, 91))
DEFAULT_LOWER_BODY_JOINT_INDICES = (13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
DEFAULT_TORSO_ANCHOR_JOINT_INDICES = (5, 6, 11, 12)
DEFAULT_IGNORED_JOINT_INDICES = (9, 10)
DEFAULT_LEFT_HAND_JOINT_INDICES = tuple(range(91, 112))
DEFAULT_RIGHT_HAND_JOINT_INDICES = tuple(range(112, 133))
DEFAULT_BOW_JOINT_INDICES = (133, 134)
DEFAULT_FINGERTIP_JOINT_INDICES = (95, 99, 103, 107, 111)
DEFAULT_BODY_HAND_WRIST_OFFSET_PAIRS = ((9, 91), (10, 112))
DEFAULT_CONTACT_FINGERTIP_INDICES = (95, 99, 103, 107, 111)


def normalize_joint_indices(indices: Optional[Iterable[int]]) -> Tuple[int, ...]:
    if indices is None:
        return ()
    return tuple(int(index) for index in indices)


def set_joint_weights(
    weights: torch.Tensor,
    indices: Optional[Iterable[int]],
    value: float,
    joint_count: int,
) -> None:
    for index in normalize_joint_indices(indices):
        if 0 <= index < joint_count:
            weights[index] = float(value)


def resolve_resume_checkpoint(output_dir: str, resume_from_checkpoint: Optional[str]) -> Optional[Path]:
    if not resume_from_checkpoint:
        return None

    output_path = Path(output_dir)
    if resume_from_checkpoint == "latest":
        if not output_path.is_dir():
            raise FileNotFoundError(f"Cannot resume latest checkpoint because output_dir does not exist: {output_dir}")
        checkpoints = [
            path
            for path in output_path.iterdir()
            if path.is_dir() and path.name.startswith("checkpoint-")
        ]
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint-* directories found in {output_dir}")
        return sorted(checkpoints, key=lambda path: int(path.name.split("-")[1]))[-1]

    candidate = Path(str(resume_from_checkpoint)).expanduser()
    if candidate.exists():
        return candidate

    by_name = output_path / candidate.name
    if by_name.exists():
        return by_name

    raise FileNotFoundError(
        f"resume_from_checkpoint was set to '{resume_from_checkpoint}', but neither "
        f"'{candidate}' nor '{by_name}' exists."
    )


def create_pose_unet(
    in_channels: int = 3,
    cross_attention_dim: int = 320,
    sample_size: int = 16,
    block_out_channels: Tuple[int, ...] = (128, 256, 512),
    layers_per_block: int = 1,
    attention_head_dim: Tuple[int, ...] = (4, 4, 8),
    norm_num_groups: int = 32,
):
    return UNet3DConditionModel(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=in_channels,
        down_block_types=("CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "DownBlock3D"),
        up_block_types=("UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D"),
        block_out_channels=tuple(block_out_channels),
        layers_per_block=layers_per_block,
        cross_attention_dim=cross_attention_dim,
        attention_head_dim=tuple(attention_head_dim),
        norm_num_groups=norm_num_groups,
        use_skeleton_adapter=False,
    )


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is None:
        return F.mse_loss(pred.float(), target.float(), reduction="mean")
    mask = mask.float()
    loss = (pred.float() - target.float()) ** 2 * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def make_joint_weight_vector(
    joint_count: int,
    device,
    dtype,
    body_weight: float = 1.0,
    left_hand_weight: float = 6.0,
    left_hand_indices: Optional[Iterable[int]] = DEFAULT_LEFT_HAND_JOINT_INDICES,
    right_hand_weight: float = 3.0,
    right_hand_indices: Optional[Iterable[int]] = DEFAULT_RIGHT_HAND_JOINT_INDICES,
    bow_weight: float = 8.0,
    bow_indices: Optional[Iterable[int]] = DEFAULT_BOW_JOINT_INDICES,
    fingertip_weight: float = 12.0,
    fingertip_indices: Optional[Iterable[int]] = DEFAULT_FINGERTIP_JOINT_INDICES,
    face_weight: Optional[float] = None,
    face_indices: Optional[Iterable[int]] = DEFAULT_FACE_JOINT_INDICES,
    lower_body_weight: Optional[float] = None,
    lower_body_indices: Optional[Iterable[int]] = DEFAULT_LOWER_BODY_JOINT_INDICES,
    torso_anchor_weight: Optional[float] = None,
    torso_anchor_indices: Optional[Iterable[int]] = DEFAULT_TORSO_ANCHOR_JOINT_INDICES,
    ignored_joint_weight: Optional[float] = 0.0,
    ignored_joint_indices: Optional[Iterable[int]] = DEFAULT_IGNORED_JOINT_INDICES,
) -> torch.Tensor:
    weights = torch.ones((joint_count,), device=device, dtype=dtype) * float(body_weight)
    if face_weight is not None:
        set_joint_weights(weights, face_indices, float(face_weight), joint_count)
    if lower_body_weight is not None:
        set_joint_weights(weights, lower_body_indices, float(lower_body_weight), joint_count)
    if torso_anchor_weight is not None:
        set_joint_weights(weights, torso_anchor_indices, float(torso_anchor_weight), joint_count)
    set_joint_weights(weights, left_hand_indices, float(left_hand_weight), joint_count)
    set_joint_weights(weights, right_hand_indices, float(right_hand_weight), joint_count)
    set_joint_weights(weights, bow_indices, float(bow_weight), joint_count)
    set_joint_weights(weights, fingertip_indices, float(fingertip_weight), joint_count)
    if ignored_joint_weight is not None:
        set_joint_weights(weights, ignored_joint_indices, float(ignored_joint_weight), joint_count)
    return weights


def make_joint_weight_grid(
    pose_grid: torch.Tensor,
    joint_count: int,
    pose_grid_size: Tuple[int, int],
    body_weight: float = 1.0,
    left_hand_weight: float = 6.0,
    left_hand_indices: Optional[Iterable[int]] = DEFAULT_LEFT_HAND_JOINT_INDICES,
    right_hand_weight: float = 3.0,
    right_hand_indices: Optional[Iterable[int]] = DEFAULT_RIGHT_HAND_JOINT_INDICES,
    bow_weight: float = 8.0,
    bow_indices: Optional[Iterable[int]] = DEFAULT_BOW_JOINT_INDICES,
    fingertip_weight: float = 12.0,
    fingertip_indices: Optional[Iterable[int]] = DEFAULT_FINGERTIP_JOINT_INDICES,
    face_weight: Optional[float] = None,
    face_indices: Optional[Iterable[int]] = DEFAULT_FACE_JOINT_INDICES,
    lower_body_weight: Optional[float] = None,
    lower_body_indices: Optional[Iterable[int]] = DEFAULT_LOWER_BODY_JOINT_INDICES,
    torso_anchor_weight: Optional[float] = None,
    torso_anchor_indices: Optional[Iterable[int]] = DEFAULT_TORSO_ANCHOR_JOINT_INDICES,
    ignored_joint_weight: Optional[float] = 0.0,
    ignored_joint_indices: Optional[Iterable[int]] = DEFAULT_IGNORED_JOINT_INDICES,
) -> torch.Tensor:
    h, w = pose_grid_size
    weights = make_joint_weight_vector(
        joint_count,
        device=pose_grid.device,
        dtype=pose_grid.dtype,
        body_weight=body_weight,
        left_hand_weight=left_hand_weight,
        left_hand_indices=left_hand_indices,
        right_hand_weight=right_hand_weight,
        right_hand_indices=right_hand_indices,
        bow_weight=bow_weight,
        bow_indices=bow_indices,
        fingertip_weight=fingertip_weight,
        fingertip_indices=fingertip_indices,
        face_weight=face_weight,
        face_indices=face_indices,
        lower_body_weight=lower_body_weight,
        lower_body_indices=lower_body_indices,
        torso_anchor_weight=torso_anchor_weight,
        torso_anchor_indices=torso_anchor_indices,
        ignored_joint_weight=ignored_joint_weight,
        ignored_joint_indices=ignored_joint_indices,
    )

    flat = torch.zeros((h * w,), device=pose_grid.device, dtype=pose_grid.dtype)
    flat[:joint_count] = weights
    grid = flat.reshape(1, 1, h, w).repeat(pose_grid.shape[0], pose_grid.shape[2], 1, 1)
    return grid.unsqueeze(1).repeat(1, pose_grid.shape[1], 1, 1, 1)


def make_joint_weight_vector_batch(
    joint_counts: torch.Tensor,
    max_joint_count: int,
    device,
    dtype,
    body_weight: float = 1.0,
    left_hand_weight: float = 6.0,
    left_hand_indices: Optional[Iterable[int]] = DEFAULT_LEFT_HAND_JOINT_INDICES,
    right_hand_weight: float = 3.0,
    right_hand_indices: Optional[Iterable[int]] = DEFAULT_RIGHT_HAND_JOINT_INDICES,
    bow_weight: float = 8.0,
    bow_indices: Optional[Iterable[int]] = DEFAULT_BOW_JOINT_INDICES,
    fingertip_weight: float = 12.0,
    fingertip_indices: Optional[Iterable[int]] = DEFAULT_FINGERTIP_JOINT_INDICES,
    face_weight: Optional[float] = None,
    face_indices: Optional[Iterable[int]] = DEFAULT_FACE_JOINT_INDICES,
    lower_body_weight: Optional[float] = None,
    lower_body_indices: Optional[Iterable[int]] = DEFAULT_LOWER_BODY_JOINT_INDICES,
    torso_anchor_weight: Optional[float] = None,
    torso_anchor_indices: Optional[Iterable[int]] = DEFAULT_TORSO_ANCHOR_JOINT_INDICES,
    ignored_joint_weight: Optional[float] = 0.0,
    ignored_joint_indices: Optional[Iterable[int]] = DEFAULT_IGNORED_JOINT_INDICES,
) -> torch.Tensor:
    weights = torch.zeros((joint_counts.shape[0], max_joint_count), device=device, dtype=dtype)
    for idx in range(joint_counts.shape[0]):
        count = min(int(joint_counts[idx].item()), int(max_joint_count))
        if count <= 0:
            continue
        weights[idx, :count] = make_joint_weight_vector(
            count,
            device=device,
            dtype=dtype,
            body_weight=body_weight,
            left_hand_weight=left_hand_weight,
            left_hand_indices=left_hand_indices,
            right_hand_weight=right_hand_weight,
            right_hand_indices=right_hand_indices,
            bow_weight=bow_weight,
            bow_indices=bow_indices,
            fingertip_weight=fingertip_weight,
            fingertip_indices=fingertip_indices,
            face_weight=face_weight,
            face_indices=face_indices,
            lower_body_weight=lower_body_weight,
            lower_body_indices=lower_body_indices,
            torso_anchor_weight=torso_anchor_weight,
            torso_anchor_indices=torso_anchor_indices,
            ignored_joint_weight=ignored_joint_weight,
            ignored_joint_indices=ignored_joint_indices,
        )
    return weights


def make_joint_weight_grid_batch(
    pose_grid: torch.Tensor,
    joint_counts: torch.Tensor,
    pose_grid_size: Tuple[int, int],
    body_weight: float = 1.0,
    left_hand_weight: float = 6.0,
    left_hand_indices: Optional[Iterable[int]] = DEFAULT_LEFT_HAND_JOINT_INDICES,
    right_hand_weight: float = 3.0,
    right_hand_indices: Optional[Iterable[int]] = DEFAULT_RIGHT_HAND_JOINT_INDICES,
    bow_weight: float = 8.0,
    bow_indices: Optional[Iterable[int]] = DEFAULT_BOW_JOINT_INDICES,
    fingertip_weight: float = 12.0,
    fingertip_indices: Optional[Iterable[int]] = DEFAULT_FINGERTIP_JOINT_INDICES,
    face_weight: Optional[float] = None,
    face_indices: Optional[Iterable[int]] = DEFAULT_FACE_JOINT_INDICES,
    lower_body_weight: Optional[float] = None,
    lower_body_indices: Optional[Iterable[int]] = DEFAULT_LOWER_BODY_JOINT_INDICES,
    torso_anchor_weight: Optional[float] = None,
    torso_anchor_indices: Optional[Iterable[int]] = DEFAULT_TORSO_ANCHOR_JOINT_INDICES,
    ignored_joint_weight: Optional[float] = 0.0,
    ignored_joint_indices: Optional[Iterable[int]] = DEFAULT_IGNORED_JOINT_INDICES,
) -> torch.Tensor:
    grids = []
    for idx in range(pose_grid.shape[0]):
        grids.append(
            make_joint_weight_grid(
                pose_grid[idx : idx + 1],
                int(joint_counts[idx].item()),
                pose_grid_size,
                body_weight=body_weight,
                left_hand_weight=left_hand_weight,
                left_hand_indices=left_hand_indices,
                right_hand_weight=right_hand_weight,
                right_hand_indices=right_hand_indices,
                bow_weight=bow_weight,
                bow_indices=bow_indices,
                fingertip_weight=fingertip_weight,
                fingertip_indices=fingertip_indices,
                face_weight=face_weight,
                face_indices=face_indices,
                lower_body_weight=lower_body_weight,
                lower_body_indices=lower_body_indices,
                torso_anchor_weight=torso_anchor_weight,
                torso_anchor_indices=torso_anchor_indices,
                ignored_joint_weight=ignored_joint_weight,
                ignored_joint_indices=ignored_joint_indices,
            )
        )
    return torch.cat(grids, dim=0)


def unpack_pose_grid_torch(pose_grid: torch.Tensor, joint_count: int, pose_grid_size: Tuple[int, int]) -> torch.Tensor:
    h, w = pose_grid_size
    pose = pose_grid.permute(0, 2, 3, 4, 1).reshape(pose_grid.shape[0], pose_grid.shape[2], h * w, 3)
    return pose[:, :, :joint_count, :]


def weighted_masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    joint_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    weight = mask.float()
    if joint_weights is not None:
        weight = weight * joint_weights.float()
    loss = (pred.float() - target.float()) ** 2 * weight
    return loss.sum() / weight.sum().clamp_min(1.0)


def predict_x0_from_model_output(
    scheduler,
    noisy_pose: torch.Tensor,
    model_pred: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    alphas_cumprod = scheduler.alphas_cumprod.to(device=noisy_pose.device, dtype=noisy_pose.dtype)
    alpha_t = alphas_cumprod[timesteps].reshape(-1, 1, 1, 1, 1)
    beta_t = (1.0 - alpha_t).clamp_min(1e-8)
    if scheduler.config.prediction_type == "epsilon":
        pred_x0 = (noisy_pose - beta_t.sqrt() * model_pred) / alpha_t.sqrt().clamp_min(1e-8)
    elif scheduler.config.prediction_type == "v_prediction":
        pred_x0 = alpha_t.sqrt() * noisy_pose - beta_t.sqrt() * model_pred
    else:
        raise ValueError(f"Unknown prediction type {scheduler.config.prediction_type}")
    return pred_x0


def temporal_velocity_loss(pred_pose: torch.Tensor, target_pose: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if pred_pose.shape[1] < 2:
        return pred_pose.new_tensor(0.0)
    pred_vel = pred_pose[:, 1:] - pred_pose[:, :-1]
    target_vel = target_pose[:, 1:] - target_pose[:, :-1]
    vel_valid = (valid[:, 1:] * valid[:, :-1]).unsqueeze(-1)
    loss = (pred_vel.float() - target_vel.float()) ** 2 * vel_valid.float()
    return loss.sum() / vel_valid.sum().clamp_min(1.0)


def temporal_velocity_loss_weighted(
    pred_pose: torch.Tensor,
    target_pose: torch.Tensor,
    valid: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
    joint_weight: Optional[torch.Tensor] = None,
    predicted_frames: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if pred_pose.shape[1] < 2:
        return pred_pose.new_tensor(0.0)
    pred_vel = pred_pose[:, 1:] - pred_pose[:, :-1]
    target_vel = target_pose[:, 1:] - target_pose[:, :-1]
    vel_valid = (valid[:, 1:] * valid[:, :-1]).unsqueeze(-1).float()
    if predicted_frames is not None:
        affected = predicted_frames[:, 1:] | predicted_frames[:, :-1]
        vel_valid = vel_valid * affected[:, :, None, None]
    if sample_weight is not None:
        vel_valid = vel_valid * sample_weight.reshape(-1, 1, 1, 1).float()
    if joint_weight is not None:
        vel_valid = vel_valid * joint_weight[:, None, :, None].float()
    loss = (pred_vel.float() - target_vel.float()) ** 2 * vel_valid
    return loss.sum() / vel_valid.sum().clamp_min(1.0)


def temporal_acceleration_loss(pred_pose: torch.Tensor, target_pose: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if pred_pose.shape[1] < 3:
        return pred_pose.new_tensor(0.0)
    pred_acc = pred_pose[:, 2:] - 2.0 * pred_pose[:, 1:-1] + pred_pose[:, :-2]
    target_acc = target_pose[:, 2:] - 2.0 * target_pose[:, 1:-1] + target_pose[:, :-2]
    acc_valid = (valid[:, 2:] * valid[:, 1:-1] * valid[:, :-2]).unsqueeze(-1)
    loss = (pred_acc.float() - target_acc.float()) ** 2 * acc_valid.float()
    return loss.sum() / acc_valid.sum().clamp_min(1.0)


def temporal_acceleration_loss_weighted(
    pred_pose: torch.Tensor,
    target_pose: torch.Tensor,
    valid: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
    joint_weight: Optional[torch.Tensor] = None,
    predicted_frames: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if pred_pose.shape[1] < 3:
        return pred_pose.new_tensor(0.0)
    pred_acc = pred_pose[:, 2:] - 2.0 * pred_pose[:, 1:-1] + pred_pose[:, :-2]
    target_acc = target_pose[:, 2:] - 2.0 * target_pose[:, 1:-1] + target_pose[:, :-2]
    acc_valid = (valid[:, 2:] * valid[:, 1:-1] * valid[:, :-2]).unsqueeze(-1).float()
    if predicted_frames is not None:
        affected = predicted_frames[:, 2:] | predicted_frames[:, 1:-1] | predicted_frames[:, :-2]
        acc_valid = acc_valid * affected[:, :, None, None]
    if sample_weight is not None:
        acc_valid = acc_valid * sample_weight.reshape(-1, 1, 1, 1).float()
    if joint_weight is not None:
        acc_valid = acc_valid * joint_weight[:, None, :, None].float()
    loss = (pred_acc.float() - target_acc.float()) ** 2 * acc_valid
    return loss.sum() / acc_valid.sum().clamp_min(1.0)


def bow_radial_velocity_loss_weighted(
    pred_pose: torch.Tensor,
    target_pose: torch.Tensor,
    strings: torch.Tensor,
    bow_valid: torch.Tensor,
    target_valid: Optional[torch.Tensor] = None,
    sample_weight: Optional[torch.Tensor] = None,
    bow_endpoint_index: int = 133,
    beta: float = 0.01,
    velocity_scale: float = 1.0,
) -> torch.Tensor:
    """Match radial velocity; scale=1 preserves the legacy objective units."""
    return bow_radial_velocity_loss(
        pred_pose, target_pose, strings, bow_valid,
        target_valid=target_valid, sample_weight=sample_weight,
        bow_endpoint_index=bow_endpoint_index, beta=beta, velocity_scale=velocity_scale,
    )


def bow_radial_velocity_direction_loss_weighted(
    pred_pose: torch.Tensor,
    target_pose: torch.Tensor,
    strings: torch.Tensor,
    bow_valid: torch.Tensor,
    target_valid: Optional[torch.Tensor] = None,
    sample_weight: Optional[torch.Tensor] = None,
    bow_endpoint_index: int = 133,
    temperature: float = 0.01,
    min_velocity: float = 0.001,
) -> torch.Tensor:
    """Classify the up/down bowing direction from bridge-distance velocity."""
    if pred_pose.shape[1] < 2 or pred_pose.shape[2] <= bow_endpoint_index:
        return pred_pose.new_tensor(0.0)
    if strings.ndim != 5 or strings.shape[-2:] != (2, 3):
        return pred_pose.new_tensor(0.0)

    device = pred_pose.device
    strings = strings.to(device=device, dtype=pred_pose.dtype)
    bow_valid = bow_valid.to(device=device).bool()

    bridge_points = strings[:, :, :, 1, :]
    bridge_point_valid = torch.isfinite(bridge_points).all(dim=-1)
    bridge_sum = torch.where(
        bridge_point_valid.unsqueeze(-1),
        bridge_points,
        torch.zeros_like(bridge_points),
    ).sum(dim=2)
    bridge_count = bridge_point_valid.sum(dim=2)
    bridge_anchor = bridge_sum / bridge_count.clamp_min(1).unsqueeze(-1).to(bridge_sum.dtype)

    pred_endpoint = pred_pose[:, :, bow_endpoint_index, :]
    target_endpoint = target_pose[:, :, bow_endpoint_index, :].to(
        device=device,
        dtype=pred_pose.dtype,
    )
    frame_valid = (
        bow_valid
        & (bridge_count > 0)
        & torch.isfinite(pred_endpoint).all(dim=-1)
        & torch.isfinite(target_endpoint).all(dim=-1)
        & torch.isfinite(bridge_anchor).all(dim=-1)
    )
    if target_valid is not None and target_valid.shape[2] > bow_endpoint_index:
        frame_valid = frame_valid & target_valid[:, :, bow_endpoint_index].to(device=device).bool()

    pred_distance = torch.linalg.norm(pred_endpoint.float() - bridge_anchor.float(), dim=-1)
    target_distance = torch.linalg.norm(target_endpoint.float() - bridge_anchor.float(), dim=-1)
    pred_velocity = pred_distance[:, 1:] - pred_distance[:, :-1]
    target_velocity = target_distance[:, 1:] - target_distance[:, :-1]

    velocity_valid = frame_valid[:, 1:] & frame_valid[:, :-1]
    velocity_valid = velocity_valid & (target_velocity.abs() >= float(min_velocity))
    if not velocity_valid.any():
        return pred_pose.new_tensor(0.0)

    target_up = (target_velocity > 0.0).to(dtype=torch.float32)
    logits = pred_velocity / max(float(temperature), 1e-8)
    loss = F.binary_cross_entropy_with_logits(logits, target_up, reduction="none")
    velocity_weight = velocity_valid.to(dtype=torch.float32)
    if sample_weight is not None:
        velocity_weight = velocity_weight * sample_weight.to(device=device, dtype=torch.float32).reshape(-1, 1)
    return (loss * velocity_weight).sum() / velocity_weight.sum().clamp_min(1.0)


def bow_endpoint_smoothness_loss_weighted(
    pred_pose: torch.Tensor,
    target_pose: torch.Tensor,
    strings: torch.Tensor,
    bow_valid: torch.Tensor,
    target_valid: Optional[torch.Tensor] = None,
    sample_weight: Optional[torch.Tensor] = None,
    bow_endpoint_indices: Tuple[int, ...] = (133, 134),
    window: int = 5,
    exclude_radius: int = 3,
    min_velocity: float = 0.001,
) -> torch.Tensor:
    """Suppress endpoint jitter away from confident GT bow reversals.

    The objective is the endpoint high-frequency residual relative to a
    centered moving average. It is applied to both bow endpoints. Frames near
    GT radial-velocity sign changes are excluded so genuine reversals are not
    flattened; ``min_velocity`` only controls which GT sign changes are
    considered confident enough to create an exclusion window.
    """
    window = int(window)
    exclude_radius = max(int(exclude_radius), 0)
    min_velocity = max(float(min_velocity), 0.0)
    if window <= 1:
        return pred_pose.new_tensor(0.0)
    if window % 2 == 0:
        raise ValueError("bow_endpoint_smoothness_window must be a positive odd value.")
    endpoint_indices = tuple(
        int(index)
        for index in bow_endpoint_indices
        if 0 <= int(index) < pred_pose.shape[2]
    )
    if not endpoint_indices or pred_pose.shape[1] < window:
        return pred_pose.new_tensor(0.0)
    if strings.ndim != 5 or strings.shape[-2:] != (2, 3):
        return pred_pose.new_tensor(0.0)

    device = pred_pose.device
    endpoints = pred_pose[:, :, endpoint_indices, :].float()
    target_endpoints = target_pose[:, :, endpoint_indices, :].to(
        device=device,
        dtype=torch.float32,
    )
    bow_valid = bow_valid.to(device=device).bool()
    endpoint_valid = (
        bow_valid.unsqueeze(-1)
        & torch.isfinite(endpoints).all(dim=-1)
        & torch.isfinite(target_endpoints).all(dim=-1)
    )
    if target_valid is not None:
        endpoint_valid = endpoint_valid & target_valid[:, :, endpoint_indices].to(
            device=device
        ).bool()

    radius = window // 2
    batch_size, frame_count, endpoint_count, _ = endpoints.shape
    safe_endpoints = torch.where(
        endpoint_valid.unsqueeze(-1),
        endpoints,
        torch.zeros_like(endpoints),
    )
    endpoint_channels = safe_endpoints.permute(0, 2, 3, 1).reshape(
        batch_size,
        endpoint_count * 3,
        frame_count,
    )
    smoothed_channels = F.avg_pool1d(
        F.pad(endpoint_channels, (radius, radius), mode="replicate"),
        kernel_size=window,
        stride=1,
    )
    high_frequency = (
        endpoint_channels - smoothed_channels
    ).reshape(batch_size, endpoint_count, 3, frame_count).permute(0, 3, 1, 2)

    # A valid loss frame needs a complete valid smoothing neighbourhood.
    valid_channels = endpoint_valid.permute(0, 2, 1).reshape(
        batch_size * endpoint_count,
        1,
        frame_count,
    ).float()
    valid_window_count = F.conv1d(
        valid_channels,
        torch.ones((1, 1, window), device=device, dtype=torch.float32),
        padding=radius,
    )
    smooth_valid = (valid_window_count >= float(window)).reshape(
        batch_size,
        endpoint_count,
        frame_count,
    ).permute(0, 2, 1)

    strings = strings.to(device=device, dtype=pred_pose.dtype)
    bridge_points = strings[:, :, :, 1, :]
    bridge_point_valid = torch.isfinite(bridge_points).all(dim=-1)
    bridge_sum = torch.where(
        bridge_point_valid.unsqueeze(-1),
        bridge_points,
        torch.zeros_like(bridge_points),
    ).sum(dim=2)
    bridge_count = bridge_point_valid.sum(dim=2)
    bridge_anchor = bridge_sum / bridge_count.clamp_min(1).unsqueeze(-1).to(
        bridge_sum.dtype
    )
    target_reference = target_endpoints[:, :, 0, :]
    radial_valid = (
        endpoint_valid[:, :, 0]
        & (bridge_count > 0)
        & torch.isfinite(bridge_anchor).all(dim=-1)
    )
    target_distance = torch.linalg.norm(
        target_reference - bridge_anchor.float(),
        dim=-1,
    )
    target_velocity = target_distance[:, 1:] - target_distance[:, :-1]
    pair_valid = radial_valid[:, 1:] & radial_valid[:, :-1]
    moving = pair_valid & (target_velocity.abs() >= min_velocity)
    reversals = (
        moving[:, 1:]
        & moving[:, :-1]
        & (target_velocity[:, 1:] * target_velocity[:, :-1] < 0.0)
    )
    reversal_frames = torch.zeros(
        (batch_size, frame_count),
        device=device,
        dtype=torch.float32,
    )
    reversal_frames[:, 1:-1] = reversals.float()
    if exclude_radius > 0:
        reversal_frames = F.max_pool1d(
            reversal_frames.unsqueeze(1),
            kernel_size=2 * exclude_radius + 1,
            stride=1,
            padding=exclude_radius,
        ).squeeze(1)
    stable_frame = reversal_frames < 0.5

    loss_weight = smooth_valid & stable_frame.unsqueeze(-1)
    if sample_weight is not None:
        loss_weight = loss_weight.float() * sample_weight.to(
            device=device,
            dtype=torch.float32,
        ).reshape(-1, 1, 1)
    else:
        loss_weight = loss_weight.float()
    coordinate_weight = loss_weight.unsqueeze(-1)
    loss = high_frequency.square() * coordinate_weight
    return loss.sum() / coordinate_weight.sum().clamp_min(1.0)


def bow_endpoint_radial_stable_sign_loss_weighted(
    pred_pose: torch.Tensor,
    target_pose: torch.Tensor,
    strings: torch.Tensor,
    bow_valid: torch.Tensor,
    target_valid: Optional[torch.Tensor] = None,
    sample_weight: Optional[torch.Tensor] = None,
    bow_endpoint_index: int = 133,
    stable_window: int = 3,
    exclude_radius: int = 3,
    min_velocity: float = 0.001,
    margin: float = 0.0,
) -> torch.Tensor:
    """Penalize wrong radial direction only in stable GT bowing segments.

    Unlike radial-velocity regression or BCE direction supervision, the zero
    margin version only targets false sign reversals. A positive margin adds a
    small cost to zero or very small correctly-directed velocities, which
    prevents the model from avoiding the penalty by collapsing radial speed.
    """
    stable_window = int(stable_window)
    exclude_radius = max(int(exclude_radius), 0)
    min_velocity = max(float(min_velocity), 1e-6)
    margin = max(float(margin), 0.0)
    if stable_window <= 0 or stable_window % 2 == 0:
        raise ValueError("bow_endpoint_radial_stable_sign_window must be positive and odd.")
    if pred_pose.shape[1] < 3 or pred_pose.shape[2] <= bow_endpoint_index:
        return pred_pose.new_tensor(0.0)
    if strings.ndim != 5 or strings.shape[-2:] != (2, 3):
        return pred_pose.new_tensor(0.0)

    device = pred_pose.device
    strings = strings.to(device=device, dtype=pred_pose.dtype)
    bow_valid = bow_valid.to(device=device).bool()
    pred_endpoint = pred_pose[:, :, bow_endpoint_index, :].float()
    target_endpoint = target_pose[:, :, bow_endpoint_index, :].to(
        device=device,
        dtype=torch.float32,
    )

    bridge_points = strings[:, :, :, 1, :]
    bridge_point_valid = torch.isfinite(bridge_points).all(dim=-1)
    bridge_sum = torch.where(
        bridge_point_valid.unsqueeze(-1),
        bridge_points,
        torch.zeros_like(bridge_points),
    ).sum(dim=2)
    bridge_count = bridge_point_valid.sum(dim=2)
    bridge_anchor = bridge_sum / bridge_count.clamp_min(1).unsqueeze(-1).to(
        bridge_sum.dtype
    )

    frame_valid = (
        bow_valid
        & (bridge_count > 0)
        & torch.isfinite(pred_endpoint).all(dim=-1)
        & torch.isfinite(target_endpoint).all(dim=-1)
        & torch.isfinite(bridge_anchor).all(dim=-1)
    )
    if target_valid is not None and target_valid.shape[2] > bow_endpoint_index:
        frame_valid = frame_valid & target_valid[:, :, bow_endpoint_index].to(
            device=device
        ).bool()

    pred_distance = torch.linalg.norm(pred_endpoint - bridge_anchor.float(), dim=-1)
    target_distance = torch.linalg.norm(target_endpoint - bridge_anchor.float(), dim=-1)
    pred_velocity = pred_distance[:, 1:] - pred_distance[:, :-1]
    target_velocity = target_distance[:, 1:] - target_distance[:, :-1]
    pair_valid = frame_valid[:, 1:] & frame_valid[:, :-1]
    moving = pair_valid & (target_velocity.abs() >= min_velocity)

    # Mark GT reversals and exclude a temporal neighbourhood around them.
    reversals = (
        moving[:, 1:]
        & moving[:, :-1]
        & (target_velocity[:, 1:] * target_velocity[:, :-1] < 0.0)
    )
    batch_size, frame_count = frame_valid.shape
    reversal_frame = torch.zeros(
        (batch_size, frame_count), device=device, dtype=torch.float32
    )
    reversal_frame[:, 1:-1] = reversals.float()
    if exclude_radius > 0:
        reversal_frame = F.max_pool1d(
            reversal_frame.unsqueeze(1),
            kernel_size=2 * exclude_radius + 1,
            stride=1,
            padding=exclude_radius,
        ).squeeze(1)
    near_reversal_pair = reversal_frame[:, 1:] > 0.5
    near_reversal_pair = near_reversal_pair | (reversal_frame[:, :-1] > 0.5)

    # Require a full local neighbourhood of the same GT direction. This
    # makes the loss insensitive to isolated low-speed annotation changes.
    radius = stable_window // 2
    moving_values = moving.float().unsqueeze(1)
    sign_values = torch.sign(target_velocity).float() * moving.float()
    kernel = torch.ones((1, 1, stable_window), device=device, dtype=torch.float32)
    moving_count = F.conv1d(moving_values, kernel, padding=radius).squeeze(1)
    sign_sum = F.conv1d(
        sign_values.unsqueeze(1), kernel, padding=radius
    ).squeeze(1)
    stable_sign_pair = (
        moving
        & ~near_reversal_pair
        & (moving_count >= float(stable_window))
        & (sign_sum.abs() >= float(stable_window) - 1e-4)
    )
    if not stable_sign_pair.any():
        return pred_pose.new_tensor(0.0)

    target_sign = torch.sign(target_velocity)
    signed_pred_velocity = target_sign * pred_velocity
    # Normalize by the GT minimum speed so the small coordinate units do not
    # make this auxiliary loss numerically irrelevant. With margin=0, only
    # wrong-sign velocities are penalized. A positive margin also penalizes
    # zero or very small velocities in stable moving segments.
    loss = F.relu(float(margin) - signed_pred_velocity) / min_velocity
    weight = stable_sign_pair.float()
    if sample_weight is not None:
        weight = weight * sample_weight.to(
            device=device, dtype=torch.float32
        ).reshape(-1, 1)
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def bow_reversal_event_direction_loss_weighted(
    pred_pose: torch.Tensor,
    target_pose: torch.Tensor,
    relation: torch.Tensor,
    relation_mask: torch.Tensor,
    strings: torch.Tensor,
    bow_valid: torch.Tensor,
    target_valid: Optional[torch.Tensor] = None,
    sample_weight: Optional[torch.Tensor] = None,
    bow_endpoint_index: int = 133,
    temperature: float = 0.01,
    target_min_velocity: float = 0.001,
    relation_min_velocity: float = 0.005,
    relation_tolerance_frames: int = 2,
) -> torch.Tensor:
    """Supervise pose direction around GT reversals supported by relation events."""
    if pred_pose.shape[1] < 3 or pred_pose.shape[2] <= bow_endpoint_index:
        return pred_pose.new_tensor(0.0)
    if relation.ndim != 3 or relation.shape[-1] <= 11:
        return pred_pose.new_tensor(0.0)
    if strings.ndim != 5 or strings.shape[-2:] != (2, 3):
        return pred_pose.new_tensor(0.0)

    device = pred_pose.device
    strings = strings.to(device=device, dtype=pred_pose.dtype)
    relation = relation.to(device=device, dtype=torch.float32)
    relation_mask = relation_mask.to(device=device).bool()
    bow_valid = bow_valid.to(device=device).bool()

    bridge_points = strings[:, :, :, 1, :]
    bridge_point_valid = torch.isfinite(bridge_points).all(dim=-1)
    bridge_sum = torch.where(
        bridge_point_valid.unsqueeze(-1),
        bridge_points,
        torch.zeros_like(bridge_points),
    ).sum(dim=2)
    bridge_count = bridge_point_valid.sum(dim=2)
    bridge_anchor = bridge_sum / bridge_count.clamp_min(1).unsqueeze(-1).to(bridge_sum.dtype)

    pred_endpoint = pred_pose[:, :, bow_endpoint_index, :]
    target_endpoint = target_pose[:, :, bow_endpoint_index, :].to(
        device=device, dtype=pred_pose.dtype
    )
    frame_valid = (
        bow_valid
        & (bridge_count > 0)
        & torch.isfinite(pred_endpoint).all(dim=-1)
        & torch.isfinite(target_endpoint).all(dim=-1)
        & torch.isfinite(bridge_anchor).all(dim=-1)
    )
    if target_valid is not None and target_valid.shape[2] > bow_endpoint_index:
        frame_valid = frame_valid & target_valid[:, :, bow_endpoint_index].to(device=device).bool()

    pred_distance = torch.linalg.norm(pred_endpoint.float() - bridge_anchor.float(), dim=-1)
    target_distance = torch.linalg.norm(target_endpoint.float() - bridge_anchor.float(), dim=-1)
    pred_velocity = pred_distance[:, 1:] - pred_distance[:, :-1]
    target_velocity = target_distance[:, 1:] - target_distance[:, :-1]
    relation_velocity = relation[:, 1:, 11] - relation[:, :-1, 11]

    pose_pair_valid = frame_valid[:, 1:] & frame_valid[:, :-1]
    relation_pair_valid = (
        bow_valid[:, 1:]
        & bow_valid[:, :-1]
        & relation_mask[:, 1:, 11]
        & relation_mask[:, :-1, 11]
    )
    target_moving = pose_pair_valid & (target_velocity.abs() >= float(target_min_velocity))
    relation_moving = relation_pair_valid & (
        relation_velocity.abs() >= float(relation_min_velocity)
    )
    target_events = (
        target_moving[:, 1:]
        & target_moving[:, :-1]
        & (target_velocity[:, 1:] * target_velocity[:, :-1] < 0.0)
    )
    relation_events = (
        relation_moving[:, 1:]
        & relation_moving[:, :-1]
        & (relation_velocity[:, 1:] * relation_velocity[:, :-1] < 0.0)
    )
    tolerance = max(int(relation_tolerance_frames), 0)
    if tolerance > 0:
        relation_near = F.max_pool1d(
            relation_events.float().unsqueeze(1),
            kernel_size=2 * tolerance + 1,
            stride=1,
            padding=tolerance,
        ).squeeze(1).bool()
    else:
        relation_near = relation_events
    matched_events = target_events & relation_near
    if not matched_events.any():
        return pred_pose.new_tensor(0.0)

    velocity_weight = torch.zeros_like(pred_velocity, dtype=torch.float32)
    velocity_weight[:, :-1] += matched_events.float()
    velocity_weight[:, 1:] += matched_events.float()
    velocity_weight = velocity_weight.clamp_max(1.0) * pose_pair_valid.float()
    if sample_weight is not None:
        velocity_weight = velocity_weight * sample_weight.to(
            device=device, dtype=torch.float32
        ).reshape(-1, 1)
    target_up = (target_velocity > 0.0).float()
    logits = pred_velocity / max(float(temperature), 1e-8)
    loss = F.binary_cross_entropy_with_logits(logits, target_up, reduction="none")
    return (loss * velocity_weight).sum() / velocity_weight.sum().clamp_min(1.0)


def left_hand_local_loss(pred_pose: torch.Tensor, target_pose: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    joint_count = pred_pose.shape[2]
    if joint_count <= 111 or joint_count <= 91:
        return pred_pose.new_tensor(0.0)
    hand_slice = slice(91, min(112, joint_count))
    wrist = pred_pose[:, :, 91:92, :]
    target_wrist = target_pose[:, :, 91:92, :]
    pred_local = pred_pose[:, :, hand_slice, :] - wrist
    target_local = target_pose[:, :, hand_slice, :] - target_wrist
    local_valid = (valid[:, :, hand_slice] * valid[:, :, 91:92]).unsqueeze(-1)
    loss = (pred_local.float() - target_local.float()) ** 2 * local_valid.float()
    return loss.sum() / local_valid.sum().clamp_min(1.0)


def left_hand_local_loss_weighted(
    pred_pose: torch.Tensor,
    target_pose: torch.Tensor,
    valid: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
    wrist_index: int = 91,
    joint_indices: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    joint_count = pred_pose.shape[2]
    if joint_indices is None:
        if joint_count <= 111 or joint_count <= 91:
            return pred_pose.new_tensor(0.0)
        hand_slice = slice(91, min(112, joint_count))
    else:
        if not joint_indices or min(joint_indices) < 0 or max(joint_indices) >= joint_count or not 0 <= wrist_index < joint_count:
            raise ValueError("Invalid native hand landmark mapping.")
        hand_slice = list(joint_indices)
    wrist = pred_pose[:, :, wrist_index:wrist_index + 1, :]
    target_wrist = target_pose[:, :, wrist_index:wrist_index + 1, :]
    pred_local = pred_pose[:, :, hand_slice, :] - wrist
    target_local = target_pose[:, :, hand_slice, :] - target_wrist
    local_valid = (valid[:, :, hand_slice] * valid[:, :, wrist_index:wrist_index + 1]).unsqueeze(-1).float()
    if sample_weight is not None:
        local_valid = local_valid * sample_weight.reshape(-1, 1, 1, 1).float()
    loss = (pred_local.float() - target_local.float()) ** 2 * local_valid
    return loss.sum() / local_valid.sum().clamp_min(1.0)


def body_hand_wrist_offset_loss_weighted(
    pred_pose: torch.Tensor,
    target_pose: torch.Tensor,
    valid: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
    wrist_pairs: Tuple[Tuple[int, int], ...] = DEFAULT_BODY_HAND_WRIST_OFFSET_PAIRS,
) -> torch.Tensor:
    total = pred_pose.new_tensor(0.0)
    weight_total = pred_pose.new_tensor(0.0)
    joint_count = pred_pose.shape[2]
    for body_wrist_idx, hand_wrist_idx in wrist_pairs:
        if body_wrist_idx >= joint_count or hand_wrist_idx >= joint_count:
            continue
        pair_valid = (valid[:, :, body_wrist_idx] * valid[:, :, hand_wrist_idx]).unsqueeze(-1).float()
        if sample_weight is not None:
            pair_valid = pair_valid * sample_weight.reshape(-1, 1, 1).float()
        pred_offset = pred_pose[:, :, hand_wrist_idx, :] - pred_pose[:, :, body_wrist_idx, :]
        target_offset = target_pose[:, :, hand_wrist_idx, :] - target_pose[:, :, body_wrist_idx, :]
        total = total + (((pred_offset.float() - target_offset.float()) ** 2) * pair_valid).sum()
        weight_total = weight_total + pair_valid.sum()
    return total / weight_total.clamp_min(1.0)


def point_to_string_distance_all(points: torch.Tensor, strings: torch.Tensor) -> torch.Tensor:
    starts = strings[:, :, :, 0, :]
    ends = strings[:, :, :, 1, :]
    line = ends - starts
    line_len2 = torch.sum(line * line, dim=-1).clamp_min(1e-8)
    point_vec = points[:, :, None, :] - starts
    u = torch.sum(point_vec * line, dim=-1) / line_len2
    u = u.clamp(0.0, 1.0)
    closest = starts + u[..., None] * line
    return torch.linalg.norm(points[:, :, None, :] - closest, dim=-1)


def gather_target_string_segments(
    strings: torch.Tensor,
    target_points: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_dist_all = point_to_string_distance_all(target_points, strings)
    finite_dist = torch.isfinite(target_dist_all)
    masked_dist = target_dist_all.masked_fill(~finite_dist, torch.finfo(target_dist_all.dtype).max)
    target_string = torch.argmin(masked_dist, dim=-1)
    has_string = finite_dist.any(dim=-1)
    gather_index = target_string[:, :, None, None, None].expand(-1, -1, 1, 2, strings.shape[-1])
    selected = strings.gather(dim=2, index=gather_index).squeeze(2)
    return selected[:, :, 0, :], selected[:, :, 1, :], has_string


def line_to_line_distance_torch(
    p1: torch.Tensor,
    p2: torch.Tensor,
    q1: torch.Tensor,
    q2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    d1 = p2 - p1
    d2 = q2 - q1
    norm_d1 = torch.linalg.norm(d1, dim=-1)
    norm_d2 = torch.linalg.norm(d2, dim=-1)
    cross = torch.cross(d1, d2, dim=-1)
    norm_cross = torch.linalg.norm(cross, dim=-1)
    r = q1 - p1

    denom_cross = norm_cross.clamp_min(1e-8)
    non_parallel = torch.abs(torch.sum(r * cross, dim=-1)) / denom_cross
    parallel_cross = torch.cross(r, d1, dim=-1)
    parallel = torch.linalg.norm(parallel_cross, dim=-1) / norm_d1.clamp_min(1e-8)
    distance = torch.where(norm_cross > 1e-8, non_parallel, parallel)
    finite = (
        torch.isfinite(p1).all(dim=-1)
        & torch.isfinite(p2).all(dim=-1)
        & torch.isfinite(q1).all(dim=-1)
        & torch.isfinite(q2).all(dim=-1)
        & torch.isfinite(distance)
        & (norm_d1 > 1e-8)
        & (norm_d2 > 1e-8)
    )
    return distance, finite


def gather_contact_supervision_finger(
    pred_pose: torch.Tensor,
    contact_target: torch.Tensor,
    contact_fingertip_indices: Tuple[int, ...],
    target_pose: Optional[torch.Tensor] = None,
    target_valid: Optional[torch.Tensor] = None,
    selection: str = "gt",
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    joint_count = pred_pose.shape[2]
    available_tips = [idx for idx in contact_fingertip_indices if idx < joint_count]
    if not available_tips:
        return None, None

    pred_tips = pred_pose[:, :, available_tips, :]
    if selection == "gt" and target_pose is not None:
        source_tips = target_pose[:, :, available_tips, :].to(device=pred_pose.device, dtype=pred_pose.dtype)
        tip_contact_dist = torch.linalg.norm(source_tips - contact_target[:, :, None, :], dim=-1)
        if target_valid is not None:
            tip_valid = target_valid[:, :, available_tips].to(device=pred_pose.device).bool()
            tip_contact_dist = tip_contact_dist.masked_fill(~tip_valid, torch.finfo(tip_contact_dist.dtype).max)
            selected_valid = tip_valid.any(dim=-1)
        else:
            selected_valid = torch.ones_like(tip_contact_dist[..., 0], dtype=torch.bool)
    else:
        tip_contact_dist = torch.linalg.norm(pred_tips - contact_target[:, :, None, :], dim=-1)
        selected_valid = torch.ones_like(tip_contact_dist[..., 0], dtype=torch.bool)

    selected_tip = torch.argmin(tip_contact_dist, dim=-1)
    gathered_tip = pred_tips.gather(
        dim=2,
        index=selected_tip[:, :, None, None].expand(-1, -1, 1, pred_pose.shape[-1]),
    ).squeeze(2)
    return gathered_tip, selected_valid


def contact_string_ce_loss(
    pred_pose: torch.Tensor,
    batch: Dict,
    target_pose: Optional[torch.Tensor] = None,
    target_valid: Optional[torch.Tensor] = None,
    contact_fingertip_indices: Tuple[int, ...] = DEFAULT_CONTACT_FINGERTIP_INDICES,
    temperature: float = 0.02,
    contact_finger_supervision: str = "pred_min",
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    joint_count = pred_pose.shape[2]
    available_tips = [idx for idx in contact_fingertip_indices if idx < joint_count]
    if not available_tips or "strings" not in batch:
        return pred_pose.new_tensor(0.0)

    device = pred_pose.device
    dtype = pred_pose.dtype
    strings = batch["strings"].to(device=device, dtype=dtype)
    contact_target = batch["contact_target_point"].to(device=device, dtype=dtype)
    contact_valid = batch["contact_valid"].to(device=device, dtype=dtype)

    target_dist_all = point_to_string_distance_all(contact_target, strings)
    target_string = torch.argmin(target_dist_all, dim=-1)
    target_has_string = torch.isfinite(target_dist_all).all(dim=-1)

    gathered_tip, selected_valid = gather_contact_supervision_finger(
        pred_pose,
        contact_target,
        tuple(available_tips),
        target_pose=target_pose,
        target_valid=target_valid,
        selection=contact_finger_supervision,
    )
    if gathered_tip is None or selected_valid is None:
        return pred_pose.new_tensor(0.0)

    pred_dist_all = point_to_string_distance_all(gathered_tip, strings)
    logits = -pred_dist_all.float() / max(float(temperature), 1e-6)
    ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_string.reshape(-1),
        reduction="none",
    ).reshape_as(contact_valid)

    weight = contact_valid.float() * target_has_string.float() * selected_valid.float()
    if sample_weight is not None:
        weight = weight * sample_weight.to(device=device, dtype=torch.float32).reshape(-1, 1)
    return (ce * weight).sum() / weight.sum().clamp_min(1.0)


def playable_geometry_loss(
    pred_pose: torch.Tensor,
    batch: Dict,
    target_pose: Optional[torch.Tensor] = None,
    target_valid: Optional[torch.Tensor] = None,
    contact_fingertip_indices: Tuple[int, ...] = DEFAULT_CONTACT_FINGERTIP_INDICES,
    contact_loss_type: str = "min",
    contact_distance_loss: str = "l2",
    contact_finger_supervision: str = "pred_min",
    sample_weight: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    device = pred_pose.device
    contact_valid = batch["contact_valid"].to(device=device, dtype=pred_pose.dtype)
    bow_valid = batch["bow_valid"].to(device=device, dtype=pred_pose.dtype)
    contact_target = batch["contact_target_point"].to(device=device, dtype=pred_pose.dtype)
    bow_target = batch["bow_target_point"].to(device=device, dtype=pred_pose.dtype)
    bow_direction_target = batch["bow_target_direction"].to(device=device, dtype=pred_pose.dtype)
    if "bow_t_on_bow" in batch:
        bow_t_on_bow = batch["bow_t_on_bow"].to(device=device, dtype=pred_pose.dtype).clamp(0.0, 1.0)
    else:
        bow_t_on_bow = torch.full_like(bow_valid, 0.5)
    if sample_weight is not None:
        sample_weight = sample_weight.to(device=device, dtype=pred_pose.dtype).reshape(-1, 1)

    out = {}
    joint_count = pred_pose.shape[2]
    available_tips = [idx for idx in contact_fingertip_indices if idx < joint_count]
    if available_tips:
        use_gt_contact_finger = contact_finger_supervision == "gt" and target_pose is not None
        supervised_tip, selected_valid = gather_contact_supervision_finger(
            pred_pose,
            contact_target,
            tuple(available_tips),
            target_pose=target_pose,
            target_valid=target_valid,
            selection=contact_finger_supervision,
        )
        if supervised_tip is not None and selected_valid is not None and use_gt_contact_finger:
            contact_dist = torch.linalg.norm(supervised_tip - contact_target, dim=-1)
            contact_weight = contact_valid * selected_valid.to(device=device, dtype=contact_valid.dtype)
        else:
            tips = pred_pose[:, :, available_tips, :]
            dist = torch.linalg.norm(tips - contact_target[:, :, None, :], dim=-1)
            contact_weight = contact_valid
            if contact_loss_type == "all":
                contact_dist = dist.mean(dim=-1)
            else:
                contact_dist = dist.min(dim=-1).values
        if sample_weight is not None:
            contact_weight = contact_weight * sample_weight
        if contact_distance_loss == "l1":
            contact_objective = contact_dist
        elif contact_distance_loss == "l2":
            contact_objective = contact_dist ** 2
        else:
            raise ValueError("contact_distance_loss must be 'l1' or 'l2'.")
        out["contact"] = (contact_objective * contact_weight).sum() / contact_weight.sum().clamp_min(1.0)
    else:
        out["contact"] = pred_pose.new_tensor(0.0)

    if joint_count > 134:
        bow = pred_pose[:, :, 133:135, :]
        bow_contact = bow[:, :, 0, :] + bow_t_on_bow[:, :, None] * (bow[:, :, 1, :] - bow[:, :, 0, :])
        bow_mid_dist = torch.linalg.norm(bow_contact - bow_target, dim=-1)
        bow_weight = bow_valid
        if sample_weight is not None:
            bow_weight = bow_weight * sample_weight
        out["bow_position"] = ((bow_mid_dist ** 2) * bow_weight).sum() / bow_weight.sum().clamp_min(1.0)

        bow_dir = bow[:, :, 1, :] - bow[:, :, 0, :]
        bow_dir = F.normalize(bow_dir.float(), dim=-1, eps=1e-6)
        target_dir = F.normalize(bow_direction_target.float(), dim=-1, eps=1e-6)
        bow_dir_loss = 1.0 - torch.abs(torch.sum(bow_dir * target_dir, dim=-1))
        out["bow_direction"] = (bow_dir_loss * bow_weight).sum() / bow_weight.sum().clamp_min(1.0)

        if "strings" in batch:
            strings = batch["strings"].to(device=device, dtype=pred_pose.dtype)
            string_start, string_end, has_string = gather_target_string_segments(strings, contact_target)
            bow_string_dist, bow_string_finite = line_to_line_distance_torch(
                bow[:, :, 0, :],
                bow[:, :, 1, :],
                string_start,
                string_end,
            )
            bow_string_weight = bow_valid * contact_valid * has_string.to(device=device, dtype=bow_valid.dtype)
            bow_string_weight = bow_string_weight * bow_string_finite.to(device=device, dtype=bow_valid.dtype)
            if sample_weight is not None:
                bow_string_weight = bow_string_weight * sample_weight
            out["bow_string"] = ((bow_string_dist ** 2) * bow_string_weight).sum() / bow_string_weight.sum().clamp_min(1.0)
        else:
            out["bow_string"] = pred_pose.new_tensor(0.0)
    else:
        out["bow_position"] = pred_pose.new_tensor(0.0)
        out["bow_direction"] = pred_pose.new_tensor(0.0)
        out["bow_string"] = pred_pose.new_tensor(0.0)

    return out


def auxiliary_sample_weight(
    timesteps: torch.Tensor,
    num_train_timesteps: int,
    max_timestep_fraction: float = 0.25,
) -> torch.Tensor:
    if max_timestep_fraction <= 0:
        return torch.ones_like(timesteps, dtype=torch.float32)
    max_timestep = max(1, int(num_train_timesteps * float(max_timestep_fraction)))
    return (timesteps <= max_timestep).float()


def ramp_weight(global_step: int, start_step: int = 0, warmup_steps: int = 0) -> float:
    if global_step < int(start_step):
        return 0.0
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, float(global_step - int(start_step)) / float(warmup_steps))


def set_dataset_training_step(dataset, global_step: int) -> None:
    """Forward the optimizer step to optional step-aware training datasets."""
    setter = getattr(dataset, "set_training_step", None)
    if setter is not None:
        setter(int(global_step))


def _save_joint_mapping_checkpoint(path: str, mapping_model: Optional[torch.nn.Module]) -> None:
    if mapping_model is None:
        return
    unwrapped = mapping_model
    if hasattr(mapping_model, "module"):
        unwrapped = mapping_model.module
    state = getattr(unwrapped, "_mapping_checkpoint", {})
    metadata = dict(state) if isinstance(state, dict) else {}
    metadata["model"] = unwrapped.state_dict()
    metadata["joint_finetuned"] = True
    torch.save(metadata, os.path.join(path, "joint_mapping_model.pt"))


def restore_joint_training_state(accelerator, checkpoint_path, lr_sched, optimizer,
                                 scheduler_name, warmup_steps, accumulation_steps):
    """Restore training state, including legacy runs whose scheduler was not registered."""
    checkpoint_path = Path(checkpoint_path)
    step = int(checkpoint_path.name.split('-')[1])
    native = checkpoint_path / 'scheduler.bin'
    explicit = checkpoint_path / 'lr_scheduler.pt'
    reconstruct = not native.is_file() and not explicit.is_file()
    if reconstruct and (scheduler_name != 'constant' or int(warmup_steps) != 0
                        or int(accumulation_steps) != 1 or accelerator.num_processes != 1):
        raise ValueError('Missing scheduler state: reconstruction requires constant LR, zero warmup, accumulation=1 and one process')
    # Modern Accelerate may register a scheduler that the old environment did
    # not save. Skip only its native loader; all model/AdamW/RNG loading remains.
    registered = getattr(accelerator, '_schedulers', None)
    if not native.is_file() and registered is not None:
        accelerator._schedulers = []
    try:
        accelerator.load_state(str(checkpoint_path))
    finally:
        if registered is not None:
            accelerator._schedulers = registered
    if explicit.is_file():
        payload = torch.load(explicit, map_location='cpu')
        if int(payload['global_step']) != step:
            raise ValueError('Scheduler sidecar step does not match checkpoint directory')
        lr_sched.load_state_dict(payload['state_dict'])
        logger.info('Restored explicit scheduler state at step %s', step)
    elif native.is_file():
        # Also restore manually if the old runtime did not register lr_sched.
        lr_sched.load_state_dict(torch.load(native, map_location='cpu'))
    else:
        state = lr_sched.state_dict()
        rates = [float(group['lr']) for group in optimizer.param_groups]
        if len(rates) != len(state['base_lrs']) or any(
                not math.isclose(rate, float(base), rel_tol=1e-6, abs_tol=1e-12)
                for rate, base in zip(rates, state['base_lrs'])):
            raise ValueError('Restored AdamW learning rates do not match the original constant-LR recipe')
        state.update(last_epoch=step, _step_count=step+1, _last_lr=rates)
        lr_sched.load_state_dict(state)
        logger.info('Reconstructed constant-LR scheduler at step %s, restored AdamW LRs=%s (not a saved scheduler state)', step, rates)


def save_checkpoint(output_dir, global_step, accelerator, condition_encoder, mapping_model=None, geometry_refiner=None, lr_sched=None):
    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
    accelerator.save_state(save_path)
    if accelerator.is_main_process:
        if lr_sched is not None:
            torch.save(dict(global_step=int(global_step), state_dict=lr_sched.state_dict()),
                       os.path.join(save_path, 'lr_scheduler.pt'))
        unwrapped_encoder = accelerator.unwrap_model(condition_encoder)
        torch.save(unwrapped_encoder.state_dict(), os.path.join(save_path, "condition_encoder.pt"))
        _save_joint_mapping_checkpoint(
            save_path,
            accelerator.unwrap_model(mapping_model) if mapping_model is not None else None,
        )
        if geometry_refiner is not None:
            from audio2pose.training.joint_geometry import save_refiner
            save_refiner(save_path,accelerator.unwrap_model(geometry_refiner),global_step,Path(output_dir)/"config.yaml")
        logger.info(f"Saved state to {save_path}")


def save_best_checkpoint(
    output_dir,
    global_step,
    accelerator,
    condition_encoder,
    metadata: Dict,
    mapping_model=None,
    geometry_refiner=None,
    lr_sched=None,
):
    save_path = os.path.join(output_dir, "best-checkpoint")
    if accelerator.is_main_process and os.path.isdir(save_path):
        shutil.rmtree(save_path)
    accelerator.wait_for_everyone()
    accelerator.save_state(save_path)
    if accelerator.is_main_process:
        if lr_sched is not None:
            torch.save(dict(global_step=int(global_step), state_dict=lr_sched.state_dict()),
                       os.path.join(save_path, 'lr_scheduler.pt'))
        unwrapped_encoder = accelerator.unwrap_model(condition_encoder)
        torch.save(unwrapped_encoder.state_dict(), os.path.join(save_path, "condition_encoder.pt"))
        _save_joint_mapping_checkpoint(
            save_path,
            accelerator.unwrap_model(mapping_model) if mapping_model is not None else None,
        )
        if geometry_refiner is not None:
            from audio2pose.training.joint_geometry import save_refiner
            save_refiner(save_path,accelerator.unwrap_model(geometry_refiner),global_step,Path(output_dir)/"config.yaml")
        with open(os.path.join(save_path, "best_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        logger.info(f"Saved best checkpoint to {save_path}")


def load_pretrained_stage2_weights(
    checkpoint_dir: str,
    *,
    unet: torch.nn.Module,
    condition_encoder: torch.nn.Module,
    condition_dim: int,
    audio_condition_dim: int,
    relation_condition_dim: int,
    pose_channels: int,
    pose_size: int,
    allow_new_pose_context: bool = False,
    expected_train_data: Optional[Dict] = None,
    expected_prediction_type: Optional[str] = None,
) -> Dict:
    """Load Stage-2 weights without restoring its old optimizer or data state."""
    from audio2pose.pipelines.pipeline_spd_pose import SPDPoseDiffusionPipeline

    pipeline = SPDPoseDiffusionPipeline.from_pretrained(
        checkpoint_dir=checkpoint_dir,
        condition_dim=int(condition_dim),
        audio_condition_dim=int(audio_condition_dim),
        relation_condition_dim=int(relation_condition_dim),
        scheduler_type="ddim",
        device="cpu",
        pose_channels=int(pose_channels),
        pose_size=int(pose_size),
        torch_dtype=torch.float32,
    )
    if expected_train_data is not None:
        require_same_representation(expected_train_data, pipeline.config["train_data"])
    if expected_prediction_type is not None and pipeline.config.get("prediction_type", "epsilon") != expected_prediction_type:
        raise ValueError("Initialization checkpoint uses a different diffusion prediction type.")
    unet.load_state_dict(pipeline.unet.state_dict(), strict=True)
    if allow_new_pose_context:
        source_state = pipeline.condition_encoder.state_dict()
        target_state = condition_encoder.state_dict()
        missing = set(target_state) - set(source_state)
        unexpected = set(source_state) - set(target_state)
        allowed = ({"pose_context_projection.weight", "pose_context_projection.bias"}
                   if allow_new_pose_context else set())
        if unexpected or not missing <= allowed:
            raise ValueError(f"Incompatible encoder initialization: missing={missing}, unexpected={unexpected}")
        condition_encoder.load_state_dict(source_state, strict=False)
    else:
        condition_encoder.load_state_dict(pipeline.condition_encoder.state_dict())
    best_path = Path(checkpoint_dir) / "best_metrics.json"
    initialization = {
        "source_checkpoint": str(checkpoint_dir), "optimizer_restored": False,
        "source_prediction_type": pipeline.config.get("prediction_type", "epsilon"),
        "source_pose_representation_id": (
            (pipeline.config.get("train_data", {}).get("pose_representation") or {}).get("profile_id")),
        "new_context_parameters": sorted(key for key in missing if key.startswith("pose_context_projection."))
                                  if allow_new_pose_context else [],
        "source_best_metrics": json.loads(best_path.read_text()) if best_path.is_file() else None,
    }
    del pipeline
    return initialization


def metric_is_better(value: float, best_value: Optional[float], mode: str) -> bool:
    if best_value is None:
        return True
    if mode == "min":
        return value < best_value
    if mode == "max":
        return value > best_value
    raise ValueError("best_checkpoint_mode must be 'min' or 'max'.")


def load_existing_best_metric(output_dir: str, metric_name: Optional[str]) -> Optional[float]:
    if not metric_name:
        return None
    path = os.path.join(output_dir, "best-checkpoint", "best_metrics.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        if metadata.get("best_checkpoint_metric") != metric_name:
            return None
        value = metadata.get("best_metric_value")
        return None if value is None else float(value)
    except Exception:
        return None


@torch.no_grad()
def _validation_motion_jerk_sum(
    pred_pose: np.ndarray, pose_valid: np.ndarray,
) -> Tuple[float, int]:
    """Sum absolute third frame differences over valid human joints."""
    pred_pose = np.asarray(pred_pose, dtype=np.float64)
    frame_count = min(pred_pose.shape[0], pose_valid.shape[0])
    joint_count = min(pred_pose.shape[1], pose_valid.shape[1])
    if pred_pose.shape[1] == 75:
        human_indices = tuple(range(75))
    else:
        body_indices = tuple(i for i in range(13) if i not in DEFAULT_IGNORED_JOINT_INDICES)
        human_indices = body_indices + DEFAULT_LEFT_HAND_JOINT_INDICES + DEFAULT_RIGHT_HAND_JOINT_INDICES
    indices = tuple(i for i in human_indices if i < joint_count)
    if frame_count < 4 or not indices:
        return 0.0, 0

    selected = pred_pose[:frame_count, indices]
    valid = pose_valid[:frame_count, indices] & np.isfinite(selected).all(axis=-1)
    jerk = selected[3:] - 3.0 * selected[2:-1] + 3.0 * selected[1:-2] - selected[:-3]
    jerk_valid = valid[3:] & valid[2:-1] & valid[1:-2] & valid[:-3] & np.isfinite(jerk).all(axis=-1)
    return metric_sum(np.abs(jerk), np.broadcast_to(jerk_valid[..., None], jerk.shape))


def run_validation(
    accelerator,
    unet,
    condition_encoder,
    noise_scheduler,
    validation_dataloader,
    output_dir,
    global_step,
    weight_dtype,
    max_batches=4,
    pose_grid_size=(16, 16),
    body_joint_weight: float = 1.0,
    left_hand_joint_weight: float = 3.0,
    left_hand_joint_indices: Tuple[int, ...] = DEFAULT_LEFT_HAND_JOINT_INDICES,
    right_hand_joint_weight: float = 2.0,
    right_hand_joint_indices: Tuple[int, ...] = DEFAULT_RIGHT_HAND_JOINT_INDICES,
    bow_joint_weight: float = 3.0,
    bow_joint_indices: Tuple[int, ...] = DEFAULT_BOW_JOINT_INDICES,
    fingertip_joint_weight: float = 5.0,
    fingertip_joint_indices: Tuple[int, ...] = DEFAULT_FINGERTIP_JOINT_INDICES,
    face_joint_weight: Optional[float] = None,
    face_joint_indices: Tuple[int, ...] = DEFAULT_FACE_JOINT_INDICES,
    lower_body_joint_weight: Optional[float] = None,
    lower_body_joint_indices: Tuple[int, ...] = DEFAULT_LOWER_BODY_JOINT_INDICES,
    torso_anchor_joint_weight: Optional[float] = None,
    torso_anchor_joint_indices: Tuple[int, ...] = DEFAULT_TORSO_ANCHOR_JOINT_INDICES,
    ignored_joint_weight: Optional[float] = 0.0,
    ignored_joint_indices: Tuple[int, ...] = DEFAULT_IGNORED_JOINT_INDICES,
    condition_encoder_type: str = "single",
    dual_condition_include_relation_mask: bool = True,
    relation_condition_mode: str = "normal",
    relation_temporal_features: Optional[Tuple[str, ...]] = None,
    relation_auxiliary_features: Optional[Tuple[str, ...]] = None,
    bow_t_velocity_temperature: float = 0.01,
    sampling_steps: int = 0,
    sampling_max_batches: int = 0,
    sampling_seed: int = 2026,
    sampled_metrics_per_piece: bool = False,
    geometry_refiner=None,
    joint_geometry_options=None,
):
    unet.eval()
    condition_encoder.eval()
    if geometry_refiner is not None:geometry_refiner.eval()
    losses = []
    metric_sums: Dict[str, float] = {}
    metric_counts: Dict[str, int] = {}
    sampled_metric_sums: Dict[str, float] = {}
    sampled_metric_counts: Dict[str, int] = {}
    sampled_batch_count = 0
    sampled_piece_metrics = {}
    sampled_windows = []
    pose_representation = getattr(validation_dataloader.dataset, "pose_representation", None)
    sampling_steps = int(sampling_steps)
    sampling_max_batches = int(sampling_max_batches)
    sampling_enabled = sampling_steps > 0 and sampling_max_batches > 0
    evaluated_batches = min(int(max_batches), len(validation_dataloader))
    sampling_batch_indices = set(evenly_spaced_indices(evaluated_batches, sampling_max_batches))
    if sampled_metrics_per_piece and sampling_enabled and len(sampling_batch_indices) != len(validation_dataloader):
        raise ValueError(
            "Per-piece validation requires sampling every selected validation batch; "
            "increase validation_max_batches/validation_sampling_max_batches."
        )
    if sampling_enabled:
        sampling_scheduler = DDIMScheduler(
            num_train_timesteps=int(noise_scheduler.config.num_train_timesteps),
            beta_start=float(noise_scheduler.config.beta_start),
            beta_end=float(noise_scheduler.config.beta_end),
            beta_schedule=str(noise_scheduler.config.beta_schedule),
            prediction_type=str(noise_scheduler.config.prediction_type),
            clip_sample=False,
        )
        generator_device = accelerator.device if accelerator.device.type == "cuda" else "cpu"
        sampling_generator = torch.Generator(device=generator_device).manual_seed(int(sampling_seed))
    else:
        sampling_scheduler = None
        sampling_generator = None
    for idx, batch in enumerate(validation_dataloader):
        if idx >= max_batches:
            break
        pose = batch["pose_grid"].to(accelerator.device, dtype=torch.float32)
        pose_mask = batch["pose_mask"].to(accelerator.device, dtype=torch.float32)
        joint_counts = batch["joint_count"].to(accelerator.device)
        if condition_encoder_type in ("dual", "dual_add", "gated_fusion"):
            audio_condition = make_audio_condition(batch).to(accelerator.device, dtype=torch.float32)
            relation_condition = make_relation_condition(
                batch,
                include_relation_mask=dual_condition_include_relation_mask,
                temporal_features=relation_temporal_features,
                auxiliary_features=relation_auxiliary_features,
                bow_t_velocity_temperature=bow_t_velocity_temperature,
            ).to(accelerator.device, dtype=torch.float32)
            if relation_condition_mode == "zero":
                relation_condition = torch.zeros_like(relation_condition)
            cond_tokens = condition_encoder(audio_condition, relation_condition)
        else:
            condition = batch["condition"].to(accelerator.device, dtype=torch.float32)
            cond_tokens = condition_encoder(condition)

        noise = torch.randn_like(pose)
        timesteps = torch.randint(
            0,
            noise_scheduler.num_train_timesteps,
            (pose.shape[0],),
            device=pose.device,
        ).long()
        joint_weights = make_joint_weight_grid_batch(
            pose,
            joint_counts,
            pose_grid_size=tuple(int(x) for x in pose_grid_size),
            body_weight=body_joint_weight,
            left_hand_weight=left_hand_joint_weight,
            left_hand_indices=left_hand_joint_indices,
            right_hand_weight=right_hand_joint_weight,
            right_hand_indices=right_hand_joint_indices,
            bow_weight=bow_joint_weight,
            bow_indices=bow_joint_indices,
            fingertip_weight=fingertip_joint_weight,
            fingertip_indices=fingertip_joint_indices,
            face_weight=face_joint_weight,
            face_indices=face_joint_indices,
            lower_body_weight=lower_body_joint_weight,
            lower_body_indices=lower_body_joint_indices,
            torso_anchor_weight=torso_anchor_joint_weight,
            torso_anchor_indices=torso_anchor_joint_indices,
            ignored_joint_weight=ignored_joint_weight,
            ignored_joint_indices=ignored_joint_indices,
        )
        active_pose_mask = pose_mask * (joint_weights > 0).to(dtype=pose_mask.dtype)
        noisy_pose = noise_scheduler.add_noise(pose, noise, timesteps)
        noisy_pose = noisy_pose * active_pose_mask
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        else:
            target = noise_scheduler.get_velocity(pose, noise, timesteps)
        pred = unet(noisy_pose, timesteps, encoder_hidden_states=cond_tokens, skeleton=None).sample
        losses.append(float(weighted_masked_mse(pred, target, pose_mask, joint_weights).detach().cpu()))

        pred_x0 = predict_x0_from_model_output(noise_scheduler, noisy_pose, pred, timesteps)
        pred_x0 = pred_x0 * active_pose_mask
        pred_pose = unpack_pose_grid_torch(
            decode_pose_grid(pred_x0, pose_representation),
            int(joint_counts.max().item()),
            pose_grid_size=tuple(int(x) for x in pose_grid_size),
        )
        if geometry_refiner is not None:
            from audio2pose.training.joint_geometry import smooth_bow
            pred_pose,_=geometry_refiner(smooth_bow(pred_pose,joint_geometry_options['validation_smoothing_window']),
                batch['relation'].to(accelerator.device),batch['strings'].to(accelerator.device))
        pred_pose=pred_pose.detach().cpu().numpy()
        target_pose = batch["pose"].detach().cpu().numpy().astype("float32")
        sampled_pred_pose = None
        should_sample = (
            sampling_enabled
            and sampled_batch_count < sampling_max_batches
            and idx in sampling_batch_indices
        )
        if should_sample:
            try:
                sampled_grid = torch.randn(
                    pose.shape,
                    generator=sampling_generator,
                    device=pose.device,
                    dtype=torch.float32,
                )
            except RuntimeError:
                sampled_grid = torch.randn(pose.shape, device=pose.device, dtype=torch.float32)
            sampled_grid = sampled_grid * active_pose_mask
            try:
                sampling_scheduler.set_timesteps(sampling_steps, device=pose.device)
            except TypeError:
                sampling_scheduler.set_timesteps(sampling_steps)
                sampling_scheduler.timesteps = sampling_scheduler.timesteps.to(pose.device)
            for sampling_timestep in sampling_scheduler.timesteps:
                sampling_input = sampling_scheduler.scale_model_input(sampled_grid, sampling_timestep)
                sampling_pred = unet(
                    sampling_input,
                    sampling_timestep,
                    encoder_hidden_states=cond_tokens,
                    skeleton=None,
                ).sample
                sampled_grid = sampling_scheduler.step(
                    sampling_pred,
                    sampling_timestep,
                    sampled_grid,
                ).prev_sample
                sampled_grid = sampled_grid * active_pose_mask
            sampled_pred_pose = unpack_pose_grid_torch(
                decode_pose_grid(sampled_grid, pose_representation),
                int(joint_counts.max().item()),
                pose_grid_size=tuple(int(x) for x in pose_grid_size),
            )
            if geometry_refiner is not None:
                sampled_pred_pose,_=geometry_refiner(smooth_bow(sampled_pred_pose,joint_geometry_options['validation_smoothing_window']),
                    batch['relation'].to(accelerator.device),batch['strings'].to(accelerator.device))
            sampled_pred_pose=sampled_pred_pose.detach().cpu().numpy()
            sampled_batch_count += 1
        for sample_idx in range(pred_pose.shape[0]):
            joint_count = int(joint_counts[sample_idx].detach().cpu().item())
            pose_valid = batch["pose_valid"][sample_idx, :, :joint_count].detach().cpu().numpy().astype(bool)
            batch_item = {
                "strings": batch["strings"][sample_idx],
                "contact_target_point": batch["contact_target_point"][sample_idx],
                "contact_valid": batch["contact_valid"][sample_idx],
                "bow_target_point": batch["bow_target_point"][sample_idx],
                "bow_valid": batch["bow_valid"][sample_idx],
                "string_frame_scale": batch["string_frame_scale"][sample_idx],
            }
            elgar_metrics = compute_elgar_metric_sums(
                pred_pose[sample_idx, :, :joint_count, :],
                target_pose[sample_idx, :, :joint_count, :],
                batch_item,
            )
            motion_jerk = _validation_motion_jerk_sum(pred_pose[sample_idx, :, :joint_count, :], pose_valid)
            audio_alignment_metrics = compute_audio_alignment_metric_sums(
                pred_pose[sample_idx, :, :joint_count, :],
                batch["onset"][sample_idx],
                batch["pitch"][sample_idx],
                strings=batch_item["strings"],
                bow_valid=batch_item["bow_valid"],
                pose_valid=pose_valid,
                fps=float(getattr(validation_dataloader.dataset, "fps", 30)),
            )
            contact_metrics = compute_contact_diagnostic_sums(
                pred_pose[sample_idx, :, :joint_count, :],
                batch_item["strings"],
                batch_item["contact_target_point"],
                batch_item["contact_valid"],
                distance_scale=batch_item["string_frame_scale"],
            )
            for key, (value_sum, count) in {
                **elgar_metrics,
                **audio_alignment_metrics,
                "syncviolinist_motion_jerk": motion_jerk,
                **contact_metrics,
            }.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value_sum)
                metric_counts[key] = metric_counts.get(key, 0) + int(count)
            if sampled_pred_pose is not None:
                sample_id = str(batch["sample_id"][sample_idx])
                sampled_windows.append({
                    "sample_id": sample_id,
                    "start_frame": int(batch["start_frame"][sample_idx].item()),
                    "end_frame": int(batch["end_frame"][sample_idx].item()),
                })
                sampled_pred = sampled_pred_pose[sample_idx, :, :joint_count, :]
                sampled_target = target_pose[sample_idx, :, :joint_count, :]
                sampled_metrics = compute_elgar_metric_sums(
                    sampled_pred,
                    sampled_target,
                    batch_item,
                )
                sampled_metrics["syncviolinist_motion_jerk"] = _validation_motion_jerk_sum(sampled_pred, pose_valid)
                for key, (value_sum, count) in sampled_metrics.items():
                    sampled_key = f"sampled_{key}"
                    if key == "syncviolinist_motion_jerk":
                        piece = sampled_piece_metrics.setdefault(sample_id, {})
                        old_sum, old_count = piece.get(sampled_key, (0.0, 0))
                        piece[sampled_key] = (old_sum + float(value_sum), old_count + int(count))
                    sampled_metric_sums[sampled_key] = (
                        sampled_metric_sums.get(sampled_key, 0.0) + float(value_sum)
                    )
                    sampled_metric_counts[sampled_key] = (
                        sampled_metric_counts.get(sampled_key, 0) + int(count)
                    )

    val_loss = sum(losses) / max(1, len(losses))
    metrics: Dict[str, Optional[float]] = {
        "val_noise_mse": float(val_loss),
    }
    metrics.update({
        key: (float(metric_sums[key] / metric_counts[key]) if metric_counts.get(key, 0) > 0 else None)
        for key in sorted(metric_sums)
    })
    metrics.update({
        key: (
            float(sampled_metric_sums[key] / sampled_metric_counts[key])
            if sampled_metric_counts.get(key, 0) > 0
            else None
        )
        for key in sorted(sampled_metric_sums)
    })
    if sampled_metrics_per_piece:
        metrics.update(average_piece_metric_sums(sampled_piece_metrics.values()))
    unwrapped_encoder = accelerator.unwrap_model(condition_encoder)
    relation_gate = getattr(unwrapped_encoder, "relation_gate", None)
    if relation_gate is not None:
        metrics["relation_gate"] = float(relation_gate.detach().float().cpu())
    add_elgar_bowing_f1(metrics)
    add_elgar_bowing_f1(metrics, prefix="sampled_")
    add_audio_alignment_f1(metrics)
    if accelerator.is_main_process:
        os.makedirs(os.path.join(output_dir, "validation"), exist_ok=True)
        with open(os.path.join(output_dir, "validation", f"step-{global_step}_sampling.json"), "w", encoding="utf-8") as file:
            json.dump({
                "sampling_seed": int(sampling_seed),
                "aggregation": "per_piece" if sampled_metrics_per_piece else "per_frame",
                "windows": sampled_windows,
                "pieces": sorted(sampled_piece_metrics),
                "metrics_by_piece": {
                    name: average_piece_metric_sums([values])
                    for name, values in sampled_piece_metrics.items()
                },
            }, file, indent=2, sort_keys=True)
        write_grouped_metrics_text(os.path.join(output_dir, "validation", f"step-{global_step}.txt"), metrics)
        if joint_geometry_options is not None and joint_geometry_options.get('bow_trace') is not None:
            clean_metrics={key:(float(value) if value is not None and math.isfinite(float(value)) else None)
                           for key,value in metrics.items()}
            with (Path(output_dir)/'validation_history.jsonl').open('a') as handle:
                handle.write(json.dumps(dict(step=global_step,metrics=clean_metrics,
                    sampling_seed=int(sampling_seed),sampling_steps=sampling_steps,
                    sampled_batches=sampled_batch_count),allow_nan=False)+'\n')
        metric_log = compact_metric_log(metrics)
        if relation_gate is not None and "relation_gate" not in metric_log:
            metric_log = f"{metric_log} relation_gate={metrics['relation_gate']:.6g}".strip()
        logger.info(f"Validation at step {global_step}: {metric_log}")
    unet.train()
    condition_encoder.train()
    if geometry_refiner is not None:geometry_refiner.train()
    return metrics


def _train_stage2(
    output_dir: str,
    train_data: Dict,
    validation_data: Optional[Dict] = None,
    pretrained_model_path: Optional[str] = None,
    use_validation: bool = True,
    validation_steps: int = 1000,
    best_checkpoint_metric: Optional[str] = "contact_wrong_string_rate",
    best_checkpoint_mode: str = "min",
    train_batch_size: int = 4,
    val_batch_size: int = 4,
    max_train_steps: int = 50000,
    learning_rate: float = 1e-4,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    checkpointing_steps: int = 1000,
    resume_from_checkpoint: Optional[str] = None,
    mixed_precision: Optional[str] = "fp16",
    pose_representation_type: Optional[str] = None,
    stage2_init_checkpoint: Optional[str] = None,
    pose_context_input_dim: int = 0,
    pose_context_prefix_frames: int = 0,
    pose_context_probability: float = 0.0,
    pose_context_jitter: float = 0.0,
    pose_context_clean_prefix: bool = False,
    seed: Optional[int] = None,
    num_train_timesteps: int = 1000,
    prediction_type: str = "epsilon",
    cond_hidden_dim: int = 256,
    cond_dim: int = 320,
    cond_layers: int = 2,
    cond_heads: int = 4,
    condition_encoder_type: str = "dual",
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
    dual_condition_include_relation_mask: Optional[bool] = None,
    relation_condition_mode: str = "normal",
    relation_temporal_features: Optional[Tuple[str, ...]] = None,
    relation_auxiliary_features: Optional[Tuple[str, ...]] = None,
    bow_t_velocity_temperature: float = 0.01,
    dual_relation_gate_init: float = 0.05,
    relation_gate_mode: str = "single",
    relation_feature_dim: int = 23,
    joint_mapping_model: Optional[torch.nn.Module] = None,
    joint_mapping_learning_rate: float = 1.0e-5,
    joint_mapping_loss_weight: float = 0.10,
    joint_mapping_loss_warmup_steps: int = 0,
    joint_mapping_categorical_temperature: float = 1.0,
    joint_mapping_hard_categorical: bool = True,
    joint_mapping_relation_dropped_slices: Optional[Sequence[Sequence[int]]] = None,
    joint_mapping_stage1_loss_kwargs: Optional[Dict[str, Any]] = None,
    joint_mapping_allow_frozen: bool = False,
    joint_freeze_unet: bool = False,
    joint_pretrained_stage2_checkpoint: Optional[str] = None,
    unet_block_out_channels: Tuple[int, ...] = (128, 256, 512),
    unet_layers_per_block: int = 1,
    unet_attention_head_dim: Tuple[int, ...] = (4, 4, 8),
    validation_max_batches: int = 4,
    validation_sampling_steps: int = 0,
    validation_sampling_max_batches: int = 0,
    validation_sampling_seed: int = 2026,
    validation_full_sequences: bool = False,
    validation_prefix_frames: int = 0,
    validation_sequence_max_batches: Optional[int] = None,
    validation_windows_per_sample: int = 0,
    validation_sampled_metrics_per_piece: bool = False,
    noise_loss_weight: float = 1.0,
    pose_reconstruction_weight: float = 0.02,
    velocity_loss_weight: float = 0.03,
    acceleration_loss_weight: float = 0.05,
    left_hand_local_loss_weight: float = 0.05,
    left_hand_local_wrist_index: int = 91,
    left_hand_local_joint_indices: Optional[Tuple[int, ...]] = None,
    body_hand_wrist_offset_loss_weight: float = 0.0,
    contact_geometry_loss_weight: float = 0.05,
    contact_distance_loss: str = "l2",
    contact_string_ce_loss_weight: float = 0.0,
    contact_string_ce_temperature: float = 0.02,
    contact_finger_supervision: str = "pred_min",
    contact_fingertip_indices: Tuple[int, ...] = DEFAULT_CONTACT_FINGERTIP_INDICES,
    bow_position_loss_weight: float = 0.05,
    bow_direction_loss_weight: float = 0.02,
    bow_string_loss_weight: float = 0.0,
    bow_radial_velocity_loss_weight: float = 0.0,
    bow_radial_velocity_loss_beta: float = 0.01,
    bow_radial_velocity_loss_scale: float = 1.0,
    bow_radial_velocity_timestep_fraction: Optional[float] = None,
    bow_radial_velocity_log_interval: int = 0,
    bow_radial_velocity_direction_loss_weight: float = 0.0,
    bow_radial_velocity_direction_temperature: float = 0.01,
    bow_radial_velocity_direction_min_velocity: float = 0.001,
    bow_endpoint_smoothness_loss_weight: float = 0.0,
    bow_endpoint_smoothness_window: int = 5,
    bow_endpoint_smoothness_exclude_radius: int = 3,
    bow_endpoint_smoothness_min_velocity: float = 0.001,
    bow_endpoint_radial_stable_sign_loss_weight: float = 0.0,
    bow_endpoint_radial_stable_sign_window: int = 3,
    bow_endpoint_radial_stable_sign_exclude_radius: int = 3,
    bow_endpoint_radial_stable_sign_min_velocity: float = 0.001,
    bow_endpoint_radial_stable_sign_margin: float = 0.0,
    bow_reversal_event_loss_weight: float = 0.0,
    bow_reversal_event_temperature: float = 0.01,
    bow_reversal_event_target_min_velocity: float = 0.001,
    bow_reversal_event_relation_min_velocity: float = 0.005,
    bow_reversal_event_relation_tolerance_frames: int = 2,
    bow_reversal_event_timestep_fraction: float = 0.25,
    body_joint_weight: float = 1.0,
    left_hand_joint_weight: float = 3.0,
    left_hand_joint_indices: Tuple[int, ...] = DEFAULT_LEFT_HAND_JOINT_INDICES,
    right_hand_joint_weight: float = 2.0,
    right_hand_joint_indices: Tuple[int, ...] = DEFAULT_RIGHT_HAND_JOINT_INDICES,
    bow_joint_weight: float = 3.0,
    bow_joint_indices: Tuple[int, ...] = DEFAULT_BOW_JOINT_INDICES,
    fingertip_joint_weight: float = 5.0,
    fingertip_joint_indices: Tuple[int, ...] = DEFAULT_FINGERTIP_JOINT_INDICES,
    face_joint_weight: Optional[float] = None,
    face_joint_indices: Tuple[int, ...] = DEFAULT_FACE_JOINT_INDICES,
    lower_body_joint_weight: Optional[float] = None,
    lower_body_joint_indices: Tuple[int, ...] = DEFAULT_LOWER_BODY_JOINT_INDICES,
    torso_anchor_joint_weight: Optional[float] = None,
    torso_anchor_joint_indices: Tuple[int, ...] = DEFAULT_TORSO_ANCHOR_JOINT_INDICES,
    ignored_joint_weight: Optional[float] = 0.0,
    ignored_joint_indices: Tuple[int, ...] = DEFAULT_IGNORED_JOINT_INDICES,
    geometry_loss_start_step: int = 0,
    auxiliary_timestep_fraction: float = 0.25,
    structure_timestep_fraction: Optional[float] = None,
    geometry_timestep_fraction: Optional[float] = None,
    auxiliary_loss_warmup_steps: int = 4000,
    training_mode: str = "diffusion",
    local_bow_refiner_options: Optional[Dict[str, Any]] = None,
    local_bow_refiner_context: Optional[Dict[str, Any]] = None,
    audio_bow_event_options: Optional[Dict[str, Any]] = None,
    audio_bow_event_context: Optional[Dict[str, Any]] = None,
    joint_geometry_options: Optional[Dict[str, Any]] = None,
    *,
    dataset_factory=SPDCanonicalPoseDataset,
    optimizer_learning_rates: Optional[Tuple[float, float]] = None,
    weights_only_resume: bool = False,
):
    if training_mode == "audio_bow_events":
        if audio_bow_event_context is None:
            raise ValueError("Use train_stage2.py for the audio-event experiment")
        from audio2pose.training.audio_bow_events import run_from_finetune
        return run_from_finetune(dict(training_mode=training_mode,audio_bow_event_options=audio_bow_event_options,
            train_data=train_data,validation_data=validation_data,seed=seed,learning_rate=learning_rate,
            train_batch_size=train_batch_size,validation_steps=validation_steps,checkpointing_steps=checkpointing_steps),
            audio_bow_event_context)
    if training_mode == "local_bow_refiner":
        if local_bow_refiner_context is None:
            raise ValueError("Use train_stage2.py for fixed Stage-1/local-head training")
        from audio2pose.training.local_bow_refiner import run_from_finetune
        return run_from_finetune(dict(
            training_mode=training_mode, local_bow_refiner_options=local_bow_refiner_options,
            output_dir=output_dir, train_data=train_data, validation_data=validation_data,
            max_train_steps=max_train_steps, resume_from_checkpoint=resume_from_checkpoint,
            use_validation=use_validation, validation_steps=validation_steps,
            seed=seed, checkpointing_steps=checkpointing_steps, train_batch_size=train_batch_size,
            learning_rate=learning_rate, adam_weight_decay=adam_weight_decay, max_grad_norm=max_grad_norm),
            local_bow_refiner_context)
    if training_mode != "diffusion":
        raise ValueError(f"Unknown training_mode: {training_mode}")
    if joint_geometry_options is not None:
        from audio2pose.training.joint_geometry import validate_options
        validate_options(joint_geometry_options)
        if joint_mapping_model is not None or joint_freeze_unet or stage2_init_checkpoint:
            raise ValueError("Joint geometry requires a fresh Stage-2 origin or full-state resume, with fixed Stage-1")
        if resume_from_checkpoint:
            checkpoint = resolve_resume_checkpoint(output_dir, resume_from_checkpoint).resolve()
            if checkpoint.parent != Path(output_dir).resolve():
                raise ValueError('Joint geometry resume must stay inside the original output_dir')
        if pose_context_input_dim or train_data.get('pose_keys') != ['coco_wholebody_string_frame','bow_string_frame']:
            raise ValueError("Joint geometry currently requires native canonical SPD without pose-prefix conditioning")
        if condition_encoder_type!='gated_fusion' or relation_feature_dim!=24:
            raise ValueError("Joint geometry requires the existing 24D gated relation interface")
        if joint_geometry_options.get('bow_trace') is not None:
            if train_data.get('mode')!='train' or not use_validation or not validation_data or validation_data.get('mode')!='val':
                raise ValueError('Joint bow trace trains on train only and requires a separate val diagnostic split')
            if validation_windows_per_sample:
                raise ValueError('Use complete val windows for this fixed recipe, not mutable wrapper window subsets')
    # Capture formal arguments only: getargvalues().locals also contains
    # runtime imports (e.g. validate_options), which cannot be saved as YAML.
    argument_state = inspect.getargvalues(inspect.currentframe())
    config = {name: argument_state.locals[name] for name in argument_state.args}
    for runtime_arg in ("joint_mapping_model", "dataset_factory", "optimizer_learning_rates", "weights_only_resume"):
        config.pop(runtime_arg)
    config["joint_mapping_enabled"] = joint_mapping_model is not None
    if stage2_init_checkpoint and resume_from_checkpoint:
        raise ValueError("Use weight initialization for a new run OR resume optimizer state, not both.")
    if stage2_init_checkpoint and not Path(stage2_init_checkpoint).is_dir():
        raise FileNotFoundError(f"Missing initialization checkpoint: {stage2_init_checkpoint}")
    if stage2_init_checkpoint:
        source = Path(stage2_init_checkpoint).resolve()
        if Path(output_dir).resolve() in (source, source.parent) or (Path(output_dir) / "config.yaml").exists():
            raise ValueError("Weight initialization needs a new output directory; preserve the source run.")
    if pose_context_input_dim < 0 or not 0 <= pose_context_probability <= 1 or pose_context_jitter < 0:
        raise ValueError("Invalid pose-context dimensions/probability/jitter.")
    if pose_context_input_dim:
        if condition_encoder_type != "dual_add" or joint_mapping_model is not None:
            raise ValueError("Pose-context experiments currently require standalone dual_add Stage-2.")
        if train_data.get("pose_keys") != ["syncviolinist_pose"]:
            raise ValueError("Pose-context training is currently restricted to native SyncViolinist.")
        if not 0 <= pose_context_prefix_frames < int(train_data["sequence_length"]):
            raise ValueError("Invalid training prefix length.")
        if pose_context_prefix_frames and not pose_context_clean_prefix:
            raise ValueError("Context training requires the matching clean-prefix sampling mode.")
        if pose_context_prefix_frames and not validation_full_sequences:
            raise ValueError("Context training must select checkpoints using full predicted-prefix validation.")
    elif pose_context_prefix_frames:
        raise ValueError("Training prefixes require pose_context_input_dim.")
    if validation_full_sequences:
        if train_data.get("pose_keys") != ["syncviolinist_pose"]:
            raise ValueError("Full-piece validation currently requires native SyncViolinist pose targets.")
        if validation_prefix_frames and condition_encoder_type != "dual_add":
            raise ValueError("Legacy pose-prefix validation requires dual_add; shared models use prefix=0.")
        if not 0 <= validation_prefix_frames < int(train_data["sequence_length"]):
            raise ValueError("Invalid validation prefix length.")
    if structure_timestep_fraction is None:
        structure_timestep_fraction = float(auxiliary_timestep_fraction)
    else:
        structure_timestep_fraction = float(structure_timestep_fraction)
    if geometry_timestep_fraction is None:
        geometry_timestep_fraction = float(auxiliary_timestep_fraction)
    else:
        geometry_timestep_fraction = float(geometry_timestep_fraction)
    config["structure_timestep_fraction"] = structure_timestep_fraction
    config["geometry_timestep_fraction"] = geometry_timestep_fraction
    if not math.isfinite(float(bow_radial_velocity_loss_scale)) or float(bow_radial_velocity_loss_scale) <= 0:
        raise ValueError("bow_radial_velocity_loss_scale must be finite and positive.")
    if bow_radial_velocity_timestep_fraction is None:
        bow_radial_velocity_timestep_fraction = structure_timestep_fraction
    else:
        bow_radial_velocity_timestep_fraction = float(bow_radial_velocity_timestep_fraction)
        if not 0.0 < bow_radial_velocity_timestep_fraction <= 1.0:
            raise ValueError("bow_radial_velocity_timestep_fraction must be in (0, 1].")
    config["bow_radial_velocity_timestep_fraction"] = bow_radial_velocity_timestep_fraction
    if int(bow_radial_velocity_log_interval) < 0:
        raise ValueError("bow_radial_velocity_log_interval must be nonnegative.")
    best_checkpoint_mode = str(best_checkpoint_mode)
    if best_checkpoint_mode not in ("min", "max"):
        raise ValueError("best_checkpoint_mode must be 'min' or 'max'.")
    contact_finger_supervision = str(contact_finger_supervision)
    if contact_finger_supervision not in ("gt", "pred_min"):
        raise ValueError("contact_finger_supervision must be 'gt' or 'pred_min'.")
    contact_distance_loss = str(contact_distance_loss)
    if contact_distance_loss not in ("l1", "l2"):
        raise ValueError("contact_distance_loss must be 'l1' or 'l2'.")
    contact_fingertip_indices = normalize_joint_indices(contact_fingertip_indices)
    left_hand_joint_indices = normalize_joint_indices(left_hand_joint_indices)
    right_hand_joint_indices = normalize_joint_indices(right_hand_joint_indices)
    bow_joint_indices = normalize_joint_indices(bow_joint_indices)
    fingertip_joint_indices = normalize_joint_indices(fingertip_joint_indices)
    condition_encoder_type = str(condition_encoder_type)
    if condition_encoder_type not in ("single", "dual", "dual_add", "gated_fusion"):
        raise ValueError("condition_encoder_type must be 'single', 'dual', 'dual_add', or 'gated_fusion'.")
    relation_condition_mode = str(relation_condition_mode).lower()
    if relation_condition_mode not in ("normal", "zero"):
        raise ValueError("relation_condition_mode must be 'normal' or 'zero'.")
    if relation_condition_mode == "zero" and condition_encoder_type == "single":
        raise ValueError(
            "relation_condition_mode='zero' requires a dual, dual_add, or gated_fusion condition encoder."
        )
    if dual_condition_include_relation_mask is None:
        dual_condition_include_relation_mask = bool(train_data.get("include_relation_mask_in_condition", True))
    else:
        dual_condition_include_relation_mask = bool(dual_condition_include_relation_mask)
    config["dual_condition_include_relation_mask"] = dual_condition_include_relation_mask
    relation_temporal_features = tuple(str(name) for name in (relation_temporal_features or ()))
    relation_auxiliary_features = tuple(
        str(name) for name in (relation_auxiliary_features or ())
    )
    audio_encoder_overrides = {
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
    }
    train_data = apply_audio_encoder_overrides(train_data, audio_encoder_overrides)
    train_data = disable_cuda_audio_encoder_workers(train_data)
    train_data = resolve_pose_representation(train_data)
    if pose_representation_type is not None:
        actual_type = (train_data.get("pose_representation") or {}).get("type")
        if actual_type != pose_representation_type:
            raise ValueError("Requested pose representation does not match the training profile.")
    use_validation = bool(use_validation and validation_data is not None and int(validation_steps) > 0)
    if use_validation:
        validation_data = apply_audio_encoder_overrides(validation_data, audio_encoder_overrides)
        validation_data = disable_cuda_audio_encoder_workers(validation_data)
        validation_data = resolve_pose_representation(validation_data)
        require_same_representation(train_data, validation_data)
    config["train_data"] = train_data
    config["validation_data"] = validation_data
    # Check before saving a new config over the previous run's frozen profile.
    if resume_from_checkpoint:
        previous_paths = [Path(output_dir) / "config.yaml"]
        if resume_from_checkpoint != "latest":
            resume_path = Path(resume_from_checkpoint)
            if not resume_path.exists():
                resume_path = Path(output_dir) / resume_path
            previous_paths.extend([resume_path / "config.yaml", resume_path.parent / "config.yaml"])
        previous_paths = [path for path in previous_paths if path.is_file()]
        if train_data.get("pose_representation") is not None and not previous_paths:
            raise ValueError("Cannot verify the resume checkpoint's pose representation; start a new run.")
        for previous_config in previous_paths:
            previous = OmegaConf.to_container(OmegaConf.load(previous_config), resolve=True)
            require_same_representation(train_data, previous["train_data"])
            if joint_geometry_options is not None:
                for key in ('joint_geometry_options', 'max_train_steps', 'gradient_accumulation_steps', 'train_batch_size'):
                    if previous.get(key) != config.get(key):
                        raise ValueError(f'Cannot change {key} during a joint geometry resume')
            if pose_context_input_dim or previous.get("pose_context_input_dim", 0):
                for key, default in (("pose_context_input_dim", 0), ("pose_context_prefix_frames", 0),
                                     ("pose_context_probability", 0.0), ("pose_context_jitter", 0.0),
                                     ("pose_context_clean_prefix", False), ("validation_prefix_frames", 0)):
                    if previous.get(key, default) != config.get(key, default):
                        raise ValueError(f"Cannot change {key} while resuming; initialize a new experiment instead.")

    accelerator_class = Accelerator
    if weights_only_resume:
        accelerator_class = _make_weight_only_accelerator_class(Accelerator)
    accelerator = accelerator_class(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    if seed is not None:
        set_seed(seed)

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        if not (resume_from_checkpoint and joint_geometry_options is not None and (Path(output_dir) / 'config.yaml').is_file()):
            OmegaConf.save(config, os.path.join(output_dir, "config.yaml"))
    logger.info("Relation condition mode: %s", relation_condition_mode)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="scaled_linear",
        prediction_type=prediction_type,
    )

    train_dataset = dataset_factory(**train_data)
    pose_representation = train_dataset.pose_representation
    validation_dataset = dataset_factory(**validation_data) if use_validation else None
    if validation_dataset is not None and int(validation_windows_per_sample) > 0 and not validation_full_sequences:
        selected = select_validation_windows(
            validation_dataset.windows, validation_dataset.records, int(validation_windows_per_sample)
        )
        validation_dataset.windows = [validation_dataset.windows[index] for index in selected]
        if accelerator.is_main_process:
            selection_path = Path(output_dir) / "validation" / "selected_windows.json"
            selection_path.parent.mkdir(parents=True, exist_ok=True)
            selection_path.write_text(json.dumps([
                {
                    "sample_id": validation_dataset.records[record_index]["id"],
                    "start_frame": int(start),
                    "end_frame": min(int(start) + validation_dataset.sequence_length,
                                     int(validation_dataset.records[record_index]["n_frames"])),
                }
                for record_index, start in validation_dataset.windows
            ], indent=2), encoding="utf-8")
    sample = train_dataset[0]
    condition_dim = int(sample["condition"].shape[-1])
    audio_condition_dim, relation_condition_dim = infer_dual_condition_dims(
        sample,
        include_relation_mask=dual_condition_include_relation_mask,
        temporal_features=relation_temporal_features,
        auxiliary_features=relation_auxiliary_features,
        bow_t_velocity_temperature=bow_t_velocity_temperature,
    )
    pose_channels = int(sample["pose_grid"].shape[0])
    pose_size = int(sample["pose_grid"].shape[-1])
    pose_grid_size = tuple(int(x) for x in sample["pose_grid"].shape[-2:])
    if pose_context_input_dim and pose_context_input_dim != pose_channels * pose_grid_size[0] * pose_grid_size[1]:
        raise ValueError("pose_context_input_dim must equal the flattened pose coordinates per frame.")

    if condition_encoder_type == "dual":
        condition_encoder = DualPoseConditionEncoder(
            audio_input_dim=audio_condition_dim,
            relation_input_dim=relation_condition_dim,
            hidden_dim=cond_hidden_dim,
            output_dim=cond_dim,
            num_layers=cond_layers,
            nhead=cond_heads,
        )
    elif condition_encoder_type == "dual_add":
        condition_encoder = AdditiveDualPoseConditionEncoder(
            audio_input_dim=audio_condition_dim,
            relation_input_dim=relation_condition_dim,
            hidden_dim=cond_hidden_dim,
            output_dim=cond_dim,
            num_layers=cond_layers,
            nhead=cond_heads,
            relation_gate_init=dual_relation_gate_init,
            pose_context_input_dim=pose_context_input_dim,
        )
    elif condition_encoder_type == "gated_fusion":
        condition_encoder = GatedFusionPoseConditionEncoder(
            audio_input_dim=audio_condition_dim,
            relation_input_dim=relation_condition_dim,
            hidden_dim=cond_hidden_dim,
            output_dim=cond_dim,
            num_layers=cond_layers,
            nhead=cond_heads,
            relation_gate_init=dual_relation_gate_init,
            relation_gate_mode=relation_gate_mode,
            relation_feature_dim=relation_feature_dim,
        )
    else:
        condition_encoder = PoseConditionEncoder(
            input_dim=condition_dim,
            hidden_dim=cond_hidden_dim,
            output_dim=cond_dim,
            num_layers=cond_layers,
            nhead=cond_heads,
        )
    unet = create_pose_unet(
        in_channels=pose_channels,
        cross_attention_dim=cond_dim,
        sample_size=pose_size,
        block_out_channels=tuple(unet_block_out_channels),
        layers_per_block=unet_layers_per_block,
        attention_head_dim=tuple(unet_attention_head_dim),
    )

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if stage2_init_checkpoint:
        if Path(stage2_init_checkpoint).resolve() == Path(output_dir).resolve():
            raise ValueError("Initialization and output directories must be different.")
        initialization = load_pretrained_stage2_weights(
            stage2_init_checkpoint, unet=unet, condition_encoder=condition_encoder,
            condition_dim=condition_dim, audio_condition_dim=audio_condition_dim,
            relation_condition_dim=relation_condition_dim, pose_channels=pose_channels,
            pose_size=pose_size, allow_new_pose_context=bool(pose_context_input_dim),
            expected_train_data=train_data, expected_prediction_type=prediction_type,
        )
        if accelerator.is_main_process:
            (Path(output_dir) / "weight_initialization.json").write_text(
                json.dumps(initialization, indent=2), encoding="utf-8")
        # Reproducible B0/B1 data/noise streams despite adding the same new adapter.
        if seed is not None:
            set_seed(seed)

    if joint_mapping_model is not None:
        if condition_encoder_type not in ("dual", "dual_add", "gated_fusion"):
            raise ValueError("joint_mapping_model requires a dual relation condition encoder.")
        if joint_freeze_unet and not joint_pretrained_stage2_checkpoint:
            raise ValueError(
                "joint_freeze_unet=true requires joint_pretrained_stage2_checkpoint."
            )
        if joint_pretrained_stage2_checkpoint:
            load_pretrained_stage2_weights(
                joint_pretrained_stage2_checkpoint,
                unet=unet,
                condition_encoder=condition_encoder,
                condition_dim=condition_dim,
                audio_condition_dim=audio_condition_dim,
                relation_condition_dim=relation_condition_dim,
                pose_channels=pose_channels,
                pose_size=pose_size,
            )
        if joint_freeze_unet:
            for parameter in unet.parameters():
                parameter.requires_grad_(False)
        _set_joint_mapping_train_mode(joint_mapping_model)

    geometry_refiner=None
    if joint_geometry_options is not None:
        from audio2pose.training.joint_geometry import create_refiner
        if accelerator.num_processes!=1:
            raise ValueError("Initial joint geometry recipe uses one GPU/process; launch the original Python entry directly")
        geometry_refiner=create_refiner(joint_geometry_options)

    trainable_mapping_parameters = (
        [parameter for parameter in joint_mapping_model.parameters() if parameter.requires_grad]
        if joint_mapping_model is not None
        else []
    )
    if (
        joint_mapping_model is not None
        and not trainable_mapping_parameters
        and not bool(joint_mapping_allow_frozen)
    ):
        raise ValueError("joint_mapping_model has no trainable parameters.")

    if scale_lr:
        learning_rate = learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes

    parameters = list(condition_encoder.parameters())
    if geometry_refiner is not None:
        parameters.extend(geometry_refiner.parameters())
    optimizer_defaults = dict(
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )
    if optimizer_learning_rates is not None:
        if joint_mapping_model is not None:
            raise ValueError("Separate Stage-2 learning rates require a frozen Stage 1.")
        unet_lr, condition_lr = optimizer_learning_rates
        print(f"Creating AdamW parameter groups: UNet lr={unet_lr:.3e}, condition encoder lr={condition_lr:.3e}.")
        optimizer = torch.optim.AdamW(
            [{"params": list(unet.parameters()), "lr": unet_lr},
             {"params": parameters, "lr": condition_lr}],
            lr=condition_lr,
            **optimizer_defaults,
        )
    elif joint_mapping_model is None:
        optimizer = torch.optim.AdamW(
            list(unet.parameters()) + parameters, lr=learning_rate, **optimizer_defaults,
        )
    else:
        parameter_groups = []
        if not joint_freeze_unet:
            parameter_groups.append({"params": list(unet.parameters()), "lr": learning_rate})
        parameter_groups.append({"params": list(condition_encoder.parameters()), "lr": learning_rate})
        if trainable_mapping_parameters:
            parameter_groups.append(
                {"params": trainable_mapping_parameters, "lr": float(joint_mapping_learning_rate)}
            )
        optimizer = torch.optim.AdamW(parameter_groups, **optimizer_defaults)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=int(train_data.get("num_workers", 4)),
        pin_memory=True,
    )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=int(validation_data.get("num_workers", 4)),
        pin_memory=True,
    ) if use_validation else None

    lr_sched = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    if joint_mapping_model is not None:
        unet, condition_encoder, joint_mapping_model, optimizer, train_dataloader, lr_sched = accelerator.prepare(
            unet, condition_encoder, joint_mapping_model, optimizer, train_dataloader, lr_sched
        )
    elif geometry_refiner is not None:
        unet,condition_encoder,geometry_refiner,optimizer,train_dataloader,lr_sched=accelerator.prepare(
            unet,condition_encoder,geometry_refiner,optimizer,train_dataloader,lr_sched)
    else:
        unet, condition_encoder, optimizer, train_dataloader, lr_sched = accelerator.prepare(
            unet, condition_encoder, optimizer, train_dataloader, lr_sched
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("spd-audio-relation-to-pose")

    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    logger.info("***** Running SPD pose diffusion training *****")
    logger.info(f"  Num train windows = {len(train_dataset)}")
    logger.info(f"  Use validation = {use_validation}")
    if validation_dataset is not None:
        logger.info(f"  Num validation windows = {len(validation_dataset)}")
    logger.info(f"  Condition encoder type = {condition_encoder_type}")
    logger.info(f"  Condition encoder class = {condition_encoder.__class__.__name__}")
    logger.info(f"  Audio encoder type = {train_data.get('audio_encoder_type', 'none')}")
    logger.info(f"  Audio feature cache dir = {train_data.get('audio_feature_cache_dir', 'none')}")
    logger.info(f"  Audio feature cache required = {train_data.get('audio_feature_cache_required', False)}")
    logger.info(f"  Train num_workers = {train_data.get('num_workers', 4)}")
    if validation_data is not None:
        logger.info(f"  Validation num_workers = {validation_data.get('num_workers', 4)}")
    logger.info(f"  Condition dim = {condition_dim}")
    logger.info(f"  Audio condition dim = {audio_condition_dim}")
    logger.info(f"  Relation condition dim = {relation_condition_dim}")
    if condition_encoder_type in ("dual_add", "gated_fusion"):
        logger.info(f"  Relation gate init = {dual_relation_gate_init}")
    logger.info(f"  Pose grid = {tuple(sample['pose_grid'].shape)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0
    resume_step = 0
    best_metric_value: Optional[float] = (
        load_existing_best_metric(output_dir, best_checkpoint_metric)
        if use_validation and best_checkpoint_metric
        else None
    )
    if accelerator.is_main_process and best_metric_value is not None:
        logger.info(
            f"Loaded existing best checkpoint score: "
            f"{best_checkpoint_metric}={best_metric_value:.6f} ({best_checkpoint_mode})"
        )
    if resume_from_checkpoint:
        checkpoint_path = resolve_resume_checkpoint(output_dir, resume_from_checkpoint)
        accelerator.print(f"Resuming from checkpoint {checkpoint_path}")
        if joint_geometry_options is not None:
            restore_joint_training_state(accelerator, checkpoint_path, lr_sched, optimizer,
                                         lr_scheduler, lr_warmup_steps, gradient_accumulation_steps)
        else:
            accelerator.load_state(str(checkpoint_path))
        global_step = int(checkpoint_path.name.split("-")[1])
        first_epoch = global_step // num_update_steps_per_epoch
        # global_step counts optimizer updates; DataLoader step counts batches.
        resume_step = (global_step % num_update_steps_per_epoch) * gradient_accumulation_steps

    set_dataset_training_step(train_dataset, global_step)

    joint_stage1_loss_kwargs = dict(
        {
            "mse_weight": 1.0,
            "l1_weight": 1.0,
            "contact_ce_weight": 1.0,
            "bow_ce_weight": 1.0,
            "validity_bce_weight": 0.1,
        },
        **(joint_mapping_stage1_loss_kwargs or {}),
    )
    if joint_mapping_model is not None:
        logger.info(
            "Joint mapping enabled: mapping_lr=%.3e stage1_loss_weight=%.4f freeze_unet=%s",
            float(joint_mapping_learning_rate),
            float(joint_mapping_loss_weight),
            bool(joint_freeze_unet),
        )

    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    context_generator = torch.Generator(
        device=accelerator.device if accelerator.device.type == "cuda" else "cpu"
    ).manual_seed(int(seed or 0) + 17017)

    for epoch in range(first_epoch, num_train_epochs):
        if joint_freeze_unet:
            unet.eval()
        else:
            unet.train()
        condition_encoder.train()
        if geometry_refiner is not None:geometry_refiner.train()
        if joint_mapping_model is not None:
            _set_joint_mapping_train_mode(joint_mapping_model)
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                continue

            # The project also supports older Accelerate releases where
            # accumulate() accepts exactly one model. Gradient accumulation
            # still applies to every parameter handled by the optimizer.
            with accelerator.accumulate(unet):
                pose = batch["pose_grid"].to(accelerator.device, dtype=torch.float32)
                pose_mask = batch["pose_mask"].to(accelerator.device, dtype=torch.float32)
                pose_valid = batch["pose_valid"].to(accelerator.device, dtype=torch.float32)
                clean_pose = batch["pose"].to(accelerator.device, dtype=torch.float32)
                joint_counts = batch["joint_count"].to(accelerator.device)
                known_pose, known_mask = None, None
                if pose_context_input_dim:
                    context_generator.manual_seed(int(seed or 0) + 17017 + global_step * gradient_accumulation_steps
                                                  + step % gradient_accumulation_steps)
                    known_pose, known_mask = training_context(
                        pose, int(pose_context_prefix_frames), float(pose_context_probability),
                        float(pose_context_jitter), context_generator)
                    known_pose = known_pose * pose_mask
                    first_window = (batch["start_frame"].to(pose.device) == 0)
                    known_mask = known_mask * (~first_window).reshape(-1, 1, 1, 1, 1)
                    known_pose = known_pose * known_mask
                predicted_frames = (torch.ones((pose.shape[0], pose.shape[2]), device=pose.device, dtype=torch.bool)
                                    if known_mask is None else ~known_mask[:, 0, :, 0, 0].bool())

                joint_stage1_loss = pose.new_tensor(0.0)
                if joint_mapping_model is not None:
                    joint_relation, joint_stage1_loss = _joint_mapping_forward(
                        batch,
                        joint_mapping_model,
                        device=accelerator.device,
                        categorical_temperature=float(joint_mapping_categorical_temperature),
                        hard_categorical=bool(joint_mapping_hard_categorical),
                        relation_dropped_slices=joint_mapping_relation_dropped_slices,
                        stage1_loss_kwargs=joint_stage1_loss_kwargs,
                    )
                    batch["relation"] = joint_relation

                if condition_encoder_type in ("dual", "dual_add", "gated_fusion"):
                    audio_condition = make_audio_condition(batch).to(accelerator.device, dtype=torch.float32)
                    relation_condition = make_relation_condition(
                        batch,
                        include_relation_mask=dual_condition_include_relation_mask,
                        temporal_features=relation_temporal_features,
                        auxiliary_features=relation_auxiliary_features,
                        bow_t_velocity_temperature=bow_t_velocity_temperature,
                    ).to(accelerator.device, dtype=torch.float32)
                    if relation_condition_mode == "zero":
                        relation_condition = torch.zeros_like(relation_condition)
                    context_kwargs = {} if known_pose is None else dict(pose_context=known_pose, pose_context_mask=known_mask)
                    cond_tokens = condition_encoder(audio_condition, relation_condition, **context_kwargs)
                else:
                    condition = batch["condition"].to(accelerator.device, dtype=torch.float32)
                    cond_tokens = condition_encoder(condition)
                noise = torch.randn_like(pose)
                bsz = pose.shape[0]
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=pose.device).long()
                joint_weights = make_joint_weight_grid_batch(
                    pose,
                    joint_counts,
                    pose_grid_size,
                    body_weight=body_joint_weight,
                    left_hand_weight=left_hand_joint_weight,
                    left_hand_indices=left_hand_joint_indices,
                    right_hand_weight=right_hand_joint_weight,
                    right_hand_indices=right_hand_joint_indices,
                    bow_weight=bow_joint_weight,
                    bow_indices=bow_joint_indices,
                    fingertip_weight=fingertip_joint_weight,
                    fingertip_indices=fingertip_joint_indices,
                    face_weight=face_joint_weight,
                    face_indices=face_joint_indices,
                    lower_body_weight=lower_body_joint_weight,
                    lower_body_indices=lower_body_joint_indices,
                    torso_anchor_weight=torso_anchor_joint_weight,
                    torso_anchor_indices=torso_anchor_joint_indices,
                    ignored_joint_weight=ignored_joint_weight,
                    ignored_joint_indices=ignored_joint_indices,
                )
                active_pose_mask = pose_mask * (joint_weights > 0).to(dtype=pose_mask.dtype)
                noisy_pose = noise_scheduler.add_noise(pose, noise, timesteps)
                noisy_pose = noisy_pose * active_pose_mask
                if known_pose is not None:
                    noisy_pose = torch.where(known_mask.bool(), known_pose, noisy_pose)
                supervised_pose_mask = pose_mask * predicted_frames[:, None, :, None, None]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(pose, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                model_pred = unet(noisy_pose, timesteps, encoder_hidden_states=cond_tokens, skeleton=None).sample
                noise_loss = weighted_masked_mse(model_pred, target, supervised_pose_mask, joint_weights)
                loss = noise_loss * float(noise_loss_weight)
                if joint_mapping_model is not None and float(joint_mapping_loss_weight) != 0.0:
                    joint_loss_ramp = ramp_weight(
                        global_step,
                        start_step=0,
                        warmup_steps=int(joint_mapping_loss_warmup_steps),
                    )
                    loss = loss + (
                        joint_loss_ramp
                        * float(joint_mapping_loss_weight)
                        * joint_stage1_loss
                    )

                pred_x0 = predict_x0_from_model_output(noise_scheduler, noisy_pose, model_pred, timesteps)
                pred_x0 = pred_x0 * active_pose_mask
                if known_pose is not None:
                    pred_x0 = torch.where(known_mask.bool(), known_pose, pred_x0)
                native_pred_x0 = decode_pose_grid(pred_x0, pose_representation)
                pred_pose = unpack_pose_grid_torch(
                    native_pred_x0,
                    int(joint_counts.max().item()),
                    pose_grid_size,
                )
                target_pose = clean_pose[:, :, : pred_pose.shape[2], :]
                target_valid = pose_valid[:, :, : pred_pose.shape[2]]
                joint_geometry_loss=pred_pose.new_tensor(0.)
                joint_geometry_terms={}
                if geometry_refiner is not None:
                    from audio2pose.training.joint_geometry import loss as geometry_refinement_loss
                    joint_geometry_loss,joint_geometry_terms=geometry_refinement_loss(geometry_refiner,
                        pred_pose,target_pose,batch,timesteps,noise_scheduler.num_train_timesteps,
                        joint_geometry_options,global_step,noise_reference=noise_loss)
                    loss=loss+joint_geometry_loss
                joint_weight_vector = make_joint_weight_vector_batch(
                    joint_counts,
                    max_joint_count=pred_pose.shape[2],
                    device=accelerator.device,
                    dtype=torch.float32,
                    body_weight=body_joint_weight,
                    left_hand_weight=left_hand_joint_weight,
                    left_hand_indices=left_hand_joint_indices,
                    right_hand_weight=right_hand_joint_weight,
                    right_hand_indices=right_hand_joint_indices,
                    bow_weight=bow_joint_weight,
                    bow_indices=bow_joint_indices,
                    fingertip_weight=fingertip_joint_weight,
                    fingertip_indices=fingertip_joint_indices,
                    face_weight=face_joint_weight,
                    face_indices=face_joint_indices,
                    lower_body_weight=lower_body_joint_weight,
                    lower_body_indices=lower_body_joint_indices,
                    torso_anchor_weight=torso_anchor_joint_weight,
                    torso_anchor_indices=torso_anchor_joint_indices,
                    ignored_joint_weight=ignored_joint_weight,
                    ignored_joint_indices=ignored_joint_indices,
                )

                structure_sample_weight = auxiliary_sample_weight(
                    timesteps,
                    noise_scheduler.num_train_timesteps,
                    max_timestep_fraction=structure_timestep_fraction,
                ).to(accelerator.device)
                geometry_sample_weight = auxiliary_sample_weight(
                    timesteps,
                    noise_scheduler.num_train_timesteps,
                    max_timestep_fraction=geometry_timestep_fraction,
                ).to(accelerator.device)
                reversal_event_sample_weight = auxiliary_sample_weight(
                    timesteps,
                    noise_scheduler.num_train_timesteps,
                    max_timestep_fraction=bow_reversal_event_timestep_fraction,
                ).to(accelerator.device)
                aux_ramp = ramp_weight(
                    global_step,
                    start_step=geometry_loss_start_step,
                    warmup_steps=auxiliary_loss_warmup_steps,
                )
                structure_pose_mask = active_pose_mask * predicted_frames[:, None, :, None, None] * structure_sample_weight.reshape(
                    -1,
                    1,
                    1,
                    1,
                    1,
                )

                # Keep all pose/velocity/acceleration supervision in released
                # native coordinates; only the diffusion state is reparameterized.
                native_target_grid = decode_pose_grid(pose, pose_representation)
                pose_recon = weighted_masked_mse(native_pred_x0, native_target_grid, structure_pose_mask, joint_weights)
                vel_loss = temporal_velocity_loss_weighted(
                    pred_pose,
                    target_pose,
                    target_valid,
                    structure_sample_weight,
                    joint_weight=joint_weight_vector,
                    predicted_frames=predicted_frames,
                )
                acc_loss = temporal_acceleration_loss_weighted(
                    pred_pose,
                    target_pose,
                    target_valid,
                    structure_sample_weight,
                    joint_weight=joint_weight_vector,
                    predicted_frames=predicted_frames,
                )
                if bow_radial_velocity_loss_weight:
                    radial_velocity_sample_weight = auxiliary_sample_weight(
                        timesteps,
                        noise_scheduler.num_train_timesteps,
                        max_timestep_fraction=bow_radial_velocity_timestep_fraction,
                    ).to(accelerator.device)
                    bow_radial_velocity = bow_radial_velocity_loss_weighted(
                        pred_pose,
                        target_pose,
                        batch["strings"],
                        batch["bow_valid"],
                        target_valid=target_valid,
                        sample_weight=radial_velocity_sample_weight,
                        beta=bow_radial_velocity_loss_beta,
                        velocity_scale=bow_radial_velocity_loss_scale,
                    )
                else:
                    bow_radial_velocity = pred_pose.new_tensor(0.0)
                    radial_velocity_sample_weight = torch.zeros_like(structure_sample_weight)
                if bow_radial_velocity_direction_loss_weight:
                    bow_radial_velocity_direction = bow_radial_velocity_direction_loss_weighted(
                        pred_pose,
                        target_pose,
                        batch["strings"],
                        batch["bow_valid"],
                        target_valid=target_valid,
                        sample_weight=structure_sample_weight,
                        temperature=bow_radial_velocity_direction_temperature,
                        min_velocity=bow_radial_velocity_direction_min_velocity,
                    )
                else:
                    bow_radial_velocity_direction = pred_pose.new_tensor(0.0)
                if bow_endpoint_smoothness_loss_weight:
                    bow_endpoint_smoothness = bow_endpoint_smoothness_loss_weighted(
                        pred_pose,
                        target_pose,
                        batch["strings"],
                        batch["bow_valid"],
                        target_valid=target_valid,
                        sample_weight=structure_sample_weight,
                        window=bow_endpoint_smoothness_window,
                        exclude_radius=bow_endpoint_smoothness_exclude_radius,
                        min_velocity=bow_endpoint_smoothness_min_velocity,
                    )
                else:
                    bow_endpoint_smoothness = pred_pose.new_tensor(0.0)
                if bow_endpoint_radial_stable_sign_loss_weight:
                    bow_endpoint_radial_stable_sign = (
                        bow_endpoint_radial_stable_sign_loss_weighted(
                            pred_pose,
                            target_pose,
                            batch["strings"],
                            batch["bow_valid"],
                            target_valid=target_valid,
                            sample_weight=structure_sample_weight,
                            stable_window=bow_endpoint_radial_stable_sign_window,
                            exclude_radius=bow_endpoint_radial_stable_sign_exclude_radius,
                            min_velocity=bow_endpoint_radial_stable_sign_min_velocity,
                            margin=bow_endpoint_radial_stable_sign_margin,
                        )
                    )
                else:
                    bow_endpoint_radial_stable_sign = pred_pose.new_tensor(0.0)
                if bow_reversal_event_loss_weight:
                    bow_reversal_event = bow_reversal_event_direction_loss_weighted(
                        pred_pose,
                        target_pose,
                        batch["relation"],
                        batch["relation_mask"],
                        batch["strings"],
                        batch["bow_valid"],
                        target_valid=target_valid,
                        sample_weight=reversal_event_sample_weight,
                        temperature=bow_reversal_event_temperature,
                        target_min_velocity=bow_reversal_event_target_min_velocity,
                        relation_min_velocity=bow_reversal_event_relation_min_velocity,
                        relation_tolerance_frames=bow_reversal_event_relation_tolerance_frames,
                    )
                else:
                    bow_reversal_event = pred_pose.new_tensor(0.0)
                hand_local = left_hand_local_loss_weighted(
                    pred_pose,
                    target_pose,
                    target_valid,
                    structure_sample_weight,
                    wrist_index=int(left_hand_local_wrist_index),
                    joint_indices=left_hand_local_joint_indices,
                )
                wrist_offset = body_hand_wrist_offset_loss_weighted(
                    pred_pose,
                    target_pose,
                    target_valid,
                    structure_sample_weight,
                )
                if contact_string_ce_loss_weight:
                    contact_string_ce = contact_string_ce_loss(
                        pred_pose,
                        batch,
                        target_pose=target_pose,
                        target_valid=target_valid,
                        contact_fingertip_indices=contact_fingertip_indices,
                        temperature=contact_string_ce_temperature,
                        contact_finger_supervision=contact_finger_supervision,
                        sample_weight=geometry_sample_weight,
                    )
                else:
                    contact_string_ce = pred_pose.new_tensor(0.0)
                geom_losses = playable_geometry_loss(
                    pred_pose,
                    batch,
                    target_pose=target_pose,
                    target_valid=target_valid,
                    contact_fingertip_indices=contact_fingertip_indices,
                    contact_distance_loss=contact_distance_loss,
                    contact_finger_supervision=contact_finger_supervision,
                    sample_weight=geometry_sample_weight,
                )

                if aux_ramp > 0.0:
                    if pose_reconstruction_weight:
                        loss = loss + aux_ramp * pose_reconstruction_weight * pose_recon
                    if velocity_loss_weight:
                        loss = loss + aux_ramp * velocity_loss_weight * vel_loss
                    if acceleration_loss_weight:
                        loss = loss + aux_ramp * acceleration_loss_weight * acc_loss
                    if bow_radial_velocity_loss_weight:
                        loss = loss + aux_ramp * bow_radial_velocity_loss_weight * bow_radial_velocity
                    if bow_radial_velocity_direction_loss_weight:
                        loss = (
                            loss
                            + aux_ramp
                            * bow_radial_velocity_direction_loss_weight
                            * bow_radial_velocity_direction
                        )
                    if bow_endpoint_smoothness_loss_weight:
                        loss = (
                            loss
                            + aux_ramp
                            * bow_endpoint_smoothness_loss_weight
                            * bow_endpoint_smoothness
                        )
                    if bow_endpoint_radial_stable_sign_loss_weight:
                        loss = (
                            loss
                            + aux_ramp
                            * bow_endpoint_radial_stable_sign_loss_weight
                            * bow_endpoint_radial_stable_sign
                        )
                    if bow_reversal_event_loss_weight:
                        loss = (
                            loss
                            + aux_ramp
                            * bow_reversal_event_loss_weight
                            * bow_reversal_event
                        )
                    if left_hand_local_loss_weight:
                        loss = loss + aux_ramp * left_hand_local_loss_weight * hand_local
                    if body_hand_wrist_offset_loss_weight:
                        loss = loss + aux_ramp * body_hand_wrist_offset_loss_weight * wrist_offset
                    loss = loss + aux_ramp * contact_geometry_loss_weight * geom_losses["contact"]
                    if contact_string_ce_loss_weight:
                        loss = loss + aux_ramp * contact_string_ce_loss_weight * contact_string_ce
                    loss = loss + aux_ramp * bow_position_loss_weight * geom_losses["bow_position"]
                    loss = loss + aux_ramp * bow_direction_loss_weight * geom_losses["bow_direction"]
                    if bow_string_loss_weight:
                        loss = loss + aux_ramp * bow_string_loss_weight * geom_losses["bow_string"]

                avg_loss = accelerator.gather(loss.detach().repeat(bsz)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    clip_parameters = list(condition_encoder.parameters())
                    if not joint_freeze_unet:
                        clip_parameters = list(unet.parameters()) + clip_parameters
                    if joint_mapping_model is not None:
                        clip_parameters += [
                            parameter
                            for parameter in joint_mapping_model.parameters()
                            if parameter.requires_grad
                        ]
                    if geometry_refiner is not None:clip_parameters+=list(geometry_refiner.parameters())
                    accelerator.clip_grad_norm_(clip_parameters, max_grad_norm)
                optimizer.step()
                lr_sched.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if geometry_refiner is not None and accelerator.is_main_process and (global_step==1 or global_step%joint_geometry_options['log_interval']==0):
                    record=dict(step=global_step,weighted_joint_geometry_loss=float(joint_geometry_loss.detach()),
                                noise_loss=float(noise_loss.detach()),**{k:float(v.detach()) for k,v in joint_geometry_terms.items()})
                    with (Path(output_dir)/'joint_geometry_diagnostics.jsonl').open('a') as handle:
                        handle.write(json.dumps(record,allow_nan=False)+'\n')
                    print('Joint geometry:',record,flush=True)
                set_dataset_training_step(train_dataset, global_step)
                radial_value = float(bow_radial_velocity.detach().float().item())
                accelerator.log({
                    "train_loss": train_loss,
                    "lr": lr_sched.get_last_lr()[0],
                    "bow_radial_velocity_loss_last_micro": radial_value,
                    "bow_radial_velocity_contribution_last_micro": float(aux_ramp) * float(bow_radial_velocity_loss_weight) * radial_value,
                    "bow_radial_velocity_sample_fraction_last_micro": float(radial_velocity_sample_weight.float().mean().item()),
                }, step=global_step)
                if (
                    int(bow_radial_velocity_log_interval) > 0
                    and global_step % int(bow_radial_velocity_log_interval) == 0
                    and accelerator.is_main_process
                ):
                    contribution = float(aux_ramp) * float(bow_radial_velocity_loss_weight) * radial_value
                    diagnostic = {
                        "global_step": global_step,
                        "total_loss_last_micro": float(loss.detach().float().item()),
                        "noise_loss_last_micro": float(noise_loss.detach().float().item()),
                        "radial_loss_normalized_last_micro": radial_value,
                        "radial_loss_contribution_last_micro": contribution,
                        "radial_sample_fraction_last_micro": float(radial_velocity_sample_weight.float().mean().item()),
                        "velocity_scale": float(bow_radial_velocity_loss_scale),
                        "timestep_fraction": float(bow_radial_velocity_timestep_fraction),
                        "loss_weight": float(bow_radial_velocity_loss_weight),
                    }
                    with (Path(output_dir) / "radial_velocity_diagnostics.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(diagnostic, sort_keys=True) + "\n")
                train_loss = 0.0

                if global_step % checkpointing_steps == 0:
                    save_checkpoint(
                        output_dir,
                        global_step,
                        accelerator,
                        condition_encoder,
                        mapping_model=joint_mapping_model,
                        geometry_refiner=geometry_refiner,
                        lr_sched=lr_sched,
                    )

                if use_validation and validation_dataloader is not None and global_step % validation_steps == 0:
                    if validation_full_sequences:
                        from scripts.validate_syncviolinist_pose import run_sequence_validation
                        val_metrics = run_sequence_validation(
                            accelerator, unet, condition_encoder, noise_scheduler,
                            validation_dataset, output_dir, global_step, config,
                            prefix_frames=int(validation_prefix_frames),
                            sampling_steps=int(validation_sampling_steps),
                            sampling_seed=int(validation_sampling_seed),
                            max_batches=validation_sequence_max_batches,
                        )
                    else:
                        val_metrics = run_validation(
                            accelerator,
                            unet,
                            condition_encoder,
                            noise_scheduler,
                            validation_dataloader,
                            output_dir,
                            global_step,
                            weight_dtype,
                            max_batches=validation_max_batches,
                            pose_grid_size=pose_grid_size,
                            body_joint_weight=body_joint_weight,
                            left_hand_joint_weight=left_hand_joint_weight,
                            left_hand_joint_indices=left_hand_joint_indices,
                            right_hand_joint_weight=right_hand_joint_weight,
                            right_hand_joint_indices=right_hand_joint_indices,
                            bow_joint_weight=bow_joint_weight,
                            bow_joint_indices=bow_joint_indices,
                            fingertip_joint_weight=fingertip_joint_weight,
                            fingertip_joint_indices=fingertip_joint_indices,
                            face_joint_weight=face_joint_weight,
                            face_joint_indices=face_joint_indices,
                            lower_body_joint_weight=lower_body_joint_weight,
                            lower_body_joint_indices=lower_body_joint_indices,
                            torso_anchor_joint_weight=torso_anchor_joint_weight,
                            torso_anchor_joint_indices=torso_anchor_joint_indices,
                            ignored_joint_weight=ignored_joint_weight,
                            ignored_joint_indices=ignored_joint_indices,
                            condition_encoder_type=condition_encoder_type,
                            dual_condition_include_relation_mask=dual_condition_include_relation_mask,
                            relation_condition_mode=relation_condition_mode,
                            relation_temporal_features=relation_temporal_features,
                            relation_auxiliary_features=relation_auxiliary_features,
                            bow_t_velocity_temperature=bow_t_velocity_temperature,
                            sampling_steps=validation_sampling_steps,
                            sampling_max_batches=validation_sampling_max_batches,
                            sampling_seed=validation_sampling_seed,
                            sampled_metrics_per_piece=validation_sampled_metrics_per_piece,
                            geometry_refiner=geometry_refiner,
                            joint_geometry_options=joint_geometry_options,
                        )
                    if best_checkpoint_metric:
                        metric_value = val_metrics.get(best_checkpoint_metric)
                        if metric_value is None:
                            if accelerator.is_main_process:
                                logger.warning(
                                    f"Best checkpoint metric '{best_checkpoint_metric}' was not available "
                                    f"at step {global_step}."
                                )
                        else:
                            metric_value = float(metric_value)
                            if metric_is_better(metric_value, best_metric_value, str(best_checkpoint_mode)):
                                best_metric_value = metric_value
                                metadata = {
                                    "global_step": int(global_step),
                                    "best_checkpoint_metric": str(best_checkpoint_metric),
                                    "best_checkpoint_mode": str(best_checkpoint_mode),
                                    "best_metric_value": metric_value,
                                    "pose_representation_id": (
                                        None if pose_representation is None else pose_representation["profile_id"]
                                    ),
                                    "validation_metrics": val_metrics,
                                }
                                save_best_checkpoint(
                                    output_dir,
                                    global_step,
                                    accelerator,
                                    condition_encoder,
                                    metadata,
                                    mapping_model=joint_mapping_model,
                                    geometry_refiner=geometry_refiner,
                                    lr_sched=lr_sched,
                                )

            logs = {
                "step_loss": loss.detach().item(),
                "noise": noise_loss.detach().item(),
                "context_frame_fraction": float((~predicted_frames).float().mean().item()),
                "contact": geom_losses["contact"].detach().item(),
                "cstr": contact_string_ce.detach().item(),
                "bow_pos": geom_losses["bow_position"].detach().item(),
                "bow_str": geom_losses["bow_string"].detach().item(),
                "bow_rvel": bow_radial_velocity.detach().item(),
                "bow_rvel_w": float(aux_ramp) * float(bow_radial_velocity_loss_weight) * bow_radial_velocity.detach().item(),
                "rv_mask": radial_velocity_sample_weight.detach().mean().item(),
                "bow_rvdir": bow_radial_velocity_direction.detach().item(),
                "bow_epsm": bow_endpoint_smoothness.detach().item(),
                "bow_rssign": bow_endpoint_radial_stable_sign.detach().item(),
                "bow_rev": bow_reversal_event.detach().item(),
                "bow_rev_mask": reversal_event_sample_weight.detach().mean().item(),
                "wrist": wrist_offset.detach().item(),
                "vel": vel_loss.detach().item(),
                "struct_mask": structure_sample_weight.detach().mean().item(),
                "geom_mask": geometry_sample_weight.detach().mean().item(),
                "aux_w": float(aux_ramp),
                "lr": lr_sched.get_last_lr()[0],
            }
            unwrapped_encoder = accelerator.unwrap_model(condition_encoder)
            relation_gate = getattr(unwrapped_encoder, "relation_gate", None)
            if relation_gate is not None:
                logs["rel_gate"] = relation_gate.detach().float().item()
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_encoder = accelerator.unwrap_model(condition_encoder)
        unwrapped_unet.save_pretrained(os.path.join(output_dir, "unet"))
        torch.save(unwrapped_encoder.state_dict(), os.path.join(output_dir, "condition_encoder.pt"))
        _save_joint_mapping_checkpoint(
            output_dir,
            accelerator.unwrap_model(joint_mapping_model) if joint_mapping_model is not None else None,
        )
        if not (resume_from_checkpoint and joint_geometry_options is not None and (Path(output_dir) / 'config.yaml').is_file()):
            OmegaConf.save(config, os.path.join(output_dir, "config.yaml"))
        if geometry_refiner is not None:
            from audio2pose.training.joint_geometry import save_refiner
            save_refiner(output_dir,accelerator.unwrap_model(geometry_refiner),global_step)
        logger.info(f"Saved final model to {output_dir}")

    # Older Accelerate releases expose end_training() without initializing the
    # trackers attribute when no tracker backend is configured. The model has
    # already been saved at this point, so skip tracker shutdown in that case.
    if hasattr(accelerator, "trackers"):
        accelerator.end_training()


# Prepare frozen relations and launch the Stage-2 trainer.

_PRETRAINED_AUDIO_KEYS = (
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


def _plain(value: Any) -> Any:
    if hasattr(value, "_metadata"):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_plain(item) for item in value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)


def _normalise_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _file_sha256(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def _prepare_trace_run(stage2,context):
    """Freeze new-training provenance before the potentially expensive cache pass."""
    if stage2['train_data'].get('mode')!='train' or (stage2.get('validation_data') or {}).get('mode')!='val' or not stage2.get('use_validation'):
        raise ValueError('Joint bow trace needs train-only optimization and a separate val split')
    if not context.get('train_from_scratch') or context.get('stage2_checkpoint_path'):
        raise ValueError('This recipe initializes Stage-2 once from random weights')
    output=_normalise_path(context['output_dir'])
    if any(output.glob('checkpoint-*')) or (output/'joint_geometry_diagnostics.jsonl').exists() or (output/'unet').exists():
        raise FileExistsError(f'Training artifacts already exist at {output}; do not restart a fresh model over this run')
    checkpoint=_normalise_path(context['mapping_checkpoint_path'])
    identity=dict(mapping_checkpoint_path=str(checkpoint),sha256=_file_sha256(checkpoint))
    cache=_normalise_path(context['predicted_relation_dataset_dir']);stamp=cache/'stage1_frozen_identity.json'
    if stamp.exists():
        if json.loads(stamp.read_text())!=identity:raise ValueError('Frozen Stage-1 weights changed; use a new predicted-relation cache directory')
    elif (cache/'manifest.json').exists():
        raise ValueError('This new recipe requires a cache with recorded Stage-1 identity; use its dedicated new directory')
    else:
        cache.mkdir(parents=True,exist_ok=True);_write_json(stamp,identity)
    source_files=('scripts/train_stage2.py','scripts/train_stage1.py','audio2pose/training/__init__.py',
        'audio2pose/pipelines/predicted_relations.py','audio2pose/models/audio_attention.py',
        'audio2pose/training/joint_geometry.py','audio2pose/losses/bow_trace.py',
        'audio2pose/models/local_bow_refiner.py','audio2pose/models/condition_encoder.py','audio2pose/models/unet.py',
        'audio2pose/data/data.py','audio2pose/metrics/elgar_original.py')
    hashes={name:_file_sha256(PROJECT_ROOT/name) for name in source_files}
    for name in source_files:
        destination=output/'source_snapshot'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(PROJECT_ROOT/name,destination)
    _write_json(output/'input_config.json',context)
    return dict(recipe='joint_bow_trace_train_only',stage1=identity,training_code_sha256=hashes,
        runtime=dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,cuda=torch.version.cuda))


def _checkpoint_step(checkpoint_path: str | Path) -> int:
    path = _normalise_path(checkpoint_path)
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match:
        return int(match.group(1))
    raise ValueError(
        f"Cannot resume from '{path}'. The Stage-2 trainer requires a "
        "checkpoint-N directory, for example checkpoint-100000."
    )


def _validate_trace_resume(stage2, context, checkpoint):
    """Check an existing run without modifying its original provenance or cache."""
    output = _normalise_path(context['output_dir'])
    if checkpoint.parent != output:
        raise ValueError('Full joint bow trace resume must use a checkpoint inside the original output_dir')
    if context.get('rebuild_predicted_relations'):
        raise ValueError('Do not rebuild the frozen Stage-1 cache during full resume')
    # Accelerate saves these three models in prepare() order, separately from
    # the inference-only condition_encoder.pt / local_bow_refiner.pt exports.
    required = ['optimizer.bin', 'random_states_0.pkl', 'config.yaml']
    scheduler_files = [checkpoint / name for name in ('scheduler.bin', 'lr_scheduler.pt')]
    if any(path.exists() and path.stat().st_size == 0 for path in scheduler_files):
        raise ValueError('Empty scheduler state file; choose an earlier complete checkpoint')
    if not any(path.is_file() for path in scheduler_files):
        if (stage2.get('lr_scheduler', 'constant') != 'constant'
                or int(stage2.get('lr_warmup_steps', 0)) != 0
                or int(stage2.get('gradient_accumulation_steps', 1)) != 1):
            raise FileNotFoundError('Missing scheduler state: only constant LR with zero warmup and accumulation=1 can be reconstructed')
        print('Scheduler state absent: will reconstruct constant LR at the checkpoint step after restoring AdamW')
    for suffix in ('', '_1', '_2'):
        alternatives = [checkpoint / f'pytorch_model{suffix}.bin', checkpoint / f'model{suffix}.safetensors']
        if not any(path.is_file() and path.stat().st_size > 0 for path in alternatives):
            raise FileNotFoundError(f'Incomplete full-state checkpoint: missing Accelerate model{suffix} at {checkpoint}')
    if stage2.get('mixed_precision') == 'fp16':
        required.append('scaler.pt')
    for name in required:
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f'Incomplete full-state checkpoint: missing {path}; choose an earlier complete checkpoint')
    previous_input = json.loads((output / 'input_config.json').read_text())
    ignored = {'resume_from_checkpoint', 'resume_optimizer_state', 'configuration_file'}
    canonical = lambda value: json.loads(json.dumps(_plain(value), default=str))
    before = {key: value for key, value in previous_input.items() if key not in ignored}
    after = {key: value for key, value in context.items() if key not in ignored}
    if canonical(before) != canonical(after):
        changed = sorted(key for key in before.keys() | after.keys() if canonical(before.get(key)) != canonical(after.get(key)))
        raise ValueError(f'Resume must preserve the original training recipe; changed input keys: {changed}')
    expected = dict(stage2, output_dir=str(output), resume_from_checkpoint=None,
                    max_train_steps=int(context['train_from_scratch_steps'] or stage2['max_train_steps']),
                    learning_rate=float(context['condition_encoder_learning_rate']), scale_lr=False)
    previous = _plain(OmegaConf.load(output / 'resolved_stage2_config.yaml'))
    if canonical(previous) != canonical(expected):
        raise ValueError('The resolved Stage-2 recipe changed (base config, data, loss, LR or total steps); restore the original config')
    metadata = json.loads((output / 'robust_finetune_metadata.json').read_text())
    provenance = metadata['joint_bow_trace_provenance']
    identity = dict(mapping_checkpoint_path=str(_normalise_path(context['mapping_checkpoint_path'])),
                    sha256=_file_sha256(context['mapping_checkpoint_path']))
    cache = _normalise_path(context['predicted_relation_dataset_dir'])
    if identity != provenance['stage1'] or json.loads((cache / 'stage1_frozen_identity.json').read_text()) != identity:
        raise ValueError('Frozen Stage-1 weights or cache identity changed')
    for name in ('metadata.json', 'manifest.json', 'splits/end_to_end.json'):
        if not (cache / name).is_file():
            raise FileNotFoundError(f'Missing original frozen cache file: {cache / name}')
    for name, digest in provenance['training_code_sha256'].items():
        # Legacy entry points can differ from the historical snapshots.
        # Current entry points and model, loss, and data code are checked.
        if name not in ('scripts/finetune_audio2pose_predicted_relations.py', 'scripts/train_audio2pose.py'):
            source = PROJECT_ROOT / name
            if not source.is_file() or _file_sha256(source) != digest:
                raise ValueError(f'Training implementation changed since this run: {name}. Use the original source snapshot.')
    return provenance


def _mapping_source_kwargs(
    train_data: Mapping[str, Any],
    num_workers: Optional[int],
    mapping_pretrained_audio_dim: int,
    mapping_sequence_length: Optional[int] = None,
    mapping_window_stride: Optional[int] = None,
) -> Dict[str, Any]:
    """Build the Stage-1 source dataset using the mapping checkpoint's inputs."""
    kwargs = dict(_plain(train_data))
    if int(mapping_pretrained_audio_dim) > 0:
        if not kwargs.get("audio_encoder_type"):
            raise ValueError(
                "The mapping checkpoint was trained with pretrained_audio_dim > 0, "
                "but train_data does not define audio_encoder_type. Add the same "
                "audio_encoder_type/audio_feature_cache_dir settings used by Stage 1."
            )
    else:
        # Old mapping checkpoints only consume mel/onset/pitch.  Strip pretrained
        # audio options so Stage-1 prediction does not load unused MERT features.
        for key in _PRETRAINED_AUDIO_KEYS:
            kwargs.pop(key, None)
    kwargs["include_relation_mask_in_condition"] = False
    # Extra scalar channels belong to Stage-2; the frozen Stage-1 checkpoint
    # still consumes the original 23-D relation target.
    kwargs.pop("relation_extra_keys", None)
    if mapping_sequence_length is not None:
        kwargs["sequence_length"] = int(mapping_sequence_length)
    if mapping_window_stride is not None:
        kwargs["window_stride"] = int(mapping_window_stride)
    if num_workers is not None:
        kwargs["num_workers"] = int(num_workers)
    return kwargs


def _normalise_slices(
    relation_dim: int,
    predicted_relation_slices: Optional[Sequence[Sequence[int]]],
) -> Sequence[Sequence[int]]:
    _, resolved = _predicted_relation_channel_mask(
        int(relation_dim),
        predicted_relation_slices,
    )
    return resolved


def _normalise_relation_group_probabilities(
    relation_dim: int,
    relation_group_probabilities: Optional[Sequence[Mapping[str, Any]]],
    default_probability: float,
) -> Sequence[Dict[str, Any]]:
    """Resolve per-relation-group prediction probabilities for fine-tuning."""
    if relation_group_probabilities is None:
        return [
            {
                "name": "all",
                "slices": [[0, int(relation_dim)]],
                "probability": float(default_probability),
            }
        ]

    groups = []
    for index, group in enumerate(relation_group_probabilities):
        if not isinstance(group, Mapping):
            raise TypeError(
                "relation_group_probabilities entries must be mappings with "
                "'slices' and 'probability'."
            )
        probability = float(group.get("probability", default_probability))
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"relation_group_probabilities[{index}].probability must be in [0, 1]."
            )
        slices = group.get("slices", group.get("predicted_relation_slices"))
        if slices is None:
            raise KeyError(
                f"relation_group_probabilities[{index}] must define 'slices'."
            )
        groups.append(
            {
                "name": str(group.get("name", f"group_{index}")),
                "slices": _normalise_slices(int(relation_dim), slices),
                "probability": probability,
            }
        )
    return groups


def _normalise_relation_mix_curriculum(
    curriculum: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Validate a step-based schedule for selected relation groups."""
    if curriculum is None:
        return None
    curriculum = _plain(curriculum)
    if not isinstance(curriculum, Mapping):
        raise TypeError("relation_mix_curriculum must be a mapping.")

    groups = curriculum.get("groups", ["bow_continuous"])
    if isinstance(groups, str):
        groups = [groups]
    groups = [str(name) for name in groups]
    if not groups:
        raise ValueError("relation_mix_curriculum.groups must not be empty.")

    raw_stages = curriculum.get("stages")
    if not isinstance(raw_stages, Sequence) or isinstance(raw_stages, (str, bytes)):
        raise TypeError(
            "relation_mix_curriculum.stages must be a list of mappings with "
            "until_step and probability."
        )
    stages = []
    previous_step = -1
    for index, stage in enumerate(raw_stages):
        if not isinstance(stage, Mapping):
            raise TypeError(f"relation_mix_curriculum.stages[{index}] must be a mapping.")
        if "until_step" not in stage or "probability" not in stage:
            raise KeyError(
                f"relation_mix_curriculum.stages[{index}] must define "
                "until_step and probability."
            )
        until_step = int(stage["until_step"])
        probability = float(stage["probability"])
        if until_step <= previous_step:
            raise ValueError(
                "relation_mix_curriculum.stages.until_step must be strictly increasing."
            )
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"relation_mix_curriculum.stages[{index}].probability must be in [0, 1]."
            )
        stages.append({"until_step": until_step, "probability": probability})
        previous_step = until_step
    if not stages:
        raise ValueError("relation_mix_curriculum.stages must not be empty.")
    return {"groups": groups, "stages": stages}


def _predicted_dataset_is_compatible(
    output_dir: Path,
    mapping_checkpoint_path: str,
    source_dataset: SPDCanonicalPoseDataset,
    predicted_relation_slices: Optional[Sequence[Sequence[int]]],
    mapping_categorical_mode: str,
    mapping_clamp_continuous: bool,
    predicted_relation_mask_mode: str,
    predicted_validity_threshold: float,
    relation_dropped_slices: Optional[Sequence[Sequence[int]]],
    hybrid_relation_mask_source: str,
    mapping_sequence_length: int,
    mapping_window_stride: int,
    mapping_stitching_mode: str,
    mapping_stitching_min_weight: float,
    predicted_relation_smoothing_window: int = 0,
    predicted_relation_smoothing_slices: Optional[Sequence[Sequence[int]]] = None,
    predicted_scalar_key: Optional[str] = None,
) -> bool:
    metadata_path = output_dir / "metadata.json"
    manifest_path = output_dir / "manifest.json"
    split_path = output_dir / "splits" / "end_to_end.json"
    if not (metadata_path.exists() and manifest_path.exists() and split_path.exists()):
        return False

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    cached_checkpoint = metadata.get("mapping_checkpoint")
    expected_checkpoint = str(_normalise_path(mapping_checkpoint_path))
    if cached_checkpoint != expected_checkpoint:
        raise RuntimeError(
            "The cached predicted-relation dataset was generated by a different mapping checkpoint:\n"
            f"  cached:   {cached_checkpoint}\n"
            f"  requested:{expected_checkpoint}\n"
            "Choose a new predicted_relation_dataset_dir or set rebuild_predicted_relations=true."
        )
    if int(metadata.get("num_records", -1)) != len(source_dataset.records):
        raise RuntimeError(
            "The cached predicted-relation dataset does not match the current stage-2 training split. "
            "Choose a new predicted_relation_dataset_dir or set rebuild_predicted_relations=true."
        )
    relation_dim = int(source_dataset.records[0].get("relation_feature_dim", 23))
    if predicted_scalar_key is not None:
        relation_dim += 1
    expected_slices = _normalise_slices(relation_dim, predicted_relation_slices)
    _, expected_dropped_slices = _dropped_relation_channel_mask(
        relation_dim,
        relation_dropped_slices,
    )
    if int(predicted_relation_smoothing_window) > 1:
        smoothing_slices = (
            predicted_relation_smoothing_slices
            if predicted_relation_smoothing_slices is not None
            else [[10, 12], [14, 20]]
        )
        _, expected_smoothing_slices = _predicted_relation_channel_mask(
            relation_dim,
            smoothing_slices,
        )
    else:
        expected_smoothing_slices = []
    # Caches written before the smoothing option existed have no smoothing
    # keys; they are equivalent to window=0 (no smoothing).
    cached_smoothing_window = metadata.get("predicted_relation_smoothing_window")
    cached_smoothing_window = 0 if cached_smoothing_window is None else int(cached_smoothing_window)
    cached_smoothing_slices = metadata.get("predicted_relation_smoothing_slices") or []
    expected_values = {
        "predicted_relation_slices": _plain(expected_slices),
        "relation_dropped_slices": _plain(expected_dropped_slices),
        "categorical_mode": str(mapping_categorical_mode),
        "clamp_continuous": bool(mapping_clamp_continuous),
        "relation_mask_mode": str(predicted_relation_mask_mode),
        "hybrid_relation_mask_source": str(hybrid_relation_mask_source),
        "validity_threshold": float(predicted_validity_threshold),
        "mapping_sequence_length": int(mapping_sequence_length),
        "mapping_window_stride": int(mapping_window_stride),
        "mapping_stitching_mode": str(mapping_stitching_mode),
        "mapping_stitching_min_weight": float(mapping_stitching_min_weight),
        "predicted_scalar_key": predicted_scalar_key,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected_values.items()
        if metadata.get(key) != value
    }
    if mismatches:
        details = "\n".join(
            f"  {key}: cached={cached!r}, requested={requested!r}"
            for key, (cached, requested) in mismatches.items()
        )
        raise RuntimeError(
            "The cached predicted-relation dataset was generated with different relation "
            "postprocess/mixing settings:\n"
            f"{details}\n"
            "Choose a new predicted_relation_dataset_dir or set rebuild_predicted_relations=true."
        )
    if (
        cached_smoothing_window != int(predicted_relation_smoothing_window)
        or _plain(cached_smoothing_slices) != _plain(expected_smoothing_slices)
    ):
        raise RuntimeError(
            "The cached predicted-relation dataset was generated with a different "
            "predicted-relation smoothing setting:\n"
            f"  cached:    window={cached_smoothing_window}, slices={_plain(cached_smoothing_slices)}\n"
            f"  requested: window={int(predicted_relation_smoothing_window)}, "
            f"slices={_plain(expected_smoothing_slices)}\n"
            "Choose a new predicted_relation_dataset_dir or set rebuild_predicted_relations=true."
        )
    return True


def prepare_predicted_relation_dataset(
    *,
    train_data: Mapping[str, Any],
    source_mode: Optional[str] = None,
    mapping_checkpoint_path: str,
    output_dir: str | Path,
    mapping_latent_dim: Optional[int],
    mapping_batch_size: int,
    mapping_num_workers: Optional[int],
    mapping_sequence_length: Optional[int],
    mapping_window_stride: Optional[int],
    mapping_stitching_mode: str,
    mapping_stitching_min_weight: float,
    mapping_categorical_mode: str,
    mapping_clamp_continuous: bool,
    predicted_relation_mask_mode: str,
    predicted_validity_threshold: float,
    predicted_relation_slices: Optional[Sequence[Sequence[int]]],
    relation_dropped_slices: Optional[Sequence[Sequence[int]]],
    hybrid_relation_mask_source: str,
    device: str,
    rebuild: bool,
    predicted_relation_smoothing_window: int = 0,
    predicted_relation_smoothing_slices: Optional[Sequence[Sequence[int]]] = None,
    predicted_scalar_key: Optional[str] = None,
    predicted_scalar_valid_key: Optional[str] = None,
) -> Tuple[Path, SPDCanonicalPoseDataset]:
    """Run frozen stage 1 once and create a shadow relation dataset."""
    output_path = _normalise_path(output_dir)
    mapping_path = _normalise_path(mapping_checkpoint_path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping checkpoint does not exist: {mapping_path}")
    mapping_meta = torch.load(str(mapping_path), map_location="cpu")
    mapping_pretrained_audio_dim = int(mapping_meta.get("pretrained_audio_dim", 0))
    mapping_onset_dim = int(mapping_meta.get("onset_dim", 1))

    source_kwargs = _mapping_source_kwargs(
        train_data,
        mapping_num_workers,
        mapping_pretrained_audio_dim=mapping_pretrained_audio_dim,
        mapping_sequence_length=mapping_sequence_length,
        mapping_window_stride=mapping_window_stride,
    )
    if mapping_onset_dim > 1:
        # Dynamics-aware mapping checkpoints consume 1+4 onset channels. The
        # frozen Stage-1 source dataset needs them, while Stage 2's own
        # training dataset keeps the original 1-channel onset so the UNet
        # condition width is unchanged relative to previous runs.
        source_kwargs["use_bow_dynamics"] = True
    if source_mode is not None:
        source_kwargs["mode"] = source_mode
    source_dataset = SPDCanonicalPoseDataset(**source_kwargs)
    # Fail fast before the Stage-1 DataLoader starts: dynamics-aware
    # checkpoints need 'dynamics' arrays in every record's audio cache, and
    # cache upgrades are split-specific (see _verify_dynamics_cache).
    _verify_dynamics_cache(source_dataset, str(source_kwargs.get("mode", "train")))
    if not rebuild and _predicted_dataset_is_compatible(
        output_path,
        str(mapping_path),
        source_dataset,
        predicted_relation_slices=predicted_relation_slices,
        mapping_categorical_mode=mapping_categorical_mode,
        mapping_clamp_continuous=bool(mapping_clamp_continuous),
        predicted_relation_mask_mode=predicted_relation_mask_mode,
        predicted_validity_threshold=float(predicted_validity_threshold),
        relation_dropped_slices=relation_dropped_slices,
        hybrid_relation_mask_source=hybrid_relation_mask_source,
        mapping_sequence_length=int(source_kwargs["sequence_length"]),
        mapping_window_stride=int(source_kwargs["window_stride"]),
        mapping_stitching_mode=mapping_stitching_mode,
        mapping_stitching_min_weight=float(mapping_stitching_min_weight),
        predicted_relation_smoothing_window=int(predicted_relation_smoothing_window),
        predicted_relation_smoothing_slices=predicted_relation_smoothing_slices,
        predicted_scalar_key=predicted_scalar_key,
    ):
        print(f"Reusing predicted-relation dataset: {output_path}")
        return output_path, source_dataset

    if rebuild and output_path.exists():
        # build_predicted_relation_dataset overwrites its own files.  It is
        # intentionally not allowed to recursively delete this user-selected
        # directory; stale sample folders are harmless because the new
        # manifest is authoritative.
        print(f"Rebuilding predicted relations in: {output_path}")

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; stage-1 prediction will run on CPU.")
        device = "cpu"
    device_obj = torch.device(device)
    workers = int(source_kwargs.get("num_workers", 4))
    loader = DataLoader(
        source_dataset,
        batch_size=int(mapping_batch_size),
        shuffle=False,
        num_workers=workers,
        pin_memory=device_obj.type == "cuda",
        persistent_workers=workers > 0,
    )
    sample = source_dataset[0]
    mapping_model = load_mapping_model(
        checkpoint_path=str(mapping_path),
        mel_dim=int(sample["mel"].shape[-1]),
        relation_dim=int(sample["relation"].shape[-1]),
        latent_dim=mapping_latent_dim,
        device=device_obj,
    )
    for parameter in mapping_model.parameters():
        parameter.requires_grad_(False)
    mapping_model.eval()

    relation_key = str(source_kwargs.get("relation_key", "relation_feature_zero_filled"))
    relation_mask_key = str(source_kwargs.get("relation_mask_key", "relation_feature_valid_mask"))
    metrics, accumulators = _stage1_metrics(
        mapping_model,
        loader,
        device_obj,
        relation_key=relation_key,
        mapping_stitching_mode=mapping_stitching_mode,
        mapping_stitching_min_weight=float(mapping_stitching_min_weight),
    )
    _write_json(output_path / "stage1_training_split_metrics.json", metrics)
    build_predicted_relation_dataset(
        source_dataset=source_dataset,
        accumulators=accumulators,
        output_dir=output_path,
        relation_key=relation_key,
        relation_mask_key=relation_mask_key,
        categorical_mode=mapping_categorical_mode,
        clamp_continuous=bool(mapping_clamp_continuous),
        relation_mask_mode=predicted_relation_mask_mode,
        mapping_checkpoint_path=str(mapping_path),
        validity_threshold=float(predicted_validity_threshold),
        predicted_relation_slices=predicted_relation_slices,
        relation_dropped_slices=relation_dropped_slices,
        hybrid_relation_mask_source=hybrid_relation_mask_source,
        mapping_sequence_length=int(source_kwargs["sequence_length"]),
        mapping_window_stride=int(source_kwargs["window_stride"]),
        mapping_stitching_mode=mapping_stitching_mode,
        mapping_stitching_min_weight=float(mapping_stitching_min_weight),
        predicted_relation_smoothing_window=int(predicted_relation_smoothing_window),
        predicted_relation_smoothing_slices=predicted_relation_smoothing_slices,
        predicted_scalar_key=predicted_scalar_key,
        predicted_scalar_valid_key=predicted_scalar_valid_key,
    )
    del mapping_model
    if device_obj.type == "cuda":
        torch.cuda.empty_cache()
    print(f"Predicted-relation dataset written to: {output_path}")
    return output_path, source_dataset


class MixedRelationPoseDataset(Dataset):
    """Use GT targets, but randomly replace the stage-2 relation condition.

    ``relation_mix_mode='group'`` preserves the historical behaviour where
    relation groups can be replaced independently.  ``'whole_vector'`` uses
    one decision for the complete relation vector.  ``'matched_joint'`` keeps
    each configured group's predicted marginal probability unchanged while
    maximally coupling the named joint groups, so it tests joint coherence
    without changing per-channel exposure rates.
    """

    def __init__(
        self,
        original_dataset_cls,
        predicted_dataset_dir: str | Path,
        predicted_probability: float,
        relation_group_probabilities: Optional[Sequence[Mapping[str, Any]]] = None,
        relation_mix_curriculum: Optional[Mapping[str, Any]] = None,
        relation_mix_granularity: str = "window",
        relation_mix_mode: str = "group",
        relation_joint_groups: Optional[Sequence[Sequence[str]]] = None,
        relation_dropped_slices: Optional[Sequence[Sequence[int]]] = None,
        **ground_truth_kwargs: Any,
    ):
        probability = float(predicted_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("predicted_relation_probability must be in [0, 1].")
        self.predicted_probability = probability
        self.relation_mix_granularity = str(relation_mix_granularity)
        if self.relation_mix_granularity not in {"window", "frame"}:
            raise ValueError("relation_mix_granularity must be 'window' or 'frame'.")
        self.relation_mix_mode = str(relation_mix_mode).lower()
        if self.relation_mix_mode not in {"group", "whole_vector", "matched_joint"}:
            raise ValueError(
                "relation_mix_mode must be 'group', 'whole_vector', or 'matched_joint'."
            )
        self.relation_joint_groups = [
            [str(name) for name in names]
            for names in (relation_joint_groups or [])
        ]
        if self.relation_mix_mode == "matched_joint":
            if not self.relation_joint_groups:
                raise ValueError(
                    "matched_joint mode requires relation_joint_groups, for example "
                    "[[bow_t, bow_geometry_without_t]]."
                )
            if any(len(names) < 2 for names in self.relation_joint_groups):
                raise ValueError("Each relation_joint_groups entry must contain at least two group names.")
            flattened = [name for names in self.relation_joint_groups for name in names]
            if len(flattened) != len(set(flattened)):
                raise ValueError("A relation group cannot appear in multiple joint groups.")
        self.ground_truth = original_dataset_cls(**ground_truth_kwargs)

        predicted_kwargs = dict(ground_truth_kwargs)
        predicted_kwargs["data_path"] = str(_normalise_path(predicted_dataset_dir))
        predicted_kwargs["mode"] = "end_to_end"
        # The shadow archive already stores the scalar as its final relation
        # channel. Do not append the canonical extra field a second time.
        predicted_kwargs.pop("relation_extra_keys", None)
        self.predicted = original_dataset_cls(**predicted_kwargs)
        self._restrict_predicted_to_ground_truth_records()
        self.include_relation_mask_in_condition = bool(
            ground_truth_kwargs.get("include_relation_mask_in_condition", True)
        )
        self._validate_alignment()
        relation_dim = self._infer_relation_dim()
        _, self.relation_dropped_slices = _dropped_relation_channel_mask(
            relation_dim,
            relation_dropped_slices,
        )
        self.relation_group_probabilities = _normalise_relation_group_probabilities(
            relation_dim,
            relation_group_probabilities,
            default_probability=probability,
        )
        self.relation_mix_curriculum = _normalise_relation_mix_curriculum(
            relation_mix_curriculum
        )
        # DataLoader workers receive this tensor by reference. The unchanged
        # Stage-2 loop updates it once per optimization step, so workers can
        # use the same curriculum without rebuilding the dataset.
        self._training_step = torch.zeros(1, dtype=torch.int64).share_memory_()

        # The Stage-2 trainer reads these attributes directly when reporting
        # dataset dimensions and building its sampler.
        self.records = self.ground_truth.records
        self.windows = self.ground_truth.windows
        # Relation replacement changes conditioning only. The pose decoder
        # must use the same representation as the original GT pose targets;
        # validation also reads the original temporal window length.
        self.pose_representation = self.ground_truth.pose_representation
        self.sequence_length = self.ground_truth.sequence_length

    def _restrict_predicted_to_ground_truth_records(self) -> None:
        """Align a full train+val prediction cache to a train or val subset."""
        predicted_by_id = {}
        for index, record in enumerate(self.predicted.records):
            record_id = str(record.get("id", ""))
            if record_id in predicted_by_id:
                raise RuntimeError(f"Predicted relation cache contains duplicate record id '{record_id}'.")
            predicted_by_id[record_id] = index

        selected_records = []
        record_index_map = {}
        for gt_index, record in enumerate(self.ground_truth.records):
            record_id = str(record.get("id", ""))
            predicted_index = predicted_by_id.get(record_id)
            if predicted_index is None:
                raise RuntimeError(
                    f"Predicted relation cache does not contain validation/training record '{record_id}'."
                )
            record_index_map[gt_index] = len(selected_records)
            selected_records.append(self.predicted.records[predicted_index])

        starts_by_record = {}
        for record_index, start in self.predicted.windows:
            starts_by_record.setdefault(int(record_index), set()).add(int(start))

        selected_windows = []
        for gt_record_index, start in self.ground_truth.windows:
            record_id = str(self.ground_truth.records[gt_record_index].get("id", ""))
            predicted_record_index = predicted_by_id[record_id]
            if int(start) not in starts_by_record.get(predicted_record_index, set()):
                raise RuntimeError(
                    f"Predicted relation cache is missing window ({record_id}, {int(start)})."
                )
            selected_windows.append((record_index_map[gt_record_index], int(start)))

        self.predicted.records = selected_records
        self.predicted.windows = selected_windows

    def _infer_relation_dim(self) -> int:
        if not self.ground_truth.records:
            raise RuntimeError("The training dataset contains no records.")
        # Read the materialised sample so relation_extra_keys are included in
        # the dimension. Manifest metadata describes only the base 23-D field.
        sample = self.ground_truth[0]
        return int(sample["relation"].shape[-1])

    def _validate_alignment(self) -> None:
        if len(self.ground_truth) != len(self.predicted):
            raise RuntimeError(
                "GT and predicted-relation datasets contain different numbers of windows: "
                f"{len(self.ground_truth)} vs {len(self.predicted)}. Rebuild the prediction cache."
            )
        if len(self.ground_truth.records) != len(self.predicted.records):
            raise RuntimeError("GT and predicted-relation datasets contain different record counts.")

        for index, (gt_record, pred_record) in enumerate(
            zip(self.ground_truth.records, self.predicted.records)
        ):
            gt_id = str(gt_record.get("id", ""))
            pred_id = str(pred_record.get("id", ""))
            if gt_id != pred_id:
                raise RuntimeError(
                    f"Dataset record {index} is misaligned: GT='{gt_id}', predicted='{pred_id}'."
                )
        for index, (gt_window, pred_window) in enumerate(
            zip(self.ground_truth.windows, self.predicted.windows)
        ):
            gt_record_index, gt_start = gt_window
            pred_record_index, pred_start = pred_window
            gt_id = str(self.ground_truth.records[gt_record_index].get("id", ""))
            pred_id = str(self.predicted.records[pred_record_index].get("id", ""))
            if gt_id != pred_id or int(gt_start) != int(pred_start):
                raise RuntimeError(
                    f"Dataset window {index} is misaligned: "
                    f"GT=({gt_id}, {gt_start}), predicted=({pred_id}, {pred_start})."
                )

    def __len__(self) -> int:
        return len(self.ground_truth)

    def set_training_step(self, global_step: int) -> None:
        self._training_step.fill_(int(global_step))

    def _scheduled_relation_group_probabilities(self) -> Sequence[Dict[str, Any]]:
        if self.relation_mix_curriculum is None:
            return self.relation_group_probabilities

        step = int(self._training_step.item())
        stages = self.relation_mix_curriculum["stages"]
        probability = float(stages[-1]["probability"])
        for stage in stages:
            if step < int(stage["until_step"]):
                probability = float(stage["probability"])
                break
        scheduled_groups = set(self.relation_mix_curriculum["groups"])
        return [
            dict(group, probability=probability)
            if str(group["name"]) in scheduled_groups
            else group
            for group in self.relation_group_probabilities
        ]

    @staticmethod
    def _audio_condition_parts(item: Mapping[str, torch.Tensor]) -> Sequence[torch.Tensor]:
        parts = [item["mel"], item["onset"], item["pitch"]]
        if "pretrained_audio" in item:
            parts.append(item["pretrained_audio"])
        return parts

    def _load_predicted_relation_condition(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        record_index, start = self.predicted.windows[index]
        record = self.predicted.records[record_index]
        end = int(start) + int(self.predicted.sequence_length)
        relation, relation_mask, _, _ = self.predicted._slice_relation(
            record["_relation_path"], int(start), end
        )
        return torch.from_numpy(relation), torch.from_numpy(relation_mask), str(record["_relation_path"])

    def _rebuild_condition(self, item: Dict[str, Any]) -> None:
        condition_parts = list(self._audio_condition_parts(item)) + [item["relation"]]
        if self.include_relation_mask_in_condition:
            condition_parts.append(item["relation_mask"])
        item["condition"] = torch.cat(condition_parts, dim=-1).to(dtype=torch.float32)

    def _mix_relation_condition(self, item: Dict[str, Any], index: int) -> float:
        predicted_relation, predicted_mask, predicted_path = self._load_predicted_relation_condition(index)
        relation = item["relation"].clone()
        relation_mask = item["relation_mask"].clone()
        predicted_channel_total = 0.0
        channel_total = 0.0

        if self.relation_mix_mode == "whole_vector":
            # One Bernoulli decision controls the entire temporal window.  The
            # predicted probability is deliberately used here rather than the
            # per-group table, so this mode has an unambiguous exposure rate.
            if self.relation_mix_granularity == "window":
                use_prediction = self.predicted_probability >= 1.0 or bool(
                    torch.rand((), dtype=torch.float32).item() < self.predicted_probability
                )
                if use_prediction:
                    relation[...] = predicted_relation
                    relation_mask[...] = predicted_mask
                    predicted_channel_total = float(relation.shape[-1])
                else:
                    predicted_channel_total = 0.0
            else:
                use_prediction = torch.rand(
                    (relation.shape[0], 1), dtype=torch.float32
                ) < self.predicted_probability
                relation = torch.where(use_prediction, predicted_relation, relation)
                relation_mask = torch.where(use_prediction, predicted_mask, relation_mask)
                predicted_channel_total = float(use_prediction.float().mean().item()) * float(
                    relation.shape[-1]
                )

            # Keep training identical to deployment for explicitly dropped
            # channels, exactly as in the group-mixing branch below.
            for start, end in self.relation_dropped_slices:
                relation[:, int(start):int(end)] = 0.0
                relation_mask[:, int(start):int(end)] = 0.0
            self._rebuild_condition(item)
            item["conditioning_relation_path"] = predicted_path
            return predicted_channel_total / max(float(relation.shape[-1]), 1.0)

        if self.relation_mix_mode == "matched_joint":
            groups = list(self._scheduled_relation_group_probabilities())
            groups_by_name = {str(group["name"]): group for group in groups}
            consumed_names = set()

            def apply_group(group: Mapping[str, Any], use_prediction: bool) -> None:
                nonlocal predicted_channel_total, channel_total
                probability_slices = group["slices"]
                for start, end in probability_slices:
                    start = int(start)
                    end = int(end)
                    width = float(end - start)
                    channel_total += width
                    if use_prediction:
                        relation[:, start:end] = predicted_relation[:, start:end]
                        relation_mask[:, start:end] = predicted_mask[:, start:end]
                        predicted_channel_total += width

            # Couple each configured joint group with a single uniform draw.
            # For probabilities p_i, the joint event has probability
            # min(p_i); each remaining group-specific event retains exactly
            # p_i - min(p_i).  This is the maximal-positive-correlation
            # coupling and therefore minimizes inconsistent mixed states while
            # preserving every group's marginal exposure.
            for joint_names in self.relation_joint_groups:
                missing = [name for name in joint_names if name not in groups_by_name]
                if missing:
                    raise ValueError(
                        f"relation_joint_groups references unknown relation groups: {missing}"
                    )
                joint = [groups_by_name[name] for name in joint_names]
                probabilities = [float(group["probability"]) for group in joint]
                shared_probability = min(probabilities)
                draw = torch.rand((), dtype=torch.float32).item()
                if draw < shared_probability:
                    selected = [True] * len(joint)
                else:
                    selected = [
                        draw < probability for probability in probabilities
                    ]
                for group, use_prediction in zip(joint, selected):
                    apply_group(group, bool(use_prediction))
                    consumed_names.add(str(group["name"]))

            # All groups outside a joint block retain the historical
            # independent per-window sampling behaviour.
            for group in groups:
                name = str(group["name"])
                if name in consumed_names:
                    continue
                probability = float(group["probability"])
                use_prediction = probability >= 1.0 or bool(
                    torch.rand((), dtype=torch.float32).item() < probability
                )
                apply_group(group, use_prediction)

            for start, end in self.relation_dropped_slices:
                relation[:, int(start):int(end)] = 0.0
                relation_mask[:, int(start):int(end)] = 0.0
            self._rebuild_condition(item)
            item["conditioning_relation_path"] = predicted_path
            return predicted_channel_total / max(channel_total, 1.0)

        for group in self._scheduled_relation_group_probabilities():
            probability = float(group["probability"])
            for start, end in group["slices"]:
                start = int(start)
                end = int(end)
                width = float(end - start)
                channel_total += width
                if probability <= 0.0:
                    continue
                if self.relation_mix_granularity == "window":
                    use_prediction = probability >= 1.0 or bool(
                        torch.rand((), dtype=torch.float32).item() < probability
                    )
                    if use_prediction:
                        relation[:, start:end] = predicted_relation[:, start:end]
                        relation_mask[:, start:end] = predicted_mask[:, start:end]
                        predicted_channel_total += width
                else:
                    use_prediction = torch.rand(
                        (relation.shape[0], 1), dtype=torch.float32
                    ) < probability
                    relation[:, start:end] = torch.where(
                        use_prediction,
                        predicted_relation[:, start:end],
                        relation[:, start:end],
                    )
                    relation_mask[:, start:end] = torch.where(
                        use_prediction,
                        predicted_mask[:, start:end],
                        relation_mask[:, start:end],
                    )
                    predicted_channel_total += float(use_prediction.float().mean().item()) * width

        # Keep training identical to deployment. These channels are absent in
        # both GT and predicted branches, including the validity mask.
        for start, end in self.relation_dropped_slices:
            relation[:, start:end] = 0.0
            relation_mask[:, start:end] = 0.0

        item["relation"] = relation
        item["relation_mask"] = relation_mask
        self._rebuild_condition(item)
        item["conditioning_relation_path"] = predicted_path
        return predicted_channel_total / max(channel_total, 1.0)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # Pose, geometry arrays, contact_valid, and bow_valid remain GT.  The
        # default Bernoulli decision is made per temporal window to preserve
        # temporal coherence inside a sequence.  Optional group probabilities
        # let damaging relation channels be exposed more often during
        # robustness fine-tuning.
        item = self.ground_truth[index]
        predicted_fraction = self._mix_relation_condition(item, index)
        if predicted_fraction <= 0.0:
            item["conditioning_relation_path"] = str(item["relation_path"])
        item["relation_is_predicted"] = torch.tensor(float(predicted_fraction > 0.0), dtype=torch.float32)
        item["relation_predicted_channel_fraction"] = torch.tensor(
            float(predicted_fraction), dtype=torch.float32
        )
        return item


@dataclass
class _TrainingDatasetFactory:
    """Use frozen predictions for the requested training and validation datasets."""

    predicted_dataset_dir: Path
    predicted_probability: float
    relation_group_probabilities: Optional[Sequence[Mapping[str, Any]]]
    relation_mix_curriculum: Optional[Mapping[str, Any]]
    relation_mix_granularity: str
    relation_mix_mode: str
    relation_joint_groups: Optional[Sequence[Sequence[str]]]
    relation_dropped_slices: Optional[Sequence[Sequence[int]]]
    expected_data_path: Path
    expected_mode: Any
    validation_mode: Any

    def __call__(self, *args: Any, **kwargs: Any):
        data_path = kwargs.get("data_path", args[0] if args else None)
        mode = kwargs.get("mode", "train")
        is_training_dataset = (
            data_path is not None
            and _normalise_path(data_path) == self.expected_data_path
            and _plain(mode) == _plain(self.expected_mode)
        )
        is_validation_dataset = (
            data_path is not None
            and _normalise_path(data_path) == self.expected_data_path
            and self.validation_mode is not None
            and _plain(mode) == _plain(self.validation_mode)
        )
        if is_training_dataset or is_validation_dataset:
            if is_validation_dataset and not is_training_dataset:
                relation_group_probabilities = None
                predicted_probability = 1.0
                relation_mix_curriculum = None
            else:
                relation_group_probabilities = self.relation_group_probabilities
                predicted_probability = self.predicted_probability
                relation_mix_curriculum = self.relation_mix_curriculum
            return MixedRelationPoseDataset(
                SPDCanonicalPoseDataset,
                self.predicted_dataset_dir,
                predicted_probability,
                relation_group_probabilities=relation_group_probabilities,
                relation_mix_curriculum=relation_mix_curriculum,
                relation_mix_granularity=self.relation_mix_granularity,
                relation_mix_mode=self.relation_mix_mode,
                relation_joint_groups=self.relation_joint_groups,
                relation_dropped_slices=self.relation_dropped_slices,
                **kwargs,
            )
        return SPDCanonicalPoseDataset(*args, **kwargs)


def _make_weight_only_accelerator_class(base_accelerator_cls):
    """Create an Accelerator variant that skips optimizer/scheduler restore.

    The Stage-2 trainer calls ``accelerator.load_state`` after all
    objects have been prepared. Temporarily unregistering the optimizer and
    scheduler lets Accelerate restore both model state dicts while preserving
    the newly-created optimizer, scheduler, and gradient scaler.
    """

    class WeightOnlyResumeAccelerator(base_accelerator_cls):
        def load_state(self, input_dir=None, *args, **kwargs):
            saved_optimizers = self._optimizers
            saved_schedulers = self._schedulers
            saved_dataloaders = self._dataloaders
            saved_scaler = self.scaler
            self._optimizers = []
            self._schedulers = []
            self._dataloaders = []
            self.scaler = None
            try:
                self.print(
                    "Loading model weights only; optimizer, scheduler, dataloader, "
                    "and GradScaler state will not be restored."
                )
                return super().load_state(input_dir, *args, **kwargs)
            finally:
                self._optimizers = saved_optimizers
                self._schedulers = saved_schedulers
                self._dataloaders = saved_dataloaders
                self.scaler = saved_scaler

    WeightOnlyResumeAccelerator.__name__ = "WeightOnlyResumeAccelerator"
    return WeightOnlyResumeAccelerator


def main(
    stage2_config: str | Path | Mapping[str, Any],
    stage2_checkpoint_path: Optional[str],
    mapping_checkpoint_path: str,
    output_dir: str,
    predicted_relation_dataset_dir: str,
    fine_tune_steps: int = 10000,
    train_from_scratch: bool = False,
    train_from_scratch_steps: Optional[int] = None,
    predicted_relation_probability: float = 0.5,
    mapping_latent_dim: Optional[int] = None,
    mapping_batch_size: int = 8,
    mapping_num_workers: Optional[int] = None,
    mapping_sequence_length: Optional[int] = None,
    mapping_window_stride: Optional[int] = None,
    mapping_stitching_mode: str = "mean",
    mapping_stitching_min_weight: float = 0.1,
    mapping_categorical_mode: str = "one_hot",
    mapping_clamp_continuous: bool = True,
    predicted_relation_mask_mode: str = "predicted",
    predicted_validity_threshold: float = 0.5,
    predicted_relation_slices: Optional[Sequence[Sequence[int]]] = None,
    relation_dropped_slices: Optional[Sequence[Sequence[int]]] = None,
    hybrid_relation_mask_source: str = "matched",
    relation_group_probabilities: Optional[Sequence[Mapping[str, Any]]] = None,
    relation_mix_curriculum: Optional[Mapping[str, Any]] = None,
    relation_mix_granularity: str = "window",
    relation_mix_mode: str = "group",
    relation_joint_groups: Optional[Sequence[Sequence[str]]] = None,
    predicted_relation_source_mode: Optional[str] = None,
    predicted_relation_smoothing_window: int = 0,
    predicted_relation_smoothing_slices: Optional[Sequence[Sequence[int]]] = None,
    predicted_scalar_key: Optional[str] = None,
    predicted_scalar_valid_key: Optional[str] = None,
    mapping_device: str = "cuda" if torch.cuda.is_available() else "cpu",
    rebuild_predicted_relations: bool = False,
    prepare_only: bool = False,
    resume_optimizer_state: bool = False,
    resume_from_checkpoint: Optional[str] = None,
    unet_learning_rate: float = 1e-6,
    condition_encoder_learning_rate: float = 1e-5,
    stage2_overrides: Optional[Mapping[str, Any]] = None,
    configuration_file: Optional[str] = None,
) -> None:
    """Prepare predicted relations and run the shared Stage-2 trainer."""
    *_, invocation = inspect.getargvalues(inspect.currentframe())
    invocation = _plain(dict(invocation))
    stage2_config_values = _merge_stage2_config(stage2_config, stage2_overrides)
    train_data = stage2_config_values.get("train_data")
    if not isinstance(train_data, Mapping):
        raise KeyError("The stage-2 config must contain a train_data mapping.")

    train_from_scratch = bool(train_from_scratch)
    checkpoint_path: Optional[Path] = None
    if train_from_scratch:
        if stage2_checkpoint_path:
            raise ValueError(
                "train_from_scratch=true requires stage2_checkpoint_path: null."
            )
        if resume_optimizer_state and not resume_from_checkpoint:
            raise ValueError(
                "resume_optimizer_state must be false when training Stage 2 from scratch."
            )
        scratch_steps = (
            int(train_from_scratch_steps)
            if train_from_scratch_steps is not None
            else int(stage2_config_values.get("max_train_steps", 0))
        )
        if scratch_steps <= 0:
            raise ValueError(
                "train_from_scratch_steps must be positive, or stage2_config must define "
                "a positive max_train_steps."
            )
        checkpoint_step = 0
    else:
        if not stage2_checkpoint_path:
            raise ValueError(
                "stage2_checkpoint_path is required unless train_from_scratch=true."
            )
        checkpoint_path = _normalise_path(stage2_checkpoint_path)
        if not checkpoint_path.is_dir():
            raise FileNotFoundError(
                f"Stage-2 checkpoint directory does not exist: {checkpoint_path}"
            )
        checkpoint_step = _checkpoint_step(checkpoint_path)
        if int(fine_tune_steps) <= 0:
            raise ValueError("fine_tune_steps must be positive.")

    training_mode = str(stage2_config_values.get("training_mode", "diffusion"))
    full_trace_resume = bool(resume_from_checkpoint)
    if full_trace_resume:
        if (not train_from_scratch or not resume_optimizer_state or training_mode != 'diffusion'
                or (stage2_config_values.get('joint_geometry_options') or {}).get('bow_trace') is None):
            raise ValueError('resume_from_checkpoint requires the original joint_bow_trace run: train_from_scratch=true, stage2_checkpoint_path=null, resume_optimizer_state=true')
        checkpoint_path = resolve_resume_checkpoint(output_dir, resume_from_checkpoint).resolve()
        checkpoint_step = _checkpoint_step(checkpoint_path)
        if not 0 < checkpoint_step < scratch_steps:
            raise ValueError(f'Checkpoint step {checkpoint_step} must be below the original total {scratch_steps}')
    if training_mode == "audio_bow_events":
        if not train_from_scratch:
            raise ValueError("Audio event prediction is an independent fresh model; set train_from_scratch=true")
        stage2_config_values.update(output_dir=str(_normalise_path(output_dir)),max_train_steps=scratch_steps,
                                   audio_bow_event_context=invocation)
        print("Audio-event evidence experiment: cached audio -> event/direction proxy; Stage-1 and pose models stay unchanged.")
        return _train_stage2(**stage2_config_values)
    if training_mode == "local_bow_refiner":
        if train_from_scratch or checkpoint_step != 100000:
            raise ValueError("Local bow experiments require the original single checkpoint-100000")
        if resume_optimizer_state:
            raise ValueError("The 100k base is frozen; no parent optimizer is resumed")
        stage2_config_values.update(output_dir=str(_normalise_path(output_dir)),
                                   max_train_steps=checkpoint_step+int(fine_tune_steps),
                                   resume_from_checkpoint=str(checkpoint_path),
                                   local_bow_refiner_context=invocation)
        print("Local bow mode: fixed Stage-1 -> frozen plain100k train/val predictions -> new local head.")
        print("UNet/condition encoder stay frozen; final point is 100k + fine_tune_steps, without averaging.")
        return _train_stage2(**stage2_config_values)
    if training_mode != "diffusion":
        raise ValueError(f"Unknown stage2 training_mode: {training_mode}")

    trace_provenance=None
    if (stage2_config_values.get('joint_geometry_options') or {}).get('bow_trace') is not None:
        trace_provenance = (_validate_trace_resume(stage2_config_values, invocation, checkpoint_path)
                            if full_trace_resume else _prepare_trace_run(stage2_config_values, invocation))

    if full_trace_resume:
        predicted_dir = _normalise_path(predicted_relation_dataset_dir)
        print(f'Reusing original frozen Stage-1 cache: {predicted_dir}')
    else:
        predicted_dir, _ = prepare_predicted_relation_dataset(
            train_data=train_data,
            mapping_checkpoint_path=mapping_checkpoint_path,
            output_dir=predicted_relation_dataset_dir,
            mapping_latent_dim=mapping_latent_dim,
            mapping_batch_size=int(mapping_batch_size),
            mapping_num_workers=mapping_num_workers,
            mapping_sequence_length=mapping_sequence_length,
            mapping_window_stride=mapping_window_stride,
            mapping_stitching_mode=mapping_stitching_mode,
            mapping_stitching_min_weight=float(mapping_stitching_min_weight),
            mapping_categorical_mode=mapping_categorical_mode,
            mapping_clamp_continuous=bool(mapping_clamp_continuous),
            predicted_relation_mask_mode=predicted_relation_mask_mode,
            predicted_validity_threshold=float(predicted_validity_threshold),
            predicted_relation_slices=predicted_relation_slices,
            relation_dropped_slices=relation_dropped_slices,
            hybrid_relation_mask_source=hybrid_relation_mask_source,
            device=mapping_device,
            rebuild=bool(rebuild_predicted_relations),
            source_mode=predicted_relation_source_mode,
            predicted_relation_smoothing_window=int(predicted_relation_smoothing_window),
            predicted_relation_smoothing_slices=predicted_relation_smoothing_slices,
            predicted_scalar_key=predicted_scalar_key,
            predicted_scalar_valid_key=predicted_scalar_valid_key,
        )
    if prepare_only:
        print(
            "prepare_only=true: predicted relations are ready; "
            "stage-2 training was not started."
        )
        return

    probability = float(predicted_relation_probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("predicted_relation_probability must be in [0, 1].")
    normalised_curriculum = _normalise_relation_mix_curriculum(relation_mix_curriculum)
    unet_lr = float(unet_learning_rate)
    condition_lr = float(condition_encoder_learning_rate)
    if unet_lr <= 0.0 or condition_lr <= 0.0:
        raise ValueError("Both Stage-2 learning rates must be positive.")
    if not bool(resume_optimizer_state) and bool(stage2_config_values.get("scale_lr", False)):
        raise ValueError(
            "scale_lr must be false when using explicit UNet/condition-encoder learning rates."
        )

    output_path = _normalise_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stage2_config_values["output_dir"] = str(output_path)
    stage2_config_values["resume_from_checkpoint"] = (
        str(checkpoint_path) if full_trace_resume or not train_from_scratch else None
    )
    stage2_config_values["max_train_steps"] = (
        scratch_steps
        if train_from_scratch
        else checkpoint_step + int(fine_tune_steps)
    )
    if not bool(resume_optimizer_state) or full_trace_resume:
        # This value remains useful in the stage-2 config written by the
        # unchanged trainer. The exact two group LRs are recorded separately
        # in robust_finetune_metadata.json.
        stage2_config_values["learning_rate"] = condition_lr
        stage2_config_values["scale_lr"] = False

    # The current repository's conservative SPD config disables validation.
    # If another config enables it, it remains the original GT diagnostic;
    # final model selection should use evaluate.py.
    run_metadata = {
        "description": (
            "Stage-2 robustness training with frozen stage-1 predictions."
            if train_from_scratch
            else "Stage-2 robustness fine-tuning with frozen stage-1 predictions."
        ),
        "stage2_config": (
            invocation["stage2_config"] if isinstance(stage2_config, Mapping)
            else str(_normalise_path(stage2_config))
        ),
        "stage2_checkpoint_path": None if train_from_scratch else str(checkpoint_path),
        "stage2_initialization": "random" if train_from_scratch else "checkpoint_weights",
        "mapping_checkpoint_path": str(_normalise_path(mapping_checkpoint_path)),
        "predicted_relation_dataset_dir": str(predicted_dir),
        "checkpoint_step": checkpoint_step,
        "fine_tune_steps": None if train_from_scratch else int(fine_tune_steps),
        "train_from_scratch_steps": (
            int(stage2_config_values["max_train_steps"]) if train_from_scratch else None
        ),
        "final_max_train_steps": int(stage2_config_values["max_train_steps"]),
        "predicted_relation_probability": probability,
        "mapping_categorical_mode": str(mapping_categorical_mode),
        "mapping_clamp_continuous": bool(mapping_clamp_continuous),
        "mapping_sequence_length": (
            int(mapping_sequence_length) if mapping_sequence_length is not None
            else int(train_data["sequence_length"])
        ),
        "mapping_window_stride": (
            int(mapping_window_stride) if mapping_window_stride is not None
            else int(train_data.get("window_stride", train_data["sequence_length"]))
        ),
        "mapping_stitching_mode": str(mapping_stitching_mode),
        "mapping_stitching_min_weight": float(mapping_stitching_min_weight),
        "predicted_relation_mask_mode": str(predicted_relation_mask_mode),
        "predicted_validity_threshold": float(predicted_validity_threshold),
        "predicted_relation_slices": _plain(predicted_relation_slices),
        "relation_dropped_slices": _plain(relation_dropped_slices),
        "hybrid_relation_mask_source": str(hybrid_relation_mask_source),
        "relation_group_probabilities": _plain(relation_group_probabilities),
        "relation_mix_curriculum": _plain(normalised_curriculum),
        "relation_mix_granularity": str(relation_mix_granularity),
        "relation_mix_mode": str(relation_mix_mode),
        "relation_joint_groups": _plain(relation_joint_groups),
        "predicted_relation_source_mode": predicted_relation_source_mode,
        "predicted_relation_smoothing_window": int(predicted_relation_smoothing_window),
        "predicted_relation_smoothing_slices": _plain(predicted_relation_smoothing_slices),
        "mapping_frozen": True,
        "geometry_supervision": "ground_truth",
        "resume_optimizer_state": bool(resume_optimizer_state),
        "optimizer": "restored AdamW" if resume_optimizer_state else "fresh AdamW",
        "unet_learning_rate": unet_lr,
        "condition_encoder_learning_rate": condition_lr,
        "stage2_overrides": _plain(stage2_overrides or {}),
    }
    if trace_provenance is not None:run_metadata['joint_bow_trace_provenance']=trace_provenance
    if full_trace_resume:
        audit = dict(input_config=invocation, checkpoint=str(checkpoint_path), checkpoint_step=checkpoint_step,
                     final_max_train_steps=stage2_config_values['max_train_steps'],
                     scheduler_state_source=('saved' if any((checkpoint_path / name).is_file()
                         for name in ('scheduler.bin', 'lr_scheduler.pt')) else 'reconstructed_constant_no_warmup'),
                     entry_point_sha256={'scripts/train_stage2.py': _file_sha256(Path(__file__))})
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        _write_json(output_path / 'resume_history' / f'{timestamp}.json', audit)
    else:
        _write_json(output_path / "robust_finetune_metadata.json", run_metadata)
        OmegaConf.save(OmegaConf.create(stage2_config_values), str(output_path / "resolved_stage2_config.yaml"))

    validation_data = stage2_config_values.get("validation_data")
    dataset_factory = _TrainingDatasetFactory(
        predicted_dataset_dir=predicted_dir,
        predicted_probability=probability,
        relation_group_probabilities=_plain(relation_group_probabilities),
        relation_mix_curriculum=_plain(normalised_curriculum),
        relation_mix_granularity=str(relation_mix_granularity),
        relation_mix_mode=str(relation_mix_mode).lower(),
        relation_joint_groups=_plain(relation_joint_groups),
        relation_dropped_slices=_plain(relation_dropped_slices),
        expected_data_path=_normalise_path(train_data["data_path"]),
        expected_mode=train_data.get("mode", "train"),
        validation_mode=validation_data.get("mode") if isinstance(validation_data, Mapping) else None,
    )
    if full_trace_resume:
        print(f"Full-state joint bow trace resume: {checkpoint_step} -> {scratch_steps}; restoring all three models, AdamW, scheduler and RNG")
    elif train_from_scratch:
        print(
            f"Starting Stage-2 training from scratch: 0 -> {stage2_config_values['max_train_steps']}; "
            f"predicted relation probability={probability:.2f}"
        )
    else:
        print(
            f"Fine-tuning Stage 2: {checkpoint_step} -> {stage2_config_values['max_train_steps']}; "
            f"predicted relation probability={probability:.2f}; "
            f"resume optimizer state={bool(resume_optimizer_state)}"
        )
    optimizer_learning_rates = (unet_lr, condition_lr) if not resume_optimizer_state or full_trace_resume else None
    return _train_stage2(
        **stage2_config_values,
        dataset_factory=dataset_factory,
        optimizer_learning_rates=optimizer_learning_rates,
        weights_only_resume=not resume_optimizer_state and not train_from_scratch and not full_trace_resume,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML for Stage-2 training with frozen Stage-1 predictions.",
    )
    arguments = parser.parse_args()
    config = OmegaConf.load(arguments.config)
    config["configuration_file"] = str(Path(arguments.config).resolve())
    main(**config)
