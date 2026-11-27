"""Train the temporal SPD audio-to-relation model for Stage 1."""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio2pose.data.data import AudioPoseDataset, SPDCanonicalRelationsDataset


logger = get_logger(__name__, log_level="INFO")


RELATION_FEATURE_NAMES = [
    "contact_string_4", "contact_string_3", "contact_string_2", "contact_string_1",
    "contact_u", "contact_distance",
    "bow_string_4", "bow_string_3", "bow_string_2", "bow_string_1",
    "bow_u", "bow_t", "bow_distance", "bow_parallel_abs_cos",
    "bow_dir_x", "bow_dir_y", "bow_dir_z",
    "bow_mid_x", "bow_mid_y", "bow_mid_z",
    "contact_bow_same_string", "finger_tip_to_contact_distance", "finger_tip_to_string_distance",
]


# =========================
# Model
# =========================

class SinusoidalPositionalEncoding(nn.Module):
    """Add deterministic frame-position information before temporal attention."""

    def __init__(self, dim, max_len=8192, dropout=0.0):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-np.log(10000.0) / dim)
        )
        pe = torch.zeros(max_len, dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        if dim > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        length = x.shape[1]
        if length > self.pe.shape[1]:
            raise ValueError(
                f"Sequence length {length} exceeds positional encoding max_len={self.pe.shape[1]}."
            )
        pos = self.pe[:, :length].to(device=x.device, dtype=x.dtype)
        return self.dropout(x + pos)


class MotionEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=256, dropout=0.1, use_positional_encoding=False):
        super().__init__()
        self.use_positional_encoding = bool(use_positional_encoding)

        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.position = (
            SinusoidalPositionalEncoding(latent_dim, dropout=dropout)
            if self.use_positional_encoding
            else nn.Identity()
        )

        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )

    def forward(self, x):
        B, T, J, C = x.shape
        x = x.view(B, T, J * C)

        x = self.input_proj(x)
        x = self.position(x)
        x = self.temporal(x)

        return x  # (B,T,D)


class AudioEncoder(nn.Module):
    def __init__(
        self,
        mel_dim,
        latent_dim=256,
        dropout=0.1,
        use_positional_encoding=False,
        pretrained_audio_dim=0,
        pretrained_audio_hidden_dim=128,
        num_instruments=0,
        instrument_embedding_dim=16,
        onset_dim=1,
    ):
        super().__init__()
        self.use_positional_encoding = bool(use_positional_encoding)
        self.pretrained_audio_dim = int(pretrained_audio_dim or 0)
        self.pretrained_audio_hidden_dim = (
            int(pretrained_audio_hidden_dim) if self.pretrained_audio_dim > 0 else 0
        )
        self.num_instruments = int(num_instruments or 0)
        self.instrument_embedding_dim = (
            int(instrument_embedding_dim) if self.num_instruments > 0 else 0
        )

        self.mel_encoder = nn.Linear(mel_dim, 128)
        self.onset_encoder = nn.Linear(int(onset_dim), 32)
        self.pitch_encoder = nn.Linear(1, 32)
        self.pretrained_audio_encoder = None
        if self.pretrained_audio_dim > 0:
            self.pretrained_audio_encoder = nn.Sequential(
                nn.LayerNorm(self.pretrained_audio_dim),
                nn.Linear(self.pretrained_audio_dim, self.pretrained_audio_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        self.instrument_embedding = None
        if self.num_instruments > 0:
            # The final index is reserved for unknown instruments.
            self.instrument_embedding = nn.Embedding(
                self.num_instruments + 1,
                self.instrument_embedding_dim,
            )

        fusion_dim = (
            192
            + self.pretrained_audio_hidden_dim
            + self.instrument_embedding_dim
        )
        self.fusion = nn.Linear(fusion_dim, latent_dim)
        self.position = (
            SinusoidalPositionalEncoding(latent_dim, dropout=dropout)
            if self.use_positional_encoding
            else nn.Identity()
        )

        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )

    def forward(self, mel, onset, pitch, pretrained_audio=None, instrument_id=None):

        mel_f = self.mel_encoder(mel)
        onset_f = self.onset_encoder(onset)
        pitch_f = self.pitch_encoder(pitch)

        parts = [mel_f, onset_f, pitch_f]
        if self.pretrained_audio_dim > 0:
            if pretrained_audio is None:
                raise ValueError(
                    "This AudioEncoder was created with pretrained_audio_dim > 0, "
                    "but the batch does not contain 'pretrained_audio'."
                )
            if pretrained_audio.shape[-1] != self.pretrained_audio_dim:
                raise ValueError(
                    f"Expected pretrained_audio dim {self.pretrained_audio_dim}, "
                    f"got {pretrained_audio.shape[-1]}."
                )
            parts.append(self.pretrained_audio_encoder(pretrained_audio))
        if self.num_instruments > 0:
            if instrument_id is None:
                raise ValueError(
                    "This AudioEncoder uses instrument conditioning, but the batch does not "
                    "contain 'instrument_id'."
                )
            instrument_id = instrument_id.to(device=mel.device, dtype=torch.long).reshape(-1)
            if instrument_id.shape[0] != mel.shape[0]:
                raise ValueError(
                    f"Expected {mel.shape[0]} instrument ids, got {instrument_id.shape[0]}."
                )
            unknown_id = torch.full_like(instrument_id, self.num_instruments)
            instrument_id = torch.where(
                (instrument_id >= 0) & (instrument_id < self.num_instruments),
                instrument_id,
                unknown_id,
            )
            instrument_feature = self.instrument_embedding(instrument_id)
            instrument_feature = instrument_feature[:, None, :].expand(-1, mel.shape[1], -1)
            parts.append(instrument_feature)

        x = torch.cat(parts, dim=-1)
        x = self.fusion(x)
        x = self.position(x)

        x = self.temporal(x)

        return x  # (B,T,D)


class RelationPredictor(nn.Module):
    def __init__(
        self,
        mel_dim,
        relation_dim,
        latent_dim=256,
        dropout=0.1,
        predict_validity=True,
        use_positional_encoding=False,
        pretrained_audio_dim=0,
        pretrained_audio_hidden_dim=128,
        num_instruments=0,
        instrument_embedding_dim=16,
        onset_dim=1,
    ):
        super().__init__()
        self.relation_dim = int(relation_dim)
        self.predict_validity = bool(predict_validity)
        self.use_positional_encoding = bool(use_positional_encoding)
        self.pretrained_audio_dim = int(pretrained_audio_dim or 0)
        self.pretrained_audio_hidden_dim = (
            int(pretrained_audio_hidden_dim) if self.pretrained_audio_dim > 0 else 0
        )
        self.num_instruments = int(num_instruments or 0)
        self.instrument_embedding_dim = (
            int(instrument_embedding_dim) if self.num_instruments > 0 else 0
        )
        self.audio_encoder = AudioEncoder(
            mel_dim,
            latent_dim,
            dropout=dropout,
            use_positional_encoding=self.use_positional_encoding,
            pretrained_audio_dim=self.pretrained_audio_dim,
            pretrained_audio_hidden_dim=self.pretrained_audio_hidden_dim,
            num_instruments=self.num_instruments,
            instrument_embedding_dim=self.instrument_embedding_dim,
            onset_dim=onset_dim,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, relation_dim),
        )
        self.validity_head = None
        if self.predict_validity:
            self.validity_head = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(latent_dim // 2, relation_dim),
            )

    def _encode(self, mel, onset, pitch, pretrained_audio=None, instrument_id=None):
        return self.audio_encoder(
            mel,
            onset,
            pitch,
            pretrained_audio=pretrained_audio,
            instrument_id=instrument_id,
        )

    def forward(self, mel, onset, pitch, pretrained_audio=None, instrument_id=None):
        x = self._encode(
            mel,
            onset,
            pitch,
            pretrained_audio=pretrained_audio,
            instrument_id=instrument_id,
        )
        return self.head(x)

    def forward_with_validity(
        self,
        mel,
        onset,
        pitch,
        pretrained_audio=None,
        instrument_id=None,
    ):
        """Return relation predictions and per-feature validity logits."""
        x = self._encode(
            mel,
            onset,
            pitch,
            pretrained_audio=pretrained_audio,
            instrument_id=instrument_id,
        )
        relation = self.head(x)
        validity = self.validity_head(x) if self.validity_head is not None else None
        return relation, validity


class TemporalConvBlock(nn.Module):
    """Residual dilated temporal convolution used by the structured model."""

    def __init__(self, dim, kernel_size=5, dilation=1, dropout=0.1):
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("structured_tcn_kernel_size must be an odd integer >= 3.")
        padding = int(dilation) * (kernel_size - 1) // 2
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=int(dilation),
            groups=dim,
        )
        self.pointwise = nn.Sequential(
            nn.Conv1d(dim, 2 * dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(2 * dim, dim, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        residual = x
        x = self.norm(x).transpose(1, 2)
        x = self.depthwise(x)
        x = self.pointwise(x).transpose(1, 2)
        return residual + x


class StructuredTCNBiGRURelationPredictor(nn.Module):
    """Data-efficient temporal predictor with a structured bow-state decoder."""

    def __init__(
        self,
        mel_dim,
        relation_dim,
        latent_dim=256,
        dropout=0.1,
        predict_validity=True,
        use_positional_encoding=False,
        pretrained_audio_dim=0,
        pretrained_audio_hidden_dim=128,
        num_instruments=0,
        instrument_embedding_dim=16,
        structured_tcn_layers=4,
        structured_tcn_kernel_size=5,
        structured_gru_layers=2,
        structured_bow_step_scale=0.1,
        bow_reversal_sharpen_tau=0.0,
        predict_bow_root_bridge_distance=False,
        bow_root_bridge_head_mode="softplus",
        onset_dim=1,
        categorical_groups=None,
    ):
        super().__init__()
        groups = normalize_categorical_groups(categorical_groups, relation_dim=int(relation_dim)) if categorical_groups else []
        if groups:
            cursor = 0
            for group in groups:
                if group['start'] != cursor:
                    raise ValueError('Categorical groups must cover output channels in order without gaps.')
                cursor = group['end']
            if cursor != int(relation_dim) or len({g['name'] for g in groups}) != len(groups):
                raise ValueError('Categorical groups must uniquely cover all relation channels.')
            if predict_bow_root_bridge_distance or predict_validity:
                raise ValueError('Categorical task heads do not use SPD scalar or validity heads.')
        if not groups and int(relation_dim) != len(RELATION_FEATURE_NAMES):
            raise ValueError(
                "StructuredTCNBiGRURelationPredictor currently requires the "
                f"{len(RELATION_FEATURE_NAMES)}-dimensional SPD relation vector."
            )
        if int(latent_dim) % 2 != 0:
            raise ValueError("latent_dim must be even for the bidirectional GRU.")
        if int(structured_tcn_layers) <= 0 or int(structured_gru_layers) <= 0:
            raise ValueError("structured TCN and GRU layer counts must be positive.")
        if float(structured_bow_step_scale) <= 0.0:
            raise ValueError("structured_bow_step_scale must be positive.")
        if float(bow_reversal_sharpen_tau) < 0.0:
            raise ValueError("bow_reversal_sharpen_tau must be non-negative.")

        self.relation_dim = int(relation_dim)
        self.categorical_groups = groups
        self.predict_validity = bool(predict_validity)
        self.use_positional_encoding = bool(use_positional_encoding)
        self.pretrained_audio_dim = int(pretrained_audio_dim or 0)
        self.pretrained_audio_hidden_dim = (
            int(pretrained_audio_hidden_dim) if self.pretrained_audio_dim > 0 else 0
        )
        self.num_instruments = int(num_instruments or 0)
        self.instrument_embedding_dim = (
            int(instrument_embedding_dim) if self.num_instruments > 0 else 0
        )
        self.structured_tcn_layers = int(structured_tcn_layers)
        self.structured_tcn_kernel_size = int(structured_tcn_kernel_size)
        self.structured_gru_layers = int(structured_gru_layers)
        self.structured_bow_step_scale = float(structured_bow_step_scale)
        self.bow_reversal_sharpen_tau = float(bow_reversal_sharpen_tau)
        self.predict_bow_root_bridge_distance = bool(
            predict_bow_root_bridge_distance
        )
        self.bow_root_bridge_head_mode = normalize_bow_root_bridge_head_mode(
            bow_root_bridge_head_mode
        )

        self.mel_encoder = nn.Linear(mel_dim, 128)
        self.onset_encoder = nn.Linear(int(onset_dim), 32)
        self.pitch_encoder = nn.Linear(1, 32)
        self.pretrained_audio_encoder = None
        if self.pretrained_audio_dim > 0:
            self.pretrained_audio_encoder = nn.Sequential(
                nn.LayerNorm(self.pretrained_audio_dim),
                nn.Linear(self.pretrained_audio_dim, self.pretrained_audio_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        self.instrument_embedding = None
        if self.num_instruments > 0:
            self.instrument_embedding = nn.Embedding(
                self.num_instruments + 1,
                self.instrument_embedding_dim,
            )

        fusion_dim = 192 + self.pretrained_audio_hidden_dim + self.instrument_embedding_dim
        self.fusion = nn.Linear(fusion_dim, latent_dim)
        self.position = (
            SinusoidalPositionalEncoding(latent_dim, dropout=dropout)
            if self.use_positional_encoding
            else nn.Identity()
        )
        self.temporal_convs = nn.ModuleList(
            TemporalConvBlock(
                latent_dim,
                kernel_size=self.structured_tcn_kernel_size,
                dilation=2 ** layer_index,
                dropout=dropout,
            )
            for layer_index in range(self.structured_tcn_layers)
        )
        self.temporal_gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=latent_dim // 2,
            num_layers=self.structured_gru_layers,
            dropout=dropout if self.structured_gru_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(latent_dim)

        def head(output_dim):
            return nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(latent_dim, output_dim),
            )

        if self.categorical_groups:
            # Same acoustic/TCN/BiGRU backbone; only the supervised task heads change.
            self.categorical_heads = nn.ModuleDict({
                group['name']: head(group['end'] - group['start']) for group in self.categorical_groups})
            self.validity_head = None
            return
        self.contact_string_head = head(4)
        self.contact_continuous_head = head(2)
        self.bow_string_head = head(4)
        self.bow_u_head = head(1)
        self.tail_continuous_head = head(11)
        self.bow_direction_head = head(3)  # down, static, up
        self.bow_step_head = head(1)
        self.initial_bow_t_head = head(1)
        self.bow_reversal_head = head(1)
        self.bow_root_bridge_head = (
            head(1) if self.predict_bow_root_bridge_distance else None
        )
        self.validity_head = head(self.relation_dim) if self.predict_validity else None

    def _encode(self, mel, onset, pitch, pretrained_audio=None, instrument_id=None):
        parts = [
            self.mel_encoder(mel),
            self.onset_encoder(onset),
            self.pitch_encoder(pitch),
        ]
        if self.pretrained_audio_dim > 0:
            if pretrained_audio is None:
                raise ValueError(
                    "The structured mapping model expects pretrained_audio, but it was not provided."
                )
            if pretrained_audio.shape[-1] != self.pretrained_audio_dim:
                raise ValueError(
                    f"Expected pretrained_audio dim {self.pretrained_audio_dim}, "
                    f"got {pretrained_audio.shape[-1]}."
                )
            parts.append(self.pretrained_audio_encoder(pretrained_audio))
        if self.num_instruments > 0:
            if instrument_id is None:
                raise ValueError(
                    "The structured mapping model uses instrument conditioning, but "
                    "instrument_id was not provided."
                )
            instrument_id = instrument_id.to(device=mel.device, dtype=torch.long).reshape(-1)
            unknown_id = torch.full_like(instrument_id, self.num_instruments)
            instrument_id = torch.where(
                (instrument_id >= 0) & (instrument_id < self.num_instruments),
                instrument_id,
                unknown_id,
            )
            instrument_feature = self.instrument_embedding(instrument_id)
            parts.append(instrument_feature[:, None, :].expand(-1, mel.shape[1], -1))

        x = self.position(self.fusion(torch.cat(parts, dim=-1)))
        for block in self.temporal_convs:
            x = block(x)
        x, _ = self.temporal_gru(x)
        return self.output_norm(x)

    def _decode(self, x):
        if self.categorical_groups:
            return torch.cat([self.categorical_heads[g['name']](x) for g in self.categorical_groups], -1), {}
        direction_logits = self.bow_direction_head(x)
        direction_probability = torch.softmax(direction_logits, dim=-1)
        expected_direction = direction_probability[..., 2] - direction_probability[..., 0]
        reversal_logits = self.bow_reversal_head(x).squeeze(-1)
        if self.bow_reversal_sharpen_tau > 0.0:
            # At a turnaround the soft direction head is uncertain, so its
            # expected_direction averages toward 0 and flattens the cumulative
            # bow_t trajectory exactly where it should reverse. The reversal
            # head already detects these frames; use its probability to sharpen
            # the direction commitment (push expected_direction toward +/-1) at
            # reversals only, leaving steady-state frames untouched. This is
            # the lever that lets bow_t actually turn instead of plateauing.
            reversal_probability = torch.sigmoid(reversal_logits)
            sharpened = torch.tanh(expected_direction / self.bow_reversal_sharpen_tau)
            expected_direction = (
                1.0 - reversal_probability
            ) * expected_direction + reversal_probability * sharpened
        bow_step = torch.sigmoid(self.bow_step_head(x).squeeze(-1)) * self.structured_bow_step_scale
        bow_delta = expected_direction * bow_step
        initial_logit = self.initial_bow_t_head(x[:, :1]).squeeze(-1)
        accumulated_delta = torch.cat(
            [torch.zeros_like(bow_delta[:, :1]), torch.cumsum(bow_delta[:, 1:], dim=1)],
            dim=1,
        )
        bow_t = torch.sigmoid(initial_logit + accumulated_delta).unsqueeze(-1)

        relation = torch.cat(
            [
                self.contact_string_head(x),       # 0:4
                self.contact_continuous_head(x),   # 4:6
                self.bow_string_head(x),           # 6:10
                self.bow_u_head(x),                # 10:11
                bow_t,                             # 11:12
                self.tail_continuous_head(x),      # 12:23
            ],
            dim=-1,
        )
        auxiliary = {
            "bow_direction_logits": direction_logits,
            "bow_reversal_logits": reversal_logits,
            "bow_step": bow_step,
        }
        if self.bow_root_bridge_head is not None:
            raw_scalar = self.bow_root_bridge_head(x).squeeze(-1)
            if self.bow_root_bridge_head_mode == "signed_offset":
                # BCS-aligned centered profile: distance - half_bow.  Inference
                # code reconstructs the absolute distance with the GT half bow
                # so the Stage-2 condition channel keeps distance semantics.
                auxiliary["bow_root_bridge_signed_offset"] = raw_scalar
            else:
                auxiliary["bow_root_bridge_distance"] = F.softplus(raw_scalar)
        return relation, auxiliary

    def forward(self, mel, onset, pitch, pretrained_audio=None, instrument_id=None):
        x = self._encode(mel, onset, pitch, pretrained_audio, instrument_id)
        relation, _ = self._decode(x)
        return relation

    def forward_with_validity(
        self, mel, onset, pitch, pretrained_audio=None, instrument_id=None
    ):
        relation, validity, _ = self.forward_with_auxiliary(
            mel, onset, pitch, pretrained_audio, instrument_id
        )
        return relation, validity

    def forward_with_auxiliary(
        self, mel, onset, pitch, pretrained_audio=None, instrument_id=None
    ):
        x = self._encode(mel, onset, pitch, pretrained_audio, instrument_id)
        relation, auxiliary = self._decode(x)
        validity = self.validity_head(x) if self.validity_head is not None else None
        return relation, validity, auxiliary


def normalize_mapping_model_type(model_type: Optional[str]) -> str:
    value = str(model_type or "transformer").lower()
    aliases = {
        "legacy": "transformer",
        "legacy_transformer": "transformer",
        "relation_predictor": "transformer",
        "tcn_bigru": "structured_tcn_bigru",
        "structured": "structured_tcn_bigru",
    }
    return aliases.get(value, value)


def normalize_bow_root_bridge_head_mode(mode):
    """Resolve the scalar head output transform.

    ``softplus`` keeps the historical behaviour: the head predicts the
    absolute frog-to-bridge distance (always positive).  ``signed_offset``
    makes the head predict the BCS-aligned centered profile
    ``distance - half_bow`` directly (raw linear output, can be negative);
    inference code reconstructs the absolute distance by adding the GT
    half bow so downstream consumers keep distance semantics.
    """
    value = str(mode or "softplus").lower()
    aliases = {"linear": "signed_offset", "signed": "signed_offset", "offset": "signed_offset"}
    value = aliases.get(value, value)
    if value not in ("softplus", "signed_offset"):
        raise ValueError(
            "bow_root_bridge_head_mode must be 'softplus' or 'signed_offset', got "
            f"'{mode}'."
        )
    return value


def create_relation_predictor(
    *,
    model_type: str = "transformer",
    mel_dim: int,
    relation_dim: int,
    latent_dim: int = 256,
    dropout: float = 0.1,
    predict_validity: bool = True,
    use_positional_encoding: bool = False,
    pretrained_audio_dim: int = 0,
    pretrained_audio_hidden_dim: int = 128,
    num_instruments: int = 0,
    instrument_embedding_dim: int = 16,
    structured_tcn_layers: int = 4,
    structured_tcn_kernel_size: int = 5,
    structured_gru_layers: int = 2,
    structured_bow_step_scale: float = 0.1,
    bow_reversal_sharpen_tau: float = 0.0,
    predict_bow_root_bridge_distance: bool = False,
    bow_root_bridge_head_mode: str = "softplus",
    onset_dim: int = 1,
    categorical_groups=None,
):
    """Build a mapping model while preserving the legacy default exactly."""
    resolved_type = normalize_mapping_model_type(model_type)
    common = dict(
        mel_dim=mel_dim,
        relation_dim=relation_dim,
        latent_dim=latent_dim,
        dropout=dropout,
        predict_validity=predict_validity,
        use_positional_encoding=use_positional_encoding,
        pretrained_audio_dim=pretrained_audio_dim,
        pretrained_audio_hidden_dim=pretrained_audio_hidden_dim,
        num_instruments=num_instruments,
        instrument_embedding_dim=instrument_embedding_dim,
        onset_dim=onset_dim,
    )
    if resolved_type == "transformer":
        return RelationPredictor(**common)
    if resolved_type == "structured_tcn_bigru":
        return StructuredTCNBiGRURelationPredictor(
            **common,
            structured_tcn_layers=structured_tcn_layers,
            structured_tcn_kernel_size=structured_tcn_kernel_size,
            structured_gru_layers=structured_gru_layers,
            structured_bow_step_scale=structured_bow_step_scale,
            bow_reversal_sharpen_tau=bow_reversal_sharpen_tau,
            predict_bow_root_bridge_distance=predict_bow_root_bridge_distance,
            bow_root_bridge_head_mode=bow_root_bridge_head_mode,
            categorical_groups=categorical_groups,
        )
    raise ValueError(
        "mapping_model_type must be 'transformer' or 'structured_tcn_bigru', "
        f"got '{model_type}'."
    )


def create_relation_predictor_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    mel_dim: int,
    relation_dim: int,
    latent_dim: Optional[int] = None,
    onset_dim: Optional[int] = None,
):
    """Recreate either legacy or structured Stage-1 models from metadata."""
    if onset_dim is None:
        onset_dim = int(checkpoint.get("onset_dim", 1))
    return create_relation_predictor(
        model_type=str(checkpoint.get("mapping_model_type", "transformer")),
        mel_dim=int(mel_dim),
        relation_dim=int(relation_dim),
        latent_dim=int(latent_dim if latent_dim is not None else checkpoint.get("latent_dim", 256)),
        dropout=float(checkpoint.get("dropout", 0.1)),
        predict_validity=bool(checkpoint.get("predict_validity", False)),
        use_positional_encoding=bool(checkpoint.get("use_positional_encoding", False)),
        pretrained_audio_dim=int(checkpoint.get("pretrained_audio_dim", 0)),
        pretrained_audio_hidden_dim=int(checkpoint.get("pretrained_audio_hidden_dim", 128)),
        num_instruments=int(checkpoint.get("num_instruments", 0)),
        instrument_embedding_dim=int(checkpoint.get("instrument_embedding_dim", 16)),
        structured_tcn_layers=int(checkpoint.get("structured_tcn_layers", 4)),
        structured_tcn_kernel_size=int(checkpoint.get("structured_tcn_kernel_size", 5)),
        structured_gru_layers=int(checkpoint.get("structured_gru_layers", 2)),
        structured_bow_step_scale=float(checkpoint.get("structured_bow_step_scale", 0.1)),
        bow_reversal_sharpen_tau=float(checkpoint.get("bow_reversal_sharpen_tau", 0.0)),
        predict_bow_root_bridge_distance=bool(
            checkpoint.get("predict_bow_root_bridge_distance", False)
        ),
        bow_root_bridge_head_mode=str(
            checkpoint.get("bow_root_bridge_head_mode", "softplus")
        ),
        onset_dim=int(onset_dim),
        categorical_groups=checkpoint.get("categorical_groups") or None,
    )


def infer_pretrained_audio_dim(sample):
    value = sample.get("pretrained_audio") if isinstance(sample, dict) else None
    if value is None:
        return 0
    return int(value.shape[-1])


def infer_onset_dim(sample):
    """Onset channel count from a dataset sample (1 = binary impulse only;
    1 + 4 = impulse + bow-dynamics channels when use_bow_dynamics is on)."""
    value = sample.get("onset") if isinstance(sample, dict) else None
    if value is None:
        return 1
    return max(1, int(value.shape[-1]))


def expand_onset_encoder_state(
    state: Mapping[str, torch.Tensor], onset_dim: int
) -> Dict[str, torch.Tensor]:
    """Zero-pad ``onset_encoder.weight`` columns for channel expansion.

    A checkpoint trained with ``ckpt_dim`` onset channels loads into a model
    built with ``onset_dim > ckpt_dim`` channels: the old columns are copied
    and the new columns are zero, so the model's initial predictions are
    exactly the checkpoint's - a clean single-variable experiment setup.
    """
    key = "onset_encoder.weight"
    if key not in state:
        return state
    old_weight = state[key]
    old_dim = int(old_weight.shape[1])
    if onset_dim <= old_dim:
        return state
    new_weight = old_weight.new_zeros(old_weight.shape[0], onset_dim)
    new_weight[:, :old_dim] = old_weight
    new_state = dict(state)
    new_state[key] = new_weight
    return new_state


def optional_pretrained_audio(batch, device=None, non_blocking=False):
    value = batch.get("pretrained_audio") if isinstance(batch, dict) else None
    if value is None:
        return None
    if device is not None:
        return value.to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    return value.float()


def optional_instrument_id(batch, device=None, non_blocking=False):
    value = batch.get("instrument_id") if isinstance(batch, dict) else None
    if value is None:
        return None
    if device is not None:
        return value.to(device=device, dtype=torch.long, non_blocking=non_blocking)
    return value.long()


# =========================
# Loss
# =========================

def alignment_loss(audio_latent, motion_latent):
    return F.mse_loss(audio_latent, motion_latent)

def info_nce_loss(audio_latent, motion_latent, temperature=0.07):
    """
    audio_latent: (B,T,D)
    motion_latent: (B,T,D)
    """

    # temporal pooling
    audio = audio_latent.mean(dim=1)  # (B,D)
    motion = motion_latent.mean(dim=1)

    audio = F.normalize(audio, dim=-1)
    motion = F.normalize(motion, dim=-1)

    logits = audio @ motion.T  # (B,B)
    labels = torch.arange(audio.shape[0], device=audio.device)
    return F.cross_entropy(logits / temperature, labels)

def compute_motion_energy(skeleton):
    vel = skeleton[:, 1:] - skeleton[:, :-1]
    speed = torch.norm(vel, dim=-1).mean(dim=-1)
    return F.pad(speed, (1,0))

def onset_loss(skeleton, onset):
    energy = compute_motion_energy(skeleton)

    energy = (energy - energy.mean(dim=1, keepdim=True)) / (
        energy.std(dim=1, keepdim=True) + 1e-6
    )

    onset = onset.squeeze(-1)

    return F.mse_loss(energy, onset)

LEFT_HAND_IDXS = [i for i in range(15,36)]

def pitch_loss(skeleton, pitch):

    finger = skeleton[:, :, LEFT_HAND_IDXS, 1]  # y-axis
    finger = finger.mean(dim=-1)

    finger = (finger - finger.mean(dim=1, keepdim=True)) / (
        finger.std(dim=1, keepdim=True) + 1e-6
    )

    pitch = pitch.squeeze(-1).clamp(0,1)

    return F.mse_loss(finger, pitch)


def masked_l1_loss(pred, target, mask):
    mask = mask.float()
    loss = torch.abs(pred - target) * mask
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom


def masked_mse_loss(pred, target, mask):
    mask = mask.float()
    loss = (pred - target) ** 2 * mask
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom


def _feature_weight_tensor(reference, feature_weights):
    if feature_weights is None:
        return None
    weights = torch.as_tensor(feature_weights, dtype=reference.dtype, device=reference.device)
    if weights.numel() != reference.shape[-1]:
        raise ValueError(
            f"feature_weights must have {reference.shape[-1]} values, got {weights.numel()}."
        )
    return weights.view(*([1] * (reference.ndim - 1)), -1)


def relation_string_accuracy(pred, target, valid, start, end):
    pred_idx = pred[..., start:end].argmax(dim=-1)
    target_idx = target[..., start:end].argmax(dim=-1)
    valid = valid.bool()
    if valid.sum() == 0:
        return pred.new_tensor(0.0)
    return (pred_idx[valid] == target_idx[valid]).float().mean()


def normalize_categorical_groups(categorical_groups, relation_dim=None):
    """Normalize YAML group specs into plain dictionaries."""
    if not categorical_groups:
        return []
    groups = []
    for index, group in enumerate(categorical_groups):
        name = str(group.get("name", f"group_{index}"))
        start = int(group["start"])
        end = int(group["end"])
        weight = float(group.get("weight", 1.0))
        if start < 0 or end <= start:
            raise ValueError(f"Invalid categorical group '{name}': [{start}, {end}).")
        if relation_dim is not None and end > int(relation_dim):
            raise ValueError(
                f"Categorical group '{name}' ends at {end}, but relation_dim={relation_dim}."
            )
        if weight < 0:
            raise ValueError(f"Categorical group '{name}' weight must be non-negative.")
        groups.append({"name": name, "start": start, "end": end, "weight": weight})
    return groups


def grouped_categorical_prediction_loss(
    pred,
    target,
    mask,
    categorical_groups,
):
    """Cross entropy for concatenated one-hot groups such as SyncViolinist PP."""
    weighted_loss = pred.new_tensor(0.0)
    weighted_accuracy = pred.new_tensor(0.0)
    total_weight = 0.0
    group_stats = {}
    valid_losses = []
    valid_accuracies = []

    for group in categorical_groups:
        name = group["name"]
        start = int(group["start"])
        end = int(group["end"])
        weight = float(group["weight"])
        logits = pred[..., start:end]
        labels = target[..., start:end].argmax(dim=-1)
        valid = mask[..., start:end].bool().all(dim=-1)
        if valid.any():
            group_loss = F.cross_entropy(logits[valid], labels[valid])
            group_accuracy = (logits[valid].argmax(dim=-1) == labels[valid]).float().mean()
            valid_losses.append(group_loss)
            valid_accuracies.append(group_accuracy)
            if weight > 0:
                weighted_loss = weighted_loss + weight * group_loss
                weighted_accuracy = weighted_accuracy + weight * group_accuracy
                total_weight += weight
        else:
            group_loss = pred.new_tensor(0.0)
            group_accuracy = pred.new_tensor(0.0)
        group_stats[f"{name}_ce"] = group_loss.detach()
        group_stats[f"{name}_acc"] = group_accuracy.detach()

    if total_weight > 0:
        loss = weighted_loss / total_weight
        accuracy = weighted_accuracy / total_weight
    elif valid_losses:
        loss = torch.stack(valid_losses).mean()
        accuracy = torch.stack(valid_accuracies).mean()
    else:
        loss = pred.sum() * 0.0
        accuracy = pred.new_tensor(0.0)

    return loss, accuracy, group_stats


def relation_prediction_loss(
    pred,
    target,
    mask,
    contact_valid,
    bow_valid,
    validity_logits=None,
    mse_weight=1.0,
    l1_weight=1.0,
    contact_ce_weight=0.5,
    bow_ce_weight=0.5,
    validity_bce_weight=0.0,
    contact_class_weights=None,
    bow_class_weights=None,
    continuous_feature_weights=None,
    validity_pos_weight=None,
    categorical_groups=None,
):
    categorical_groups = normalize_categorical_groups(categorical_groups, relation_dim=pred.shape[-1])
    if categorical_groups:
        categorical_ce, categorical_acc, group_stats = grouped_categorical_prediction_loss(
            pred,
            target,
            mask,
            categorical_groups,
        )
        first_name = categorical_groups[0]["name"]
        last_name = categorical_groups[-1]["name"]
        zero = pred.new_tensor(0.0)
        stats = {
            "mse": zero.detach(),
            "l1": zero.detach(),
            "contact_ce": group_stats[f"{first_name}_ce"],
            "bow_ce": group_stats[f"{last_name}_ce"],
            "contact_acc": group_stats[f"{first_name}_acc"],
            "bow_acc": group_stats[f"{last_name}_acc"],
            "categorical_ce": categorical_ce.detach(),
            "categorical_acc": categorical_acc.detach(),
            "validity_bce": zero.detach(),
            "validity_acc": zero.detach(),
            "validity_f1": zero.detach(),
            **group_stats,
        }
        return categorical_ce, stats

    continuous_mask = mask.clone()
    continuous_mask[..., 0:4] = 0
    continuous_mask[..., 6:10] = 0
    feature_weights = _feature_weight_tensor(pred, continuous_feature_weights)
    if feature_weights is not None:
        continuous_mask = continuous_mask * feature_weights

    loss_mse = masked_mse_loss(pred, target, continuous_mask)
    loss_l1 = masked_l1_loss(pred, target, continuous_mask)
    loss = mse_weight * loss_mse + l1_weight * loss_l1

    loss_contact = pred.new_tensor(0.0)
    if contact_ce_weight > 0 and contact_valid.sum() > 0:
        contact_logits = pred[..., 0:4][contact_valid.bool()]
        contact_target = target[..., 0:4].argmax(dim=-1)[contact_valid.bool()]
        contact_weight = None
        if contact_class_weights is not None:
            contact_weight = torch.as_tensor(
                contact_class_weights, dtype=pred.dtype, device=pred.device
            )
        loss_contact = F.cross_entropy(contact_logits, contact_target, weight=contact_weight)
        loss = loss + contact_ce_weight * loss_contact

    loss_bow = pred.new_tensor(0.0)
    if bow_ce_weight > 0 and bow_valid.sum() > 0:
        bow_logits = pred[..., 6:10][bow_valid.bool()]
        bow_target = target[..., 6:10].argmax(dim=-1)[bow_valid.bool()]
        bow_weight = None
        if bow_class_weights is not None:
            bow_weight = torch.as_tensor(bow_class_weights, dtype=pred.dtype, device=pred.device)
        loss_bow = F.cross_entropy(bow_logits, bow_target, weight=bow_weight)
        loss = loss + bow_ce_weight * loss_bow

    contact_acc = relation_string_accuracy(pred, target, contact_valid, 0, 4)
    bow_acc = relation_string_accuracy(pred, target, bow_valid, 6, 10)

    validity_bce = pred.new_tensor(0.0)
    validity_acc = pred.new_tensor(0.0)
    validity_f1 = pred.new_tensor(0.0)
    if validity_logits is not None and validity_bce_weight > 0:
        validity_target = mask.float().clamp(0.0, 1.0)
        pos_weight = None
        if validity_pos_weight is not None:
            pos_weight = torch.as_tensor(
                validity_pos_weight, dtype=pred.dtype, device=pred.device
            )
        validity_bce = F.binary_cross_entropy_with_logits(
            validity_logits,
            validity_target,
            pos_weight=pos_weight,
        )
        loss = loss + validity_bce_weight * validity_bce
        validity_pred = validity_logits.sigmoid() >= 0.5
        validity_true = validity_target >= 0.5
        validity_acc = (validity_pred == validity_true).float().mean()
        tp = (validity_pred & validity_true).sum().float()
        fp = (validity_pred & ~validity_true).sum().float()
        fn = (~validity_pred & validity_true).sum().float()
        validity_f1 = (2 * tp / (2 * tp + fp + fn).clamp_min(1.0))

    return loss, {
        "mse": loss_mse.detach(),
        "l1": loss_l1.detach(),
        "contact_ce": loss_contact.detach(),
        "bow_ce": loss_bow.detach(),
        "contact_acc": contact_acc.detach(),
        "bow_acc": bow_acc.detach(),
        "categorical_ce": pred.new_tensor(0.0).detach(),
        "categorical_acc": pred.new_tensor(0.0).detach(),
        "validity_bce": validity_bce.detach(),
        "validity_acc": validity_acc.detach(),
        "validity_f1": validity_f1.detach(),
    }


def compute_spd_relation_class_weights(dataset, max_weight=4.0):
    """Compute balanced weights for contact/bow string classes from training files."""
    contact_counts = np.zeros(4, dtype=np.float64)
    bow_counts = np.zeros(4, dtype=np.float64)
    for record in dataset.records:
        with np.load(record["_relation_path"], allow_pickle=False) as archive:
            relation = np.asarray(archive[dataset.relation_key], dtype=np.float32)
            contact_valid = np.asarray(archive["contact_valid"], dtype=bool)
            bow_valid = np.asarray(archive["bow_valid"], dtype=bool)
        if relation.ndim != 2 or relation.shape[-1] < 10:
            continue
        if np.any(contact_valid):
            contact_counts += np.bincount(
                relation[contact_valid, 0:4].argmax(axis=-1), minlength=4
            )
        if np.any(bow_valid):
            bow_counts += np.bincount(
                relation[bow_valid, 6:10].argmax(axis=-1), minlength=4
            )

    def balanced(counts):
        weights = np.ones_like(counts)
        positive = counts > 0
        if np.any(positive):
            weights[positive] = counts.sum() / (positive.sum() * counts[positive])
            weights[~positive] = 0.0
            weights[positive] /= max(weights[positive].mean(), 1e-8)
            weights[positive] = np.clip(weights[positive], 1.0 / max_weight, max_weight)
        return weights.astype(np.float32)

    return balanced(contact_counts), balanced(bow_counts), contact_counts, bow_counts


def compute_spd_relation_validity_pos_weight(dataset, max_weight=20.0):
    """Compute per-feature BCE positive weights for the validity head."""
    positive = None
    total = 0
    for record in dataset.records:
        with np.load(record["_relation_path"], allow_pickle=False) as archive:
            mask = np.asarray(archive[dataset.relation_mask_key], dtype=bool)
        if mask.ndim != 2:
            continue
        if positive is None:
            positive = np.zeros(mask.shape[-1], dtype=np.float64)
        positive += mask.sum(axis=0)
        total += mask.shape[0]
    if positive is None or total <= 0:
        return None
    negative = float(total) - positive
    weights = np.ones_like(positive, dtype=np.float32)
    valid = positive > 0
    weights[valid] = np.clip(negative[valid] / positive[valid], 1.0, max_weight)
    weights[~valid] = 1.0
    return weights


def compute_relation_feature_mae(pred, target, mask):
    """Return per-feature masked MAE, useful for diagnosing cascade failures."""
    values = []
    for index in range(pred.shape[-1]):
        valid = mask[..., index].bool()
        if valid.any():
            values.append(torch.abs(pred[..., index][valid] - target[..., index][valid]).mean())
        else:
            values.append(pred.new_tensor(0.0))
    return torch.stack(values)


@torch.no_grad()
def evaluate_spd_relations(
    model,
    dataloader,
    accelerator,
    loss_kwargs,
    selection_contact_weight: float = 0.0,
    selection_bow_weight: float = 0.0,
    selection_validity_weight: float = 0.0,
    selection_categorical_weight: float = 0.0,
    predict_validity: Optional[bool] = None,
):
    model.eval()
    totals = {"loss": 0.0}
    n_batches = 0

    use_validity = bool(getattr(model, "predict_validity", False)) if predict_validity is None else bool(predict_validity)
    for batch in dataloader:
        if use_validity:
            pred, validity_logits = model.forward_with_validity(
                batch["mel"].float(),
                batch["onset"].float(),
                batch["pitch"].float(),
                optional_pretrained_audio(batch),
                optional_instrument_id(batch),
            )
        else:
            pred = model(
                batch["mel"].float(),
                batch["onset"].float(),
                batch["pitch"].float(),
                optional_pretrained_audio(batch),
                optional_instrument_id(batch),
            )
            validity_logits = None
        target = batch["relation"].float()
        mask = batch["relation_mask"].float()
        contact_valid = batch["contact_valid"].bool()
        bow_valid = batch["bow_valid"].bool()

        loss, stats = relation_prediction_loss(
            pred,
            target,
            mask,
            contact_valid,
            bow_valid,
            validity_logits=validity_logits,
            **loss_kwargs,
        )
        gathered_loss = accelerator.gather_for_metrics(loss.detach().reshape(1)).mean()
        totals["loss"] += gathered_loss.item()
        for key, value in stats.items():
            gathered = accelerator.gather_for_metrics(value.reshape(1)).mean()
            totals[key] = totals.get(key, 0.0) + gathered.item()
        n_batches += 1

    model.train()
    if n_batches == 0:
        return totals
    result = {key: value / n_batches for key, value in totals.items()}
    result["selection_score"] = (
        result["loss"]
        + float(selection_contact_weight) * (1.0 - result["contact_acc"])
        + float(selection_bow_weight) * (1.0 - result["bow_acc"])
        + float(selection_validity_weight) * (1.0 - result["validity_f1"])
        + float(selection_categorical_weight) * (1.0 - result["categorical_acc"])
    )
    return result

# =========================
# Train
# =========================

def main_spd_relations(
    output_dir: str,
    train_data: Dict,
    validation_data: Dict = None,
    train_batch_size: int = 4,
    val_batch_size: int = 4,
    max_train_steps: int = 10000,
    learning_rate: float = 1e-4,
    latent_dim: int = 256,
    seed: int = 42,
    save_steps: int = 1000,
    validation_steps: int = 1000,
    num_workers: int = 4,
    mse_weight: float = 1.0,
    l1_weight: float = 1.0,
    contact_ce_weight: float = 0.5,
    bow_ce_weight: float = 0.5,
    validity_bce_weight: float = 0.5,
    predict_validity: bool = True,
    use_positional_encoding: bool = True,
    pretrained_audio_hidden_dim: int = 128,
    num_instruments: int = 0,
    instrument_embedding_dim: int = 16,
    dropout: float = 0.1,
    max_grad_norm: float = 1.0,
    warmup_steps: int = 1000,
    min_lr_ratio: float = 0.1,
    selection_contact_weight: float = 0.5,
    selection_bow_weight: float = 0.5,
    selection_validity_weight: float = 0.2,
    selection_categorical_weight: float = 0.5,
    class_weight_max: float = 4.0,
    validity_pos_weight_max: float = 20.0,
    continuous_feature_weights=None,
    categorical_groups=None,
    mapping_model_type: str = "transformer",
    structured_tcn_layers: int = 4,
    structured_tcn_kernel_size: int = 5,
    structured_gru_layers: int = 2,
):
    accelerator = Accelerator()
    set_seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    if continuous_feature_weights is not None:
        continuous_feature_weights = [float(value) for value in continuous_feature_weights]

    train_num_workers = int(train_data.get("num_workers", num_workers))
    train_dataset = SPDCanonicalRelationsDataset(**train_data)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=train_num_workers,
        pin_memory=True,
    )

    validation_dataloader = None
    if validation_data is not None:
        val_num_workers = int(validation_data.get("num_workers", num_workers))
        validation_dataset = SPDCanonicalRelationsDataset(**validation_data)
        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=val_num_workers,
            pin_memory=True,
        )

    sample = train_dataset[0]
    mel_dim = sample["mel"].shape[-1]
    pretrained_audio_dim = infer_pretrained_audio_dim(sample)
    relation_dim = sample["relation"].shape[-1]
    categorical_groups = normalize_categorical_groups(categorical_groups, relation_dim=relation_dim)
    if categorical_groups and predict_validity:
        logger.warning(
            "categorical_groups is enabled; the validity head is not used by the grouped PP loss. "
            "Set predict_validity=false for this task."
        )

    onset_dim = infer_onset_dim(sample)
    mapping_model_type = normalize_mapping_model_type(mapping_model_type)
    model_metadata = dict(mapping_model_type=mapping_model_type, onset_dim=int(onset_dim),
                          structured_tcn_layers=int(structured_tcn_layers),
                          structured_tcn_kernel_size=int(structured_tcn_kernel_size),
                          structured_gru_layers=int(structured_gru_layers))
    model = create_relation_predictor(
        model_type=mapping_model_type,
        mel_dim=mel_dim,
        relation_dim=relation_dim,
        latent_dim=latent_dim,
        dropout=dropout,
        predict_validity=predict_validity,
        use_positional_encoding=use_positional_encoding,
        pretrained_audio_dim=pretrained_audio_dim,
        pretrained_audio_hidden_dim=pretrained_audio_hidden_dim,
        num_instruments=num_instruments,
        instrument_embedding_dim=instrument_embedding_dim,
        onset_dim=onset_dim,
        categorical_groups=categorical_groups,
        structured_tcn_layers=structured_tcn_layers,
        structured_tcn_kernel_size=structured_tcn_kernel_size,
        structured_gru_layers=structured_gru_layers,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def lr_lambda(step):
        if warmup_steps and step < int(warmup_steps):
            return max(float(step + 1) / float(warmup_steps), 1e-3)
        decay_steps = max(1, int(max_train_steps) - int(warmup_steps))
        progress = min(1.0, max(0.0, (step - int(warmup_steps)) / decay_steps))
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return float(min_lr_ratio) + (1.0 - float(min_lr_ratio)) * float(cosine)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if categorical_groups:
        contact_class_weights = bow_class_weights = np.ones(4, dtype=np.float32)
        contact_counts = bow_counts = np.zeros(4, dtype=np.float64)
        validity_pos_weight = None
    else:
        contact_class_weights, bow_class_weights, contact_counts, bow_counts = (
            compute_spd_relation_class_weights(train_dataset, max_weight=class_weight_max)
        )
        validity_pos_weight = compute_spd_relation_validity_pos_weight(
            train_dataset,
            max_weight=validity_pos_weight_max,
        )

    if validation_dataloader is not None:
        model, optimizer, scheduler, train_dataloader, validation_dataloader = accelerator.prepare(
            model, optimizer, scheduler, train_dataloader, validation_dataloader
        )
    else:
        model, optimizer, scheduler, train_dataloader = accelerator.prepare(
            model, optimizer, scheduler, train_dataloader
        )

    loss_kwargs = dict(
        mse_weight=mse_weight,
        l1_weight=l1_weight,
        contact_ce_weight=contact_ce_weight,
        bow_ce_weight=bow_ce_weight,
        validity_bce_weight=validity_bce_weight,
        contact_class_weights=contact_class_weights,
        bow_class_weights=bow_class_weights,
        continuous_feature_weights=continuous_feature_weights,
        validity_pos_weight=validity_pos_weight,
        categorical_groups=categorical_groups,
    )
    evaluation_loss_kwargs = dict(loss_kwargs)

    logger.info("***** Running relation prediction training *****")
    logger.info(f"  Num train windows = {len(train_dataset)}")
    logger.info(f"  Relation dim = {relation_dim}")
    logger.info(f"  Mapping backbone = {mapping_model_type}")
    logger.info(f"  Mel dim = {mel_dim}")
    logger.info(f"  Pretrained audio dim = {pretrained_audio_dim}")
    logger.info(f"  Instrument classes = {int(num_instruments)}")
    logger.info(f"  Predict validity = {bool(predict_validity)}")
    logger.info(f"  Use positional encoding = {bool(use_positional_encoding)}")
    if categorical_groups:
        logger.info(f"  Categorical groups = {categorical_groups}")
    else:
        logger.info(f"  Contact class counts = {contact_counts.tolist()}")
        logger.info(f"  Bow class counts = {bow_counts.tolist()}")
        logger.info(f"  Contact class weights = {contact_class_weights.tolist()}")
        logger.info(f"  Bow class weights = {bow_class_weights.tolist()}")
    if validity_pos_weight is not None:
        logger.info(f"  Validity positive weights = {validity_pos_weight.tolist()}")

    global_step = 0
    best_val_score = float("inf")
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)

    while global_step < max_train_steps:
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                if predict_validity:
                    pred, validity_logits = model.forward_with_validity(
                        batch["mel"].float(),
                        batch["onset"].float(),
                        batch["pitch"].float(),
                        optional_pretrained_audio(batch),
                        optional_instrument_id(batch),
                    )
                else:
                    pred = model(
                        batch["mel"].float(),
                        batch["onset"].float(),
                        batch["pitch"].float(),
                        optional_pretrained_audio(batch),
                        optional_instrument_id(batch),
                    )
                    validity_logits = None
                target = batch["relation"].float()
                mask = batch["relation_mask"].float()
                contact_valid = batch["contact_valid"].bool()
                bow_valid = batch["bow_valid"].bool()

                loss, stats = relation_prediction_loss(
                    pred,
                    target,
                    mask,
                    contact_valid,
                    bow_valid,
                    validity_logits=validity_logits,
                    **loss_kwargs,
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients and max_grad_norm and max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                scheduler.step()
                progress_bar.update(1)
                global_step += 1

                progress_bar.set_postfix({
                    "step": global_step,
                    "loss": f"{loss.item():.4f}",
                    "l1": f"{stats['l1'].item():.4f}",
                    "mse": f"{stats['mse'].item():.4f}",
                    "cat_acc": f"{stats['categorical_acc'].item():.3f}",
                    "c_acc": f"{stats['contact_acc'].item():.3f}",
                    "b_acc": f"{stats['bow_acc'].item():.3f}",
                    "v_f1": f"{stats['validity_f1'].item():.3f}",
                })

                if accelerator.is_main_process and global_step % 50 == 0:
                    train_group_accuracy = " ".join(
                        f"{group['name']}_acc={stats[group['name'] + '_acc'].item():.4f}"
                        for group in categorical_groups
                    )
                    logger.info(
                        f"step {global_step} "
                        f"loss={loss.item():.6f} "
                        f"l1={stats['l1'].item():.6f} "
                        f"mse={stats['mse'].item():.6f} "
                        f"categorical_acc={stats['categorical_acc'].item():.4f} "
                        f"contact_acc={stats['contact_acc'].item():.4f} "
                        f"bow_acc={stats['bow_acc'].item():.4f} "
                        f"validity_f1={stats['validity_f1'].item():.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.3e} "
                        f"{train_group_accuracy}"
                    )

                if validation_dataloader is not None and validation_steps > 0 and global_step % validation_steps == 0:
                    val_stats = evaluate_spd_relations(
                        model,
                        validation_dataloader,
                        accelerator,
                        evaluation_loss_kwargs,
                        selection_contact_weight=selection_contact_weight,
                        selection_bow_weight=selection_bow_weight,
                        selection_validity_weight=selection_validity_weight if predict_validity else 0.0,
                        selection_categorical_weight=(
                            selection_categorical_weight if categorical_groups else 0.0
                        ),
                        predict_validity=predict_validity,
                    )
                    if accelerator.is_main_process:
                        validation_group_accuracy = " ".join(
                            f"{group['name']}_acc={val_stats[group['name'] + '_acc']:.4f}"
                            for group in categorical_groups
                        )
                        logger.info(
                            f"validation step {global_step} "
                            f"loss={val_stats['loss']:.6f} "
                            f"l1={val_stats['l1']:.6f} "
                            f"mse={val_stats['mse']:.6f} "
                            f"categorical_acc={val_stats['categorical_acc']:.4f} "
                            f"contact_acc={val_stats['contact_acc']:.4f} "
                            f"bow_acc={val_stats['bow_acc']:.4f} "
                            f"validity_f1={val_stats['validity_f1']:.4f} "
                            f"selection_score={val_stats['selection_score']:.6f} "
                            f"{validation_group_accuracy}"
                        )
                        if val_stats["selection_score"] < best_val_score:
                            best_val_score = val_stats["selection_score"]
                            accelerator.save({
                                **model_metadata,
                                "model": accelerator.unwrap_model(model).state_dict(),
                                "relation_dim": relation_dim,
                                "mel_dim": mel_dim,
                                "pretrained_audio_dim": int(pretrained_audio_dim),
                                "pretrained_audio_hidden_dim": int(pretrained_audio_hidden_dim),
                                "num_instruments": int(num_instruments),
                                "instrument_embedding_dim": int(instrument_embedding_dim),
                                "global_step": global_step,
                                "val_stats": val_stats,
                                "predict_validity": bool(predict_validity),
                                "use_positional_encoding": bool(use_positional_encoding),
                                "latent_dim": int(latent_dim),
                                "dropout": float(dropout),
                                "categorical_groups": categorical_groups,
                                "validity_bce_weight": float(validity_bce_weight),
                                "continuous_feature_weights": continuous_feature_weights,
                                "contact_class_weights": contact_class_weights.tolist(),
                                "bow_class_weights": bow_class_weights.tolist(),
                                "validity_pos_weight": (
                                    validity_pos_weight.tolist() if validity_pos_weight is not None else None
                                ),
                            }, os.path.join(output_dir, "best.pt"))

                if save_steps > 0 and global_step % save_steps == 0:
                    save_path = os.path.join(output_dir, f"checkpoint-{global_step}.pt")
                    accelerator.save({
                        **model_metadata,
                        "model": accelerator.unwrap_model(model).state_dict(),
                        "relation_dim": relation_dim,
                        "mel_dim": mel_dim,
                        "pretrained_audio_dim": int(pretrained_audio_dim),
                        "pretrained_audio_hidden_dim": int(pretrained_audio_hidden_dim),
                        "num_instruments": int(num_instruments),
                        "instrument_embedding_dim": int(instrument_embedding_dim),
                        "global_step": global_step,
                        "predict_validity": bool(predict_validity),
                        "use_positional_encoding": bool(use_positional_encoding),
                        "latent_dim": int(latent_dim),
                        "dropout": float(dropout),
                        "categorical_groups": categorical_groups,
                        "validity_bce_weight": float(validity_bce_weight),
                    }, save_path)

            if global_step >= max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save({
            **model_metadata,
            "model": accelerator.unwrap_model(model).state_dict(),
            "relation_dim": relation_dim,
            "mel_dim": mel_dim,
            "pretrained_audio_dim": int(pretrained_audio_dim),
            "pretrained_audio_hidden_dim": int(pretrained_audio_hidden_dim),
            "num_instruments": int(num_instruments),
            "instrument_embedding_dim": int(instrument_embedding_dim),
            "global_step": global_step,
            "predict_validity": bool(predict_validity),
            "use_positional_encoding": bool(use_positional_encoding),
            "latent_dim": int(latent_dim),
            "dropout": float(dropout),
            "categorical_groups": categorical_groups,
            "validity_bce_weight": float(validity_bce_weight),
            "continuous_feature_weights": continuous_feature_weights,
            "contact_class_weights": contact_class_weights.tolist(),
            "bow_class_weights": bow_class_weights.tolist(),
            "validity_pos_weight": (
                validity_pos_weight.tolist() if validity_pos_weight is not None else None
            ),
        }, os.path.join(output_dir, "final.pt"))

    # Older Accelerate releases do not create ``trackers`` when no tracker
    # backend is configured. Training and final.pt are already complete here.
    if hasattr(accelerator, "trackers"):
        accelerator.end_training()


def train_legacy_mapping(
    output_dir: str,
    train_data: Dict,
    task: str = "contrastive",
    validation_data: Dict = None,
    val_batch_size: int = 4,
    save_steps: int = 1000,
    validation_steps: int = 1000,
    num_workers: int = 4,
    mse_weight: float = 1.0,
    l1_weight: float = 1.0,
    contact_ce_weight: float = 0.5,
    bow_ce_weight: float = 0.5,
    validity_bce_weight: float = 0.5,
    predict_validity: bool = True,
    use_positional_encoding: bool = True,
    pretrained_audio_hidden_dim: int = 128,
    num_instruments: int = 0,
    instrument_embedding_dim: int = 16,
    dropout: float = 0.1,
    max_grad_norm: float = 1.0,
    warmup_steps: int = 1000,
    min_lr_ratio: float = 0.1,
    selection_contact_weight: float = 0.5,
    selection_bow_weight: float = 0.5,
    selection_validity_weight: float = 0.2,
    selection_categorical_weight: float = 0.5,
    class_weight_max: float = 4.0,
    validity_pos_weight_max: float = 20.0,
    continuous_feature_weights=None,
    categorical_groups=None,
    mapping_model_type: str = "transformer",
    structured_tcn_layers: int = 4,
    structured_tcn_kernel_size: int = 5,
    structured_gru_layers: int = 2,
    train_batch_size: int = 4,
    max_train_steps: int = 10000,
    learning_rate: float = 1e-4,
    latent_dim: int = 256,
    seed: int = 42,
):
    if task in ("spd_relations", "relation_prediction"):
        return main_spd_relations(
            output_dir=output_dir,
            train_data=train_data,
            validation_data=validation_data,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            max_train_steps=max_train_steps,
            learning_rate=learning_rate,
            latent_dim=latent_dim,
            seed=seed,
            save_steps=save_steps,
            validation_steps=validation_steps,
            num_workers=num_workers,
            mse_weight=mse_weight,
            l1_weight=l1_weight,
            contact_ce_weight=contact_ce_weight,
            bow_ce_weight=bow_ce_weight,
            validity_bce_weight=validity_bce_weight,
            predict_validity=predict_validity,
            use_positional_encoding=use_positional_encoding,
            pretrained_audio_hidden_dim=pretrained_audio_hidden_dim,
            num_instruments=num_instruments,
            instrument_embedding_dim=instrument_embedding_dim,
            dropout=dropout,
            max_grad_norm=max_grad_norm,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
            selection_contact_weight=selection_contact_weight,
            selection_bow_weight=selection_bow_weight,
            selection_validity_weight=selection_validity_weight,
            selection_categorical_weight=selection_categorical_weight,
            class_weight_max=class_weight_max,
            validity_pos_weight_max=validity_pos_weight_max,
            continuous_feature_weights=continuous_feature_weights,
            categorical_groups=categorical_groups,
            mapping_model_type=mapping_model_type,
            structured_tcn_layers=structured_tcn_layers,
            structured_tcn_kernel_size=structured_tcn_kernel_size,
            structured_gru_layers=structured_gru_layers,
        )


    accelerator = Accelerator()
    set_seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # dataset
    train_dataset = AudioPoseDataset(**train_data)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=4
    )

    sample = train_dataset[0]
    audio_dim = sample["mel"].shape[-1]
    J, C = sample["skeleton"].shape[-2:]

    # models
    audio_encoder = AudioEncoder(audio_dim)
    motion_encoder = MotionEncoder(J*C, latent_dim)

    optimizer = torch.optim.AdamW(
        list(audio_encoder.parameters()) + list(motion_encoder.parameters()),
        lr=learning_rate
    )

    audio_encoder, motion_encoder, optimizer, train_dataloader = accelerator.prepare(
        audio_encoder, motion_encoder, optimizer, train_dataloader
    )

    # training
    logger.info("***** Running mapping training *****")
    global_step = 0
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)

    while global_step < max_train_steps:
        for batch in train_dataloader:

            with accelerator.accumulate(audio_encoder):

                mel = batch["mel"].float()  # (B,T,F)
                onset = batch["onset"].float()
                pitch = batch["pitch"].float()
                motion = batch["skeleton"].float()

                # forward
                audio_latent = audio_encoder(mel, onset, pitch)
                motion_latent = motion_encoder(motion)

                loss_align = info_nce_loss(audio_latent, motion_latent)

                loss_on = onset_loss(motion, onset)
                loss_pi = pitch_loss(motion, pitch)

                λ_on = 0.1
                λ_pi = 0.05

                loss = loss_align + λ_on * loss_on + λ_pi * loss_pi
                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # 把所有 loss 显示在进度条上
                progress_bar.set_postfix({
                    "step": global_step,
                    "loss": f"{loss.item():.4f}",
                    "align": f"{loss_align.item():.4f}",
                    "onset": f"{loss_on.item():.4f}",
                    "pitch": f"{loss_pi.item():.4f}",
                })

                if accelerator.is_main_process:
                    logger.info(f"step {global_step} loss: {loss.item():.6f}")

                if global_step % 1000 == 0:
                    save_path = os.path.join(output_dir, f"checkpoint-{global_step}.pt")
                    accelerator.save({
                        "audio_encoder": accelerator.unwrap_model(audio_encoder).state_dict(),
                        "motion_encoder": accelerator.unwrap_model(motion_encoder).state_dict()
                    }, save_path)

            if global_step >= max_train_steps:
                break

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        torch.save({
            "audio_encoder": accelerator.unwrap_model(audio_encoder).state_dict(),
            "motion_encoder": accelerator.unwrap_model(motion_encoder).state_dict()
        }, os.path.join(output_dir, "final.pt"))

    # Keep the mapping entrypoint compatible with older Accelerate releases
    # that omit ``trackers`` when tracker logging is disabled.
    if hasattr(accelerator, "trackers"):
        accelerator.end_training()


# Temporal supervision and the Stage-1 training loop.

def _feature_weights(
    reference: torch.Tensor,
    weights: Optional[Sequence[float]],
) -> torch.Tensor:
    if weights is None:
        values = torch.ones(reference.shape[-1], device=reference.device, dtype=reference.dtype)
        values[0:4] = 0.0
        values[6:10] = 0.0
    else:
        values = torch.as_tensor(weights, device=reference.device, dtype=reference.dtype)
        if values.numel() != reference.shape[-1]:
            raise ValueError(
                f"temporal_feature_weights must contain {reference.shape[-1]} values, "
                f"got {values.numel()}."
            )
    return values.view(1, 1, -1)


def _weighted_masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    feature_weights: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    weight = mask.float() * feature_weights
    if not torch.any(weight > 0):
        return pred.sum() * 0.0
    values = F.smooth_l1_loss(pred, target, reduction="none", beta=float(beta))
    return (values * weight).sum() / weight.sum().clamp_min(1.0)


def temporal_relation_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    bow_valid: torch.Tensor,
    temporal_feature_weights: Optional[Sequence[float]] = None,
    smooth_l1_beta: float = 0.02,
    bow_t_direction_temperature: float = 0.05,
    bow_t_min_velocity: float = 0.005,
    bow_t_event_margin: float = 0.005,
    bow_t_reversal_event_weight: float = 8.0,
    bow_t_reversal_positive_weight: float = 1.0,
    bow_t_ignore_string_transitions: bool = True,
    bow_mid_min_speed: float = 0.005,
) -> Dict[str, torch.Tensor]:
    """Return differentiable temporal losses for a batch of relation windows."""
    if pred.ndim != 3 or target.shape != pred.shape or mask.shape != pred.shape:
        raise ValueError(
            "pred, target and mask must share shape (B, T, F), got "
            f"{tuple(pred.shape)}, {tuple(target.shape)}, {tuple(mask.shape)}."
        )
    if pred.shape[1] < 2:
        zero = pred.sum() * 0.0
        return {
            "velocity": zero,
            "acceleration": zero,
            "bow_t_direction": zero,
            "bow_t_event_margin": zero,
            "bow_t_static_velocity": zero,
            "bow_t_reversal": zero,
            "bow_t_reversal_consistency": zero,
            "bow_t_reversal_count": zero.detach(),
            "bow_mid_direction": zero,
            "bow_transition_ce": zero,
            "bow_transition_count": zero.detach(),
        }

    weights = _feature_weights(pred, temporal_feature_weights)
    pred_velocity = pred[:, 1:] - pred[:, :-1]
    target_velocity = target[:, 1:] - target[:, :-1]
    pair_mask = mask[:, 1:].float() * mask[:, :-1].float()
    temporal_pair_mask = pair_mask
    if bow_t_ignore_string_transitions and target.shape[-1] >= 10:
        target_bow_class = target[..., 6:10].argmax(dim=-1)
        bow_t_same_string = target_bow_class[:, 1:] == target_bow_class[:, :-1]
        temporal_pair_mask = pair_mask.clone()
        temporal_pair_mask[..., 11] = (
            temporal_pair_mask[..., 11] * bow_t_same_string.float()
        )

    velocity = _weighted_masked_smooth_l1(
        pred_velocity,
        target_velocity,
        temporal_pair_mask,
        weights,
        beta=smooth_l1_beta,
    )

    if pred.shape[1] >= 3:
        pred_acceleration = pred_velocity[:, 1:] - pred_velocity[:, :-1]
        target_acceleration = target_velocity[:, 1:] - target_velocity[:, :-1]
        triple_mask = temporal_pair_mask[:, 1:] * temporal_pair_mask[:, :-1]
        acceleration = _weighted_masked_smooth_l1(
            pred_acceleration,
            target_acceleration,
            triple_mask,
            weights,
            beta=smooth_l1_beta,
        )
    else:
        acceleration = pred.sum() * 0.0

    # bow_t is the contact location along the bow. Its velocity sign is a
    # direct, differentiable proxy for up-bow/down-bow motion. Ignore nearly
    # static target frames so small annotation noise does not define direction.
    bow_t_direction = pred.sum() * 0.0
    bow_t_event_margin_loss = pred.sum() * 0.0
    bow_t_static_velocity_loss = pred.sum() * 0.0
    bow_t_reversal = pred.sum() * 0.0
    bow_t_reversal_consistency = pred.sum() * 0.0
    bow_t_reversal_count = pred.new_tensor(0.0)
    if pred.shape[-1] > 11:
        target_bow_t_velocity = target_velocity[..., 11]
        pred_bow_t_velocity = pred_velocity[..., 11]
        bow_pair_available = (
            bow_valid[:, 1:].bool()
            & bow_valid[:, :-1].bool()
            & temporal_pair_mask[..., 11].bool()
        )
        bow_pair_valid = bow_pair_available & (
            target_bow_t_velocity.abs() >= float(bow_t_min_velocity)
        )
        if bow_pair_valid.any():
            target_up = (target_bow_t_velocity[bow_pair_valid] > 0).to(pred.dtype)
            direction_logits = pred_bow_t_velocity[bow_pair_valid] / max(
                float(bow_t_direction_temperature), 1e-6
            )
            bow_t_direction = F.binary_cross_entropy_with_logits(direction_logits, target_up)

        # Align training with the event detector used by evaluation. Moving
        # target frames must cross the velocity threshold in the correct
        # direction, while target-static frames should stay inside the same
        # threshold. Dividing by the margin keeps both losses dimensionless.
        event_margin = max(float(bow_t_event_margin), 1e-6)
        if bow_pair_valid.any():
            target_sign = torch.sign(target_bow_t_velocity[bow_pair_valid])
            signed_pred_velocity = target_sign * pred_bow_t_velocity[bow_pair_valid]
            bow_t_event_margin_loss = (
                F.relu(event_margin - signed_pred_velocity) / event_margin
            ).mean()
        static_pair_valid = bow_pair_available & (
            target_bow_t_velocity.abs() < float(bow_t_min_velocity)
        )
        if static_pair_valid.any():
            predicted_static_speed = pred_bow_t_velocity[static_pair_valid].abs()
            bow_t_static_velocity_loss = (
                F.relu(predicted_static_speed - event_margin) / event_margin
            ).mean()

        # Reversal frames are sparse, so an unweighted direction objective is
        # dominated by long, constant bow strokes. Emphasize the moving frames
        # immediately before and after each GT direction change.
        reversal_logits = []
        reversal_targets = []
        reversal_weights = []
        reversal_consistency_losses = []
        for batch_index in range(pred.shape[0]):
            moving_indices = torch.nonzero(
                bow_pair_valid[batch_index], as_tuple=False
            ).flatten()
            if moving_indices.numel() < 2:
                continue
            moving_target_velocity = target_bow_t_velocity[batch_index, moving_indices]
            reversal = (
                moving_target_velocity[1:] * moving_target_velocity[:-1]
            ) < 0

            # Match the event metric more closely than frame-wise direction
            # BCE: estimate the probability that two consecutive non-static
            # bow velocities have different signs. Non-reversal pairs directly
            # suppress the extra direction changes that lower precision.
            moving_pred_velocity = pred_bow_t_velocity[batch_index, moving_indices]
            direction_probability = torch.sigmoid(
                moving_pred_velocity
                / max(float(bow_t_direction_temperature), 1e-6)
            )
            predicted_reversal_probability = (
                direction_probability[1:] * (1.0 - direction_probability[:-1])
                + (1.0 - direction_probability[1:]) * direction_probability[:-1]
            ).clamp(min=1e-6, max=1.0 - 1e-6)
            non_reversal = ~reversal
            negative_loss = (
                -torch.log1p(-predicted_reversal_probability[non_reversal]).mean()
                if non_reversal.any()
                else None
            )
            positive_loss = (
                -torch.log(predicted_reversal_probability[reversal]).mean()
                if reversal.any()
                else None
            )
            positive_weight = max(float(bow_t_reversal_positive_weight), 0.0)
            if negative_loss is not None and positive_loss is not None:
                reversal_consistency_losses.append(
                    (negative_loss + positive_weight * positive_loss)
                    / max(1.0 + positive_weight, 1e-6)
                )
            elif negative_loss is not None:
                reversal_consistency_losses.append(negative_loss)
            elif positive_loss is not None and positive_weight > 0.0:
                reversal_consistency_losses.append(positive_loss)

            if not reversal.any():
                continue
            sample_weight = torch.ones_like(moving_target_velocity)
            event_weight = max(float(bow_t_reversal_event_weight), 1.0)
            sample_weight[:-1] = torch.where(
                reversal,
                sample_weight[:-1].new_full(reversal.shape, event_weight),
                sample_weight[:-1],
            )
            sample_weight[1:] = torch.where(
                reversal,
                sample_weight[1:].new_full(reversal.shape, event_weight),
                sample_weight[1:],
            )
            reversal_logits.append(
                pred_bow_t_velocity[batch_index, moving_indices]
                / max(float(bow_t_direction_temperature), 1e-6)
            )
            reversal_targets.append((moving_target_velocity > 0).to(pred.dtype))
            reversal_weights.append(sample_weight)
            bow_t_reversal_count = bow_t_reversal_count + reversal.sum().to(pred.dtype)
        if reversal_logits:
            logits = torch.cat(reversal_logits)
            targets = torch.cat(reversal_targets)
            sample_weights = torch.cat(reversal_weights)
            reversal_bce = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                reduction="none",
            )
            bow_t_reversal = (
                reversal_bce * sample_weights
            ).sum() / sample_weights.sum().clamp_min(1.0)
        if reversal_consistency_losses:
            bow_t_reversal_consistency = torch.stack(
                reversal_consistency_losses
            ).mean()

    # Preserve the direction of bow-midpoint movement in 3D, not just its
    # per-coordinate magnitude. This specifically targets trajectory collapse
    # that can leave point-wise MAE reasonable while BCS remains poor.
    bow_mid_direction = pred.sum() * 0.0
    if pred.shape[-1] >= 20:
        pred_mid_velocity = pred_velocity[..., 17:20]
        target_mid_velocity = target_velocity[..., 17:20]
        mid_pair_valid = (
            bow_valid[:, 1:].bool()
            & bow_valid[:, :-1].bool()
            & pair_mask[..., 17:20].bool().all(dim=-1)
        )
        target_mid_speed = torch.linalg.vector_norm(target_mid_velocity, dim=-1)
        mid_pair_valid &= target_mid_speed >= float(bow_mid_min_speed)
        if mid_pair_valid.any():
            cosine = F.cosine_similarity(
                pred_mid_velocity[mid_pair_valid],
                target_mid_velocity[mid_pair_valid],
                dim=-1,
                eps=1e-6,
            )
            bow_mid_direction = (1.0 - cosine).mean()

    # Frame-wise CE is dominated by long constant-string regions. Reweight the
    # sparse frames on which the GT bow string changes so transition timing is
    # not washed out by the ~91% overall bow classification accuracy.
    bow_transition_ce = pred.sum() * 0.0
    transition_count = pred.new_tensor(0.0)
    if pred.shape[-1] >= 10:
        target_bow_class = target[..., 6:10].argmax(dim=-1)
        bow_transition = (
            bow_valid[:, 1:].bool()
            & bow_valid[:, :-1].bool()
            & (target_bow_class[:, 1:] != target_bow_class[:, :-1])
        )
        transition_count = bow_transition.sum().to(pred.dtype)
        if bow_transition.any():
            bow_transition_ce = F.cross_entropy(
                pred[:, 1:, 6:10][bow_transition],
                target_bow_class[:, 1:][bow_transition],
            )

    return {
        "velocity": velocity,
        "acceleration": acceleration,
        "bow_t_direction": bow_t_direction,
        "bow_t_event_margin": bow_t_event_margin_loss,
        "bow_t_static_velocity": bow_t_static_velocity_loss,
        "bow_t_reversal": bow_t_reversal,
        "bow_t_reversal_consistency": bow_t_reversal_consistency,
        "bow_t_reversal_count": bow_t_reversal_count.detach(),
        "bow_mid_direction": bow_mid_direction,
        "bow_transition_ce": bow_transition_ce,
        "bow_transition_count": transition_count.detach(),
    }


def structured_bow_auxiliary_losses(
    auxiliary: Optional[Mapping[str, torch.Tensor]],
    target: torch.Tensor,
    relation_mask: torch.Tensor,
    bow_valid: torch.Tensor,
    *,
    bow_t_min_velocity: float,
    bow_t_ignore_string_transitions: bool = True,
    direction_class_weights: Optional[Sequence[float]] = None,
    reversal_pos_weight: float = 4.0,
    bow_step_beta: float = 0.05,
    bow_step_target_max: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Supervise explicit bow state emitted by the structured predictor."""
    zero = target.sum() * 0.0
    result = {
        "direction": zero,
        "reversal": zero,
        "bow_step": zero,
        "reversal_count": zero.detach(),
    }
    if auxiliary is None or target.shape[1] < 2 or target.shape[-1] <= 11:
        return result

    direction_logits = auxiliary["bow_direction_logits"]
    reversal_logits = auxiliary["bow_reversal_logits"]
    predicted_step = auxiliary["bow_step"]
    if direction_logits.shape[:2] != target.shape[:2] or direction_logits.shape[-1] != 3:
        raise ValueError("bow_direction_logits must have shape (B, T, 3).")
    if reversal_logits.shape != target.shape[:2] or predicted_step.shape != target.shape[:2]:
        raise ValueError("Structured reversal logits and bow step must have shape (B, T).")

    target_bow_t = target[..., 11]
    target_velocity = target_bow_t[:, 1:] - target_bow_t[:, :-1]
    pair_valid = (
        bow_valid[:, 1:].bool()
        & bow_valid[:, :-1].bool()
        & relation_mask[:, 1:, 11].bool()
        & relation_mask[:, :-1, 11].bool()
    )
    # A change of nearest bow string can create a discontinuity in the
    # geometric bow_t coordinate without representing a physical bow reversal.
    # Keep structured direction/reversal/step supervision consistent with the
    # ordinary temporal loss, which already supports this exclusion.
    if bow_t_ignore_string_transitions and target.shape[-1] >= 10:
        target_bow_class = target[..., 6:10].argmax(dim=-1)
        same_bow_string = target_bow_class[:, 1:] == target_bow_class[:, :-1]
        pair_valid = pair_valid & same_bow_string
    direction_target = torch.ones(target.shape[:2], device=target.device, dtype=torch.long)
    moving_up = pair_valid & (target_velocity >= float(bow_t_min_velocity))
    moving_down = pair_valid & (target_velocity <= -float(bow_t_min_velocity))
    direction_target[:, 1:][moving_down] = 0
    direction_target[:, 1:][moving_up] = 2
    direction_valid = torch.zeros_like(direction_target, dtype=torch.bool)
    direction_valid[:, 1:] = pair_valid
    if direction_valid.any():
        class_weights = None
        if direction_class_weights is not None:
            class_weights = torch.as_tensor(
                direction_class_weights,
                device=target.device,
                dtype=direction_logits.dtype,
            )
            if class_weights.numel() != 3:
                raise ValueError("structured_direction_class_weights must contain 3 values.")
        result["direction"] = F.cross_entropy(
            direction_logits[direction_valid],
            direction_target[direction_valid],
            weight=class_weights,
        )

    reversal_target = torch.zeros_like(reversal_logits)
    reversal_valid = torch.zeros_like(reversal_logits, dtype=torch.bool)
    reversal_valid[:, 1:] = pair_valid
    reversal_count = zero
    for batch_index in range(target.shape[0]):
        moving_indices = torch.nonzero(
            moving_up[batch_index] | moving_down[batch_index], as_tuple=False
        ).flatten()
        if moving_indices.numel() < 2:
            continue
        moving_velocity = target_velocity[batch_index, moving_indices]
        reversal = moving_velocity[1:] * moving_velocity[:-1] < 0
        event_frames = moving_indices[1:][reversal] + 1
        reversal_target[batch_index, event_frames] = 1.0
        reversal_count = reversal_count + reversal.sum().to(target.dtype)
    if reversal_valid.any():
        positive_weight = torch.as_tensor(
            max(float(reversal_pos_weight), 1.0),
            device=target.device,
            dtype=reversal_logits.dtype,
        )
        result["reversal"] = F.binary_cross_entropy_with_logits(
            reversal_logits[reversal_valid],
            reversal_target[reversal_valid],
            pos_weight=positive_weight,
        )

    clamped_bow_t = target_bow_t.clamp(1e-3, 1.0 - 1e-3)
    target_state = torch.logit(clamped_bow_t)
    target_step = (target_state[:, 1:] - target_state[:, :-1]).abs()
    target_step = target_step.clamp(max=max(float(bow_step_target_max), 1e-6))
    if pair_valid.any():
        result["bow_step"] = F.smooth_l1_loss(
            predicted_step[:, 1:][pair_valid],
            target_step[pair_valid],
            beta=float(bow_step_beta),
        )
    result["reversal_count"] = reversal_count.detach()
    return result


def _masked_profile_cosine_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    half_bow: Optional[torch.Tensor] = None,
    centered: bool = False,
) -> torch.Tensor:
    """BCS-aligned sequence cosine: 1 - cos(pred - offset, target - offset).

    The offset reproduces the evaluation metric: the same reference is
    subtracted from both vectors (GT half bow length when available, else the
    per-sample masked target mean), mirroring the shared-offset cosine that
    BCS computes on the frog-bridge distance profile.  When ``centered`` is
    set (signed-offset head), prediction and target already hold the
    centered profile and no offset is subtracted.
    """
    mask = valid.float()
    counts = mask.sum(dim=1)
    active = counts > 1
    if not bool(active.any()):
        return prediction.sum() * 0.0
    pred = prediction[active]
    tgt = target[active]
    mask = mask[active]
    if centered:
        offset = torch.zeros_like(pred)
    elif half_bow is not None:
        offset = half_bow.to(device=pred.device, dtype=pred.dtype)[active] * mask
    else:
        offset = (tgt * mask).sum(dim=1, keepdim=True) / mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
    a = (pred - offset) * mask
    b = (tgt - offset) * mask
    dot = (a * b).sum(dim=1)
    norms = a.norm(dim=1) * b.norm(dim=1)
    usable = norms > 1e-6
    if not bool(usable.any()):
        return pred.sum() * 0.0
    cosine = dot[usable] / norms[usable].clamp_min(1e-8)
    return (1.0 - cosine).mean()


def _masked_pooled_cosine_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    half_bow: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Deployed-BCS pooled cosine over the whole batch.

    The deployed ``elgar_bowing_cosine_similarity`` accumulates dot products
    and squared norms over all valid frames of all takes before a single
    cosine, so takes with large deviations dominate the metric.  This loss
    reproduces that aggregation inside one training batch: one cosine over
    all valid frames of all windows, instead of the mean of per-window
    cosines.  Gradient-wise it is exactly the deployed metric restricted to
    the batch, so it optimizes what is reported.
    """
    mask = valid.float()
    pred = prediction * mask
    tgt = target * mask
    if half_bow is not None:
        half = half_bow.to(device=pred.device, dtype=pred.dtype) * mask
        pred = pred - half
        tgt = tgt - half
    dot = (pred * tgt).sum()
    pred_sq = (pred * pred).sum()
    target_sq = (tgt * tgt).sum()
    if bool(pred_sq <= 1e-12) or bool(target_sq <= 1e-12):
        return prediction.sum() * 0.0
    cosine = dot / (pred_sq.sqrt() * target_sq.sqrt()).clamp_min(1e-8)
    return 1.0 - cosine


def _masked_moving_average_torch(
    values: torch.Tensor, mask: torch.Tensor, window: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Masked centered moving average along the time axis.

    ``values``/``mask`` are ``(B, T)``. Each output frame averages the valid
    frames inside a centered window (zero padding, renormalized by the
    valid count), so the estimate stays unbiased when the window extends
    past the segment edge or into invalid frames. When ``window >= T`` the
    band degenerates to the per-sample masked mean (the slowest component a
    training crop can resolve). Returns ``(moving_average, band_valid)``.
    """
    b, t = values.shape
    w = int(window)
    if w < 1:
        raise ValueError(f"moving-average window must be >= 1, got {window}")
    if w >= t:
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (values * mask).sum(dim=1, keepdim=True) / denom
        return mean, mask > 0
    if w % 2 == 0:
        w += 1
    pad = w // 2
    kernel = torch.ones(1, 1, w, device=values.device, dtype=values.dtype)
    numerator = F.conv1d(
        F.pad((values * mask).unsqueeze(1), (pad, pad)), kernel
    ).squeeze(1)
    denominator = F.conv1d(
        F.pad(mask.unsqueeze(1), (pad, pad)), kernel
    ).squeeze(1)
    moving_average = numerator / denominator.clamp_min(1.0)
    band_valid = denominator > 0
    return moving_average, band_valid


def _masked_multiscale_ma_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    windows: Sequence[int],
    beta: float,
) -> Dict[str, torch.Tensor]:
    """Multi-scale smooth-L1 on moving averages of the scalar profile.

    The frozen-e2e probes (13.12.3/13.12.4) showed the dominant BCS gap is
    non-affine low-frequency shape error: the deployed head tracks
    frame-level detail (pooled cosine ~0.95) while its slow component
    drifts within takes. This loss supervises the profile after centered
    box smoothing at several time scales, so gradient budget is spent on
    the slow shape the deployed Stage-2 actually reacts to. Windows >= the
    crop length degenerate to masked-mean (DC) matching, pinning the
    per-crop LF amplitude that the 1.78x gain overshoot violated.
    """
    result: Dict[str, torch.Tensor] = {}
    mask = valid.float()
    total: Optional[torch.Tensor] = None
    count = 0
    for window in windows:
        w = int(window)
        if w < 2:
            continue
        pred_ma, band_valid = _masked_moving_average_torch(prediction, mask, w)
        tgt_ma, _ = _masked_moving_average_torch(target, mask, w)
        band_mask = band_valid.float()
        if float(band_mask.sum().item()) < 1.0:
            continue
        error = F.smooth_l1_loss(
            pred_ma, tgt_ma, reduction="none", beta=max(float(beta), 1e-8)
        )
        band_loss = (error * band_mask).sum() / band_mask.sum().clamp_min(1.0)
        result[f"ma{w}"] = band_loss
        total = band_loss if total is None else total + band_loss
        count += 1
    if total is None or count == 0:
        return result
    result["mean"] = total / float(count)
    return result


def _masked_velocity_cosine_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Sign/direction aligned velocity cosine (BF1 event direction)."""
    if prediction.shape[1] < 2:
        return prediction.sum() * 0.0
    pair = (valid[:, 1:] & valid[:, :-1]).float()
    active = pair.sum(dim=1) > 1
    if not bool(active.any()):
        return prediction.sum() * 0.0
    pair = pair[active]
    pred = prediction[active]
    tgt = target[active]
    pred_v = (pred[:, 1:] - pred[:, :-1]) * pair
    tgt_v = (tgt[:, 1:] - tgt[:, :-1]) * pair
    dot = (pred_v * tgt_v).sum(dim=1)
    norms = pred_v.norm(dim=1) * tgt_v.norm(dim=1)
    usable = norms > 1e-8
    if not bool(usable.any()):
        return pred.sum() * 0.0
    cosine = dot[usable] / norms[usable].clamp_min(1e-8)
    return (1.0 - cosine).mean()


def _masked_reversal_event_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    min_speed: float,
    pos_weight: float,
    sharpness: float,
) -> torch.Tensor:
    """BF1-aligned bow-change event loss.

    A bow change (BF1 event) is a sign flip of the target distance velocity
    across two consecutive steps, with a dead-band on speed. The predicted
    event logit is derived from the predicted velocity sign agreement:

        logit_t = -sharpness * cos(v_pred[t], v_pred[t-1])

    so a confident sign flip in the prediction drives the logit positive while
    a confident sign continuation drives it negative. Positively weighted BCE
    compensates event rarity, mirroring how BF1 counts rare bow-change events.
    No extra model head is needed, so warm starting keeps working.
    """
    if prediction.shape[1] < 3:
        return prediction.sum() * 0.0
    pair = valid[:, 1:] & valid[:, :-1]
    triple = pair[:, 1:] & pair[:, :-1]
    active = triple.sum(dim=1) > 0
    if not bool(active.any()):
        return prediction.sum() * 0.0
    triple = triple[active]
    pred = prediction[active]
    tgt = target[active]
    pred_v = pred[:, 1:] - pred[:, :-1]
    tgt_v = tgt[:, 1:] - tgt[:, :-1]
    tgt_speed = tgt_v.abs()
    speed = max(float(min_speed), 0.0)
    events = (
        (tgt_v[:, 1:] * tgt_v[:, :-1] < 0)
        & (tgt_speed[:, 1:] > speed)
        & (tgt_speed[:, :-1] > speed)
        & triple
    )
    pv_prev = pred_v[:, :-1]
    pv_next = pred_v[:, 1:]
    sign_cos = (pv_next * pv_prev) / (
        pv_next.abs() * pv_prev.abs() + 1e-6
    )
    logits = -float(sharpness) * sign_cos
    labels = events.float()
    mask = triple.float()
    weight = labels * (max(float(pos_weight), 1.0) - 1.0) + 1.0
    per_pixel = F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    return (per_pixel * weight * mask).sum() / mask.sum().clamp_min(1.0)


def bow_root_bridge_auxiliary_losses(
    auxiliary: Optional[Mapping[str, torch.Tensor]],
    batch: Mapping[str, torch.Tensor],
    *,
    smooth_l1_beta: float,
    reversal_min_speed: float = 0.005,
    reversal_pos_weight: float = 20.0,
    reversal_sharpness: float = 8.0,
    multiscale_windows: Optional[Sequence[int]] = None,
) -> Dict[str, torch.Tensor]:
    """Supervise the BCS-aligned 1D bow-root-to-bridge profile.

    Two head modes are supported and detected from the auxiliary keys:

    * ``bow_root_bridge_distance`` (softplus head): prediction and target are
      the absolute frog-to-bridge distance; the cosine term subtracts the GT
      half bow, mirroring the deployed BCS.
    * ``bow_root_bridge_signed_offset`` (signed-offset head): prediction and
      target are the centered profile ``distance - half_bow`` directly, so
      every regression term (position, velocity) spends its budget on the
      variation BCS measures rather than on the per-take DC level.

    Pointwise smooth-L1 on position/velocity anchors absolute scale and bias;
    the sequence-level cosine terms align the prediction with the evaluation
    metrics: the profile cosine reproduces BCS (cosine of the
    distance-minus-half-bow profile) and the velocity cosine rewards the
    velocity sign agreement that BF1's bow-change events depend on. The
    reversal event BCE targets the rare bow-change frames directly, matching
    BF1's event counting. ``pooled_cosine`` reproduces the deployed metric's
    aggregation (one cosine over all valid frames of the batch) instead of
    the mean of per-window cosines.
    """
    reference = batch["relation"]
    zero = reference.sum() * 0.0
    result = {
        "position": zero,
        "velocity": zero,
        "cosine": zero,
        "velocity_cosine": zero,
        "pooled_cosine": zero,
        "reversal": zero,
        "multiscale": zero,
        "mae": zero.detach(),
    }
    if auxiliary is None or (
        "bow_root_bridge_distance" not in auxiliary
        and "bow_root_bridge_signed_offset" not in auxiliary
    ):
        return result
    signed_offset_mode = "bow_root_bridge_signed_offset" in auxiliary
    if signed_offset_mode:
        if "bow_root_bridge_distance" not in batch or "bow_root_bridge_half_bow" not in batch:
            raise KeyError(
                "The signed-offset scalar head is enabled but the dataset has no "
                "bow_root_bridge_distance / bow_root_bridge_half_bow targets. Set "
                "train_data.bow_root_bridge_distance_key and "
                "train_data.bow_root_bridge_half_bow_key."
            )
        prediction = auxiliary["bow_root_bridge_signed_offset"].float()
        target = (
            batch["bow_root_bridge_distance"].to(
                device=prediction.device, dtype=torch.float32
            )
            - batch["bow_root_bridge_half_bow"].to(
                device=prediction.device, dtype=torch.float32
            )
        )
    else:
        if "bow_root_bridge_distance" not in batch:
            raise KeyError(
                "Scalar head is enabled but the dataset has no "
                "bow_root_bridge_distance target. Regenerate playable_relations.npz "
                "and set train_data.bow_root_bridge_distance_key."
            )
        prediction = auxiliary["bow_root_bridge_distance"].float()
        target = batch["bow_root_bridge_distance"].to(
            device=prediction.device, dtype=torch.float32
        )
    valid = batch["bow_root_bridge_distance_valid"].to(
        device=prediction.device
    ).bool()
    if signed_offset_mode:
        half_bow = batch["bow_root_bridge_half_bow"].to(
            device=prediction.device, dtype=torch.float32
        )
        half_bow = torch.nan_to_num(half_bow, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        half_bow = batch.get("bow_root_bridge_half_bow")
        if half_bow is not None:
            half_bow = half_bow.to(device=prediction.device, dtype=torch.float32)
            half_bow = torch.nan_to_num(half_bow, nan=0.0, posinf=0.0, neginf=0.0)
    if valid.any():
        error = F.smooth_l1_loss(
            prediction,
            target,
            reduction="none",
            beta=max(float(smooth_l1_beta), 1e-8),
        )
        result["position"] = (error * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
        result["mae"] = (prediction.detach() - target).abs()[valid].mean()
        result["cosine"] = _masked_profile_cosine_loss(
            prediction,
            target,
            valid,
            half_bow if not signed_offset_mode else None,
            centered=signed_offset_mode,
        )
        result["pooled_cosine"] = _masked_pooled_cosine_loss(
            prediction,
            target,
            valid,
            None if signed_offset_mode else half_bow,
        )
        if multiscale_windows:
            bands = _masked_multiscale_ma_loss(
                prediction,
                target,
                valid,
                [int(w) for w in multiscale_windows],
                beta=float(smooth_l1_beta),
            )
            result.update(bands)
            result["multiscale"] = bands.get("mean", zero)

    if prediction.shape[1] >= 2:
        pair_valid = valid[:, 1:] & valid[:, :-1]
        if pair_valid.any():
            pred_velocity = prediction[:, 1:] - prediction[:, :-1]
            target_velocity = target[:, 1:] - target[:, :-1]
            velocity_error = F.smooth_l1_loss(
                pred_velocity,
                target_velocity,
                reduction="none",
                beta=max(float(smooth_l1_beta), 1e-8),
            )
            result["velocity"] = (
                velocity_error * pair_valid.float()
            ).sum() / pair_valid.float().sum().clamp_min(1.0)
            result["velocity_cosine"] = _masked_velocity_cosine_loss(
                prediction, target, valid
            )
            result["reversal"] = _masked_reversal_event_loss(
                prediction,
                target,
                valid,
                min_speed=float(reversal_min_speed),
                pos_weight=float(reversal_pos_weight),
                sharpness=float(reversal_sharpness),
            )
    return result


@torch.inference_mode()
def _calibrate_signed_offset_bias(
    model,
    dataloader,
    device: torch.device,
    max_batches: int = 8,
) -> float:
    """Empirically center a warm-started signed-offset scalar head.

    A checkpoint trained in softplus/distance mode drives the head to output
    values near the absolute distance, while the signed-offset target is
    centered near zero.  This shifts the final bias of the scalar head by the
    observed mean gap between prediction and centered target, so training
    starts at the correct DC level and the short fast-loop budget is spent on
    the BCS-relevant variation rather than on removing a constant offset.
    Returns the applied bias shift.
    """
    was_training = model.training
    model.eval()
    prediction_sum = 0.0
    target_sum = 0.0
    count = 0
    batches = 0
    for batch in dataloader:
        if batches >= int(max_batches):
            break
        mel = batch["mel"].to(device=device, dtype=torch.float32)
        onset = batch["onset"].to(device=device, dtype=torch.float32)
        pitch = batch["pitch"].to(device=device, dtype=torch.float32)
        _, _, auxiliary = model.forward_with_auxiliary(
            mel,
            onset,
            pitch,
            optional_pretrained_audio(batch, device=device),
            optional_instrument_id(batch, device=device),
        )
        prediction = auxiliary["bow_root_bridge_signed_offset"].float()
        target = batch["bow_root_bridge_distance"].to(
            device=device, dtype=torch.float32
        )
        half_bow = batch["bow_root_bridge_half_bow"].to(
            device=device, dtype=torch.float32
        )
        valid = (
            batch["bow_root_bridge_distance_valid"]
            .to(device=device)
            .bool()
            & torch.isfinite(target)
            & torch.isfinite(half_bow)
        )
        if valid.any():
            prediction_sum += float(prediction[valid].double().sum().item())
            target_sum += float((target - half_bow)[valid].double().sum().item())
            count += int(valid.sum().item())
        batches += 1
    if was_training:
        model.train()
    if count == 0:
        raise ValueError(
            "Signed-offset bias calibration found no valid scalar frames. Check "
            "bow_root_bridge_distance_key / bow_root_bridge_half_bow_key."
        )
    shift = (prediction_sum - target_sum) / count
    final_layer = model.bow_root_bridge_head[-1]
    final_layer.bias.data.add_(-shift)
    return shift


def _checkpoint_payload(
    accelerator: Accelerator,
    model,
    *,
    relation_dim: int,
    mel_dim: int,
    onset_dim: int = 1,
    global_step: int,
    predict_validity: bool,
    use_positional_encoding: bool,
    pretrained_audio_dim: int,
    pretrained_audio_hidden_dim: int,
    num_instruments: int,
    instrument_embedding_dim: int,
    latent_dim: int,
    dropout: float,
    mapping_model_type: str,
    structured_tcn_layers: int,
    structured_tcn_kernel_size: int,
    structured_gru_layers: int,
    structured_bow_step_scale: float,
    bow_reversal_sharpen_tau: float,
    predict_bow_root_bridge_distance: bool,
    bow_root_bridge_head_mode: str,
    validity_bce_weight: float,
    continuous_feature_weights,
    contact_class_weights,
    bow_class_weights,
    validity_pos_weight,
    temporal_config: Mapping[str, Any],
    train_mode: Any,
    selection: str,
) -> Dict[str, Any]:
    return {
        "model": accelerator.unwrap_model(model).state_dict(),
        "relation_dim": int(relation_dim),
        "mel_dim": int(mel_dim),
        "onset_dim": int(onset_dim),
        "pretrained_audio_dim": int(pretrained_audio_dim),
        "pretrained_audio_hidden_dim": int(pretrained_audio_hidden_dim),
        "num_instruments": int(num_instruments),
        "instrument_embedding_dim": int(instrument_embedding_dim),
        "global_step": int(global_step),
        "predict_validity": bool(predict_validity),
        "use_positional_encoding": bool(use_positional_encoding),
        "latent_dim": int(latent_dim),
        "dropout": float(dropout),
        "mapping_model_type": str(mapping_model_type),
        "structured_tcn_layers": int(structured_tcn_layers),
        "structured_tcn_kernel_size": int(structured_tcn_kernel_size),
        "structured_gru_layers": int(structured_gru_layers),
        "structured_bow_step_scale": float(structured_bow_step_scale),
        "bow_reversal_sharpen_tau": float(bow_reversal_sharpen_tau),
        "predict_bow_root_bridge_distance": bool(
            predict_bow_root_bridge_distance
        ),
        "bow_root_bridge_head_mode": str(bow_root_bridge_head_mode),
        "categorical_groups": [],
        "validity_bce_weight": float(validity_bce_weight),
        "continuous_feature_weights": continuous_feature_weights,
        "contact_class_weights": contact_class_weights.tolist(),
        "bow_class_weights": bow_class_weights.tolist(),
        "validity_pos_weight": validity_pos_weight.tolist() if validity_pos_weight is not None else None,
        "temporal_training": dict(temporal_config),
        "train_mode": train_mode,
        "checkpoint_selection": str(selection),
    }


def main(
    output_dir: str,
    train_data: Dict,
    validation_data: Optional[Dict] = None,
    initial_checkpoint_path: Optional[str] = None,
    train_batch_size: int = 32,
    max_train_steps: int = 30000,
    learning_rate: float = 1e-4,
    latent_dim: int = 256,
    mapping_model_type: str = "transformer",
    structured_tcn_layers: int = 4,
    structured_tcn_kernel_size: int = 5,
    structured_gru_layers: int = 2,
    structured_bow_step_scale: float = 0.1,
    bow_reversal_sharpen_tau: float = 0.0,
    predict_bow_root_bridge_distance: bool = False,
    bow_root_bridge_head_mode: str = "softplus",
    bow_root_bridge_distance_loss_weight: float = 0.0,
    bow_root_bridge_velocity_loss_weight: float = 0.0,
    bow_root_bridge_cosine_loss_weight: float = 0.0,
    bow_root_bridge_velocity_cosine_loss_weight: float = 0.0,
    bow_root_bridge_pooled_cosine_loss_weight: float = 0.0,
    bow_root_bridge_multiscale_loss_weight: float = 0.0,
    bow_root_bridge_multiscale_windows: Optional[Sequence[int]] = None,
    bow_root_bridge_reversal_loss_weight: float = 0.0,
    bow_root_bridge_reversal_pos_weight: float = 20.0,
    bow_root_bridge_reversal_min_speed: float = 0.005,
    bow_root_bridge_reversal_sharpness: float = 8.0,
    bow_root_bridge_smooth_l1_beta: float = 0.02,
    bow_root_bridge_signed_offset_bias_calibration: bool = False,
    seed: int = 2026,
    save_steps: int = 2000,
    num_workers: int = 4,
    mse_weight: float = 1.0,
    l1_weight: float = 1.0,
    contact_ce_weight: float = 1.0,
    bow_ce_weight: float = 1.0,
    validity_bce_weight: float = 0.5,
    predict_validity: bool = True,
    use_positional_encoding: bool = True,
    pretrained_audio_hidden_dim: int = 128,
    num_instruments: int = 0,
    instrument_embedding_dim: int = 16,
    dropout: float = 0.1,
    max_grad_norm: float = 1.0,
    warmup_steps: int = 1000,
    min_lr_ratio: float = 0.1,
    class_weight_max: float = 4.0,
    validity_pos_weight_max: float = 20.0,
    continuous_feature_weights=None,
    temporal_feature_weights=None,
    velocity_loss_weight: float = 1.0,
    acceleration_loss_weight: float = 0.25,
    bow_t_direction_loss_weight: float = 0.2,
    bow_t_event_margin_loss_weight: float = 0.0,
    bow_t_static_loss_weight: float = 0.0,
    bow_t_event_margin: float = 0.005,
    bow_t_reversal_loss_weight: float = 0.0,
    bow_t_reversal_event_weight: float = 8.0,
    bow_t_reversal_consistency_loss_weight: float = 0.0,
    bow_t_reversal_positive_weight: float = 1.0,
    bow_t_ignore_string_transitions: bool = False,
    bow_mid_direction_loss_weight: float = 0.1,
    bow_transition_ce_loss_weight: float = 0.25,
    temporal_smooth_l1_beta: float = 0.02,
    bow_t_direction_temperature: float = 0.05,
    bow_t_min_velocity: float = 0.005,
    bow_mid_min_speed: float = 0.005,
    structured_direction_loss_weight: float = 0.0,
    structured_reversal_loss_weight: float = 0.0,
    structured_bow_step_loss_weight: float = 0.0,
    structured_direction_class_weights: Optional[Sequence[float]] = None,
    structured_reversal_pos_weight: float = 4.0,
    structured_bow_step_beta: float = 0.05,
    structured_bow_step_target_max: float = 1.0,
) -> None:
    *_, resolved_config = inspect.getargvalues(inspect.currentframe())
    if validation_data is not None:
        raise ValueError(
            "This train+val entry point intentionally has no validation_data. "
            "Set train_data.mode='train+val' and validation_data=null; evaluate only on test after training."
        )
    if continuous_feature_weights is not None:
        continuous_feature_weights = [float(value) for value in continuous_feature_weights]
    if temporal_feature_weights is not None:
        temporal_feature_weights = [float(value) for value in temporal_feature_weights]
    if structured_direction_class_weights is not None:
        structured_direction_class_weights = [
            float(value) for value in structured_direction_class_weights
        ]
    if bow_root_bridge_multiscale_windows is not None:
        bow_root_bridge_multiscale_windows = [
            int(value) for value in bow_root_bridge_multiscale_windows
        ]
    else:
        bow_root_bridge_multiscale_windows = []
    mapping_model_type = normalize_mapping_model_type(mapping_model_type)

    accelerator = Accelerator()
    set_seed(int(seed))
    output_path = Path(output_dir)
    if accelerator.is_main_process:
        output_path.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create(resolved_config), str(output_path / "config.yaml"))

    workers = int(train_data.get("num_workers", num_workers))
    train_dataset = SPDCanonicalRelationsDataset(**train_data)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=int(train_batch_size),
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    sample = train_dataset[0]
    mel_dim = int(sample["mel"].shape[-1])
    onset_dim = infer_onset_dim(sample)
    pretrained_audio_dim = infer_pretrained_audio_dim(sample)
    relation_dim = int(sample["relation"].shape[-1])
    if relation_dim != len(RELATION_FEATURE_NAMES):
        raise ValueError(
            f"Temporal SPD training expects {len(RELATION_FEATURE_NAMES)} relation features, "
            f"got {relation_dim}."
        )

    model = create_relation_predictor(
        model_type=mapping_model_type,
        mel_dim=mel_dim,
        relation_dim=relation_dim,
        latent_dim=int(latent_dim),
        dropout=float(dropout),
        predict_validity=bool(predict_validity),
        use_positional_encoding=bool(use_positional_encoding),
        pretrained_audio_dim=pretrained_audio_dim,
        pretrained_audio_hidden_dim=int(pretrained_audio_hidden_dim),
        num_instruments=int(num_instruments),
        instrument_embedding_dim=int(instrument_embedding_dim),
        structured_tcn_layers=int(structured_tcn_layers),
        structured_tcn_kernel_size=int(structured_tcn_kernel_size),
        structured_gru_layers=int(structured_gru_layers),
        structured_bow_step_scale=float(structured_bow_step_scale),
        bow_reversal_sharpen_tau=float(bow_reversal_sharpen_tau),
        predict_bow_root_bridge_distance=bool(
            predict_bow_root_bridge_distance
        ),
        bow_root_bridge_head_mode=bow_root_bridge_head_mode,
        onset_dim=int(onset_dim),
    )
    if initial_checkpoint_path:
        checkpoint_path = Path(initial_checkpoint_path).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Initial mapping checkpoint does not exist: {checkpoint_path}")
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        if "model" not in checkpoint:
            raise KeyError(f"Initial checkpoint {checkpoint_path} does not contain 'model'.")
        # Zero-pad the onset encoder when this run adds bow-dynamics channels:
        # the warm-started model then reproduces the checkpoint exactly until
        # the new channels receive gradient.
        warm_state = expand_onset_encoder_state(checkpoint["model"], int(onset_dim))
        if warm_state is not checkpoint["model"]:
            checkpoint = {**checkpoint, "model": warm_state}
            logger.info(
                "Expanded onset_encoder input channels to %d (new channels zero-init).",
                int(onset_dim),
            )
        checkpoint_model_type = normalize_mapping_model_type(
            checkpoint.get("mapping_model_type", "transformer")
        )
        if checkpoint_model_type != mapping_model_type:
            raise ValueError(
                f"Cannot warm-start mapping_model_type={mapping_model_type} from "
                f"a {checkpoint_model_type} checkpoint."
            )
        try:
            model.load_state_dict(checkpoint["model"])
            logger.info(f"Warm-started mapping weights from {checkpoint_path}")
            checkpoint_head_mode = normalize_bow_root_bridge_head_mode(
                checkpoint.get("bow_root_bridge_head_mode", "softplus")
            )
            if (
                bow_root_bridge_head_mode == "signed_offset"
                and checkpoint_head_mode != "signed_offset"
            ):
                logger.info(
                    "Scalar head mode changed %s -> signed_offset; predictions start "
                    "centered on the distance level and will be DC-calibrated before "
                    "training (bow_root_bridge_signed_offset_bias_calibration=%s).",
                    checkpoint_head_mode,
                    bool(bow_root_bridge_signed_offset_bias_calibration),
                )
        except RuntimeError:
            if (
                bool(predict_bow_root_bridge_distance)
                and not bool(checkpoint.get("predict_bow_root_bridge_distance", False))
            ):
                incompatible = model.load_state_dict(checkpoint["model"], strict=False)
                allowed_missing = {
                    key
                    for key in model.state_dict()
                    if key.startswith("bow_root_bridge_head.")
                }
                actual_missing = set(incompatible.missing_keys)
                if incompatible.unexpected_keys or actual_missing != allowed_missing:
                    raise RuntimeError(
                        "Warm-start mismatch was not limited to the new scalar head: "
                        f"missing={sorted(actual_missing)}, "
                        f"unexpected={sorted(incompatible.unexpected_keys)}"
                    )
                logger.info(
                    "Warm-started mapping weights from %s; initialized the new "
                    "bow_root_bridge_head from scratch.",
                    checkpoint_path,
                )
            else:
                checkpoint_pretrained_dim = int(checkpoint.get("pretrained_audio_dim", 0))
                if pretrained_audio_dim <= 0 or checkpoint_pretrained_dim > 0:
                    raise
                current_state = model.state_dict()
                compatible_state = {
                    key: value
                    for key, value in checkpoint["model"].items()
                    if key in current_state and tuple(current_state[key].shape) == tuple(value.shape)
                }
                model.load_state_dict(compatible_state, strict=False)
                logger.warning(
                    "Warm-started %d compatible tensors from %s. The MERT projection and "
                    "audio fusion layer were reinitialized because the checkpoint was trained "
                    "without pretrained_audio.",
                    len(compatible_state),
                    checkpoint_path,
                )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < int(warmup_steps):
            return max(float(step + 1) / float(warmup_steps), 1e-3)
        decay_steps = max(1, int(max_train_steps) - int(warmup_steps))
        progress = min(1.0, max(0.0, (step - int(warmup_steps)) / decay_steps))
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return float(min_lr_ratio) + (1.0 - float(min_lr_ratio)) * float(cosine)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    contact_weights, bow_weights, contact_counts, bow_counts = compute_spd_relation_class_weights(
        train_dataset,
        max_weight=float(class_weight_max),
    )
    validity_pos_weight = compute_spd_relation_validity_pos_weight(
        train_dataset,
        max_weight=float(validity_pos_weight_max),
    )

    model, optimizer, scheduler, train_dataloader = accelerator.prepare(
        model,
        optimizer,
        scheduler,
        train_dataloader,
    )
    if (
        bool(bow_root_bridge_signed_offset_bias_calibration)
        and bool(predict_bow_root_bridge_distance)
        and bow_root_bridge_head_mode == "signed_offset"
        and initial_checkpoint_path
    ):
        shift = _calibrate_signed_offset_bias(
            accelerator.unwrap_model(model),
            train_dataloader,
            accelerator.device,
        )
        logger.info(
            "Signed-offset bias calibration: shifted the scalar head bias by "
            "%.6f m so warm-started predictions start at the centered target "
            "level.",
            -shift,
        )
    pointwise_kwargs = {
        "mse_weight": float(mse_weight),
        "l1_weight": float(l1_weight),
        "contact_ce_weight": float(contact_ce_weight),
        "bow_ce_weight": float(bow_ce_weight),
        "validity_bce_weight": float(validity_bce_weight),
        "contact_class_weights": contact_weights,
        "bow_class_weights": bow_weights,
        "continuous_feature_weights": continuous_feature_weights,
        "validity_pos_weight": validity_pos_weight,
    }
    temporal_config = {
        "temporal_feature_weights": temporal_feature_weights,
        "velocity_loss_weight": float(velocity_loss_weight),
        "acceleration_loss_weight": float(acceleration_loss_weight),
        "bow_t_direction_loss_weight": float(bow_t_direction_loss_weight),
        "bow_t_event_margin_loss_weight": float(bow_t_event_margin_loss_weight),
        "bow_t_static_loss_weight": float(bow_t_static_loss_weight),
        "bow_t_event_margin": float(bow_t_event_margin),
        "bow_t_reversal_loss_weight": float(bow_t_reversal_loss_weight),
        "bow_t_reversal_event_weight": float(bow_t_reversal_event_weight),
        "bow_t_reversal_consistency_loss_weight": float(
            bow_t_reversal_consistency_loss_weight
        ),
        "bow_t_reversal_positive_weight": float(bow_t_reversal_positive_weight),
        "bow_t_ignore_string_transitions": bool(bow_t_ignore_string_transitions),
        "bow_mid_direction_loss_weight": float(bow_mid_direction_loss_weight),
        "bow_transition_ce_loss_weight": float(bow_transition_ce_loss_weight),
        "temporal_smooth_l1_beta": float(temporal_smooth_l1_beta),
        "bow_t_direction_temperature": float(bow_t_direction_temperature),
        "bow_t_min_velocity": float(bow_t_min_velocity),
        "bow_mid_min_speed": float(bow_mid_min_speed),
        "structured_direction_loss_weight": float(structured_direction_loss_weight),
        "structured_reversal_loss_weight": float(structured_reversal_loss_weight),
        "structured_bow_step_loss_weight": float(structured_bow_step_loss_weight),
        "structured_direction_class_weights": structured_direction_class_weights,
        "structured_reversal_pos_weight": float(structured_reversal_pos_weight),
        "structured_bow_step_beta": float(structured_bow_step_beta),
        "structured_bow_step_target_max": float(structured_bow_step_target_max),
        "predict_bow_root_bridge_distance": bool(
            predict_bow_root_bridge_distance
        ),
        "bow_root_bridge_distance_loss_weight": float(
            bow_root_bridge_distance_loss_weight
        ),
        "bow_root_bridge_velocity_loss_weight": float(
            bow_root_bridge_velocity_loss_weight
        ),
        "bow_root_bridge_cosine_loss_weight": float(
            bow_root_bridge_cosine_loss_weight
        ),
        "bow_root_bridge_velocity_cosine_loss_weight": float(
            bow_root_bridge_velocity_cosine_loss_weight
        ),
        "bow_root_bridge_pooled_cosine_loss_weight": float(
            bow_root_bridge_pooled_cosine_loss_weight
        ),
        "bow_root_bridge_multiscale_loss_weight": float(
            bow_root_bridge_multiscale_loss_weight
        ),
        "bow_root_bridge_multiscale_windows": list(
            bow_root_bridge_multiscale_windows
        ),
        "bow_root_bridge_reversal_loss_weight": float(
            bow_root_bridge_reversal_loss_weight
        ),
        "bow_root_bridge_reversal_pos_weight": float(
            bow_root_bridge_reversal_pos_weight
        ),
        "bow_root_bridge_reversal_min_speed": float(
            bow_root_bridge_reversal_min_speed
        ),
        "bow_root_bridge_reversal_sharpness": float(
            bow_root_bridge_reversal_sharpness
        ),
        "bow_root_bridge_smooth_l1_beta": float(
            bow_root_bridge_smooth_l1_beta
        ),
    }

    logger.info("***** Running temporal SPD relation training *****")
    logger.info(f"  Train split = {train_data.get('mode', 'train')}")
    logger.info(f"  Num records = {len(train_dataset.records)}")
    logger.info(f"  Num windows = {len(train_dataset)}")
    logger.info(f"  Initial checkpoint = {initial_checkpoint_path or 'none (from scratch)'}")
    logger.info(f"  Mapping model type = {mapping_model_type}")
    logger.info(f"  Use positional encoding = {bool(use_positional_encoding)}")
    logger.info(f"  Pretrained audio dim = {pretrained_audio_dim}")
    logger.info(f"  Instrument classes = {int(num_instruments)}")
    logger.info(f"  Contact counts = {contact_counts.tolist()}")
    logger.info(f"  Bow counts = {bow_counts.tolist()}")
    logger.info(f"  Temporal loss config = {temporal_config}")

    global_step = 0
    progress = tqdm(range(int(max_train_steps)), disable=not accelerator.is_local_main_process)
    while global_step < int(max_train_steps):
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                structured_auxiliary = None
                if mapping_model_type == "structured_tcn_bigru":
                    pred, validity_logits, structured_auxiliary = model.forward_with_auxiliary(
                        batch["mel"].float(),
                        batch["onset"].float(),
                        batch["pitch"].float(),
                        optional_pretrained_audio(batch),
                        optional_instrument_id(batch),
                    )
                elif predict_validity:
                    pred, validity_logits = model.forward_with_validity(
                        batch["mel"].float(),
                        batch["onset"].float(),
                        batch["pitch"].float(),
                        optional_pretrained_audio(batch),
                        optional_instrument_id(batch),
                    )
                else:
                    pred = model(
                        batch["mel"].float(),
                        batch["onset"].float(),
                        batch["pitch"].float(),
                        optional_pretrained_audio(batch),
                        optional_instrument_id(batch),
                    )
                    validity_logits = None

                target = batch["relation"].float()
                relation_mask = batch["relation_mask"].float()
                contact_valid = batch["contact_valid"].bool()
                bow_valid = batch["bow_valid"].bool()
                pointwise_loss, pointwise_stats = relation_prediction_loss(
                    pred,
                    target,
                    relation_mask,
                    contact_valid,
                    bow_valid,
                    validity_logits=validity_logits,
                    **pointwise_kwargs,
                )
                temporal = temporal_relation_losses(
                    pred,
                    target,
                    relation_mask,
                    bow_valid,
                    temporal_feature_weights=temporal_feature_weights,
                    smooth_l1_beta=float(temporal_smooth_l1_beta),
                    bow_t_direction_temperature=float(bow_t_direction_temperature),
                    bow_t_min_velocity=float(bow_t_min_velocity),
                    bow_t_event_margin=float(bow_t_event_margin),
                    bow_t_reversal_event_weight=float(bow_t_reversal_event_weight),
                    bow_t_reversal_positive_weight=float(bow_t_reversal_positive_weight),
                    bow_t_ignore_string_transitions=bool(bow_t_ignore_string_transitions),
                    bow_mid_min_speed=float(bow_mid_min_speed),
                )
                structured = structured_bow_auxiliary_losses(
                    structured_auxiliary,
                    target,
                    relation_mask,
                    bow_valid,
                    bow_t_min_velocity=float(bow_t_min_velocity),
                    bow_t_ignore_string_transitions=bool(bow_t_ignore_string_transitions),
                    direction_class_weights=structured_direction_class_weights,
                    reversal_pos_weight=float(structured_reversal_pos_weight),
                    bow_step_beta=float(structured_bow_step_beta),
                    bow_step_target_max=float(structured_bow_step_target_max),
                )
                bridge_scalar = bow_root_bridge_auxiliary_losses(
                    structured_auxiliary,
                    batch,
                    smooth_l1_beta=float(bow_root_bridge_smooth_l1_beta),
                    reversal_min_speed=float(bow_root_bridge_reversal_min_speed),
                    reversal_pos_weight=float(bow_root_bridge_reversal_pos_weight),
                    reversal_sharpness=float(bow_root_bridge_reversal_sharpness),
                    multiscale_windows=bow_root_bridge_multiscale_windows,
                )
                loss = pointwise_loss
                loss = loss + float(velocity_loss_weight) * temporal["velocity"]
                loss = loss + float(acceleration_loss_weight) * temporal["acceleration"]
                loss = loss + float(bow_t_direction_loss_weight) * temporal["bow_t_direction"]
                loss = loss + float(bow_t_event_margin_loss_weight) * temporal[
                    "bow_t_event_margin"
                ]
                loss = loss + float(bow_t_static_loss_weight) * temporal[
                    "bow_t_static_velocity"
                ]
                loss = loss + float(bow_t_reversal_loss_weight) * temporal["bow_t_reversal"]
                loss = loss + float(bow_t_reversal_consistency_loss_weight) * temporal[
                    "bow_t_reversal_consistency"
                ]
                loss = loss + float(bow_mid_direction_loss_weight) * temporal["bow_mid_direction"]
                loss = loss + float(bow_transition_ce_loss_weight) * temporal["bow_transition_ce"]
                loss = loss + float(structured_direction_loss_weight) * structured["direction"]
                loss = loss + float(structured_reversal_loss_weight) * structured["reversal"]
                loss = loss + float(structured_bow_step_loss_weight) * structured["bow_step"]
                loss = loss + float(
                    bow_root_bridge_distance_loss_weight
                ) * bridge_scalar["position"]
                loss = loss + float(
                    bow_root_bridge_velocity_loss_weight
                ) * bridge_scalar["velocity"]
                loss = loss + float(
                    bow_root_bridge_cosine_loss_weight
                ) * bridge_scalar["cosine"]
                loss = loss + float(
                    bow_root_bridge_velocity_cosine_loss_weight
                ) * bridge_scalar["velocity_cosine"]
                loss = loss + float(
                    bow_root_bridge_pooled_cosine_loss_weight
                ) * bridge_scalar["pooled_cosine"]
                loss = loss + float(
                    bow_root_bridge_multiscale_loss_weight
                ) * bridge_scalar["multiscale"]
                loss = loss + float(
                    bow_root_bridge_reversal_loss_weight
                ) * bridge_scalar["reversal"]

                accelerator.backward(loss)
                if accelerator.sync_gradients and max_grad_norm and max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), float(max_grad_norm))
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                scheduler.step()
                global_step += 1
                progress.update(1)
                logs = {
                    "loss": float(loss.detach().item()),
                    "base": float(pointwise_loss.detach().item()),
                    "vel": float(temporal["velocity"].detach().item()),
                    "acc": float(temporal["acceleration"].detach().item()),
                    "bt_dir": float(temporal["bow_t_direction"].detach().item()),
                    "bt_margin": float(temporal["bow_t_event_margin"].detach().item()),
                    "bt_static": float(temporal["bow_t_static_velocity"].detach().item()),
                    "bt_rev": float(temporal["bow_t_reversal"].detach().item()),
                    "bt_rev_cons": float(
                        temporal["bow_t_reversal_consistency"].detach().item()
                    ),
                    "bm_dir": float(temporal["bow_mid_direction"].detach().item()),
                    "btrans": float(temporal["bow_transition_ce"].detach().item()),
                    "sdir": float(structured["direction"].detach().item()),
                    "srev": float(structured["reversal"].detach().item()),
                    "sstep": float(structured["bow_step"].detach().item()),
                    "bridge_pos": float(bridge_scalar["position"].detach().item()),
                    "bridge_vel": float(bridge_scalar["velocity"].detach().item()),
                    "bridge_cos": float(bridge_scalar["cosine"].detach().item()),
                    "bridge_pool_cos": float(
                        bridge_scalar["pooled_cosine"].detach().item()
                    ),
                    "bridge_ms": float(bridge_scalar["multiscale"].detach().item()),
                    "bridge_vel_cos": float(
                        bridge_scalar["velocity_cosine"].detach().item()
                    ),
                    "bridge_rev": float(bridge_scalar["reversal"].detach().item()),
                    "bridge_mae": float(bridge_scalar["mae"].detach().item()),
                    "bow_acc": float(pointwise_stats["bow_acc"].item()),
                    "lr": float(scheduler.get_last_lr()[0]),
                }
                progress.set_postfix(**logs)
                if accelerator.is_main_process and global_step % 100 == 0:
                    logger.info(
                        "step %d loss=%.6f base=%.6f vel=%.6f acc=%.6f "
                        "bow_t_dir=%.6f bow_t_margin=%.6f bow_t_static=%.6f "
                        "bow_t_reversal=%.6f bow_t_reversal_cons=%.6f "
                        "bow_mid_dir=%.6f bow_transition=%.6f "
                        "structured_dir=%.6f structured_rev=%.6f structured_step=%.6f "
                        "contact_acc=%.4f bow_acc=%.4f lr=%.3e",
                        global_step,
                        logs["loss"],
                        logs["base"],
                        logs["vel"],
                        logs["acc"],
                        logs["bt_dir"],
                        logs["bt_margin"],
                        logs["bt_static"],
                        logs["bt_rev"],
                        logs["bt_rev_cons"],
                        logs["bm_dir"],
                        logs["btrans"],
                        logs["sdir"],
                        logs["srev"],
                        logs["sstep"],
                        float(pointwise_stats["contact_acc"].item()),
                        logs["bow_acc"],
                        logs["lr"],
                    )

                if save_steps > 0 and global_step % int(save_steps) == 0:
                    payload = _checkpoint_payload(
                        accelerator,
                        model,
                        relation_dim=relation_dim,
                        mel_dim=mel_dim,
                        onset_dim=onset_dim,
                        global_step=global_step,
                        predict_validity=bool(predict_validity),
                        use_positional_encoding=bool(use_positional_encoding),
                        pretrained_audio_dim=pretrained_audio_dim,
                        pretrained_audio_hidden_dim=int(pretrained_audio_hidden_dim),
                        num_instruments=int(num_instruments),
                        instrument_embedding_dim=int(instrument_embedding_dim),
                        latent_dim=int(latent_dim),
                        dropout=float(dropout),
                        mapping_model_type=mapping_model_type,
                        structured_tcn_layers=int(structured_tcn_layers),
                        structured_tcn_kernel_size=int(structured_tcn_kernel_size),
                        structured_gru_layers=int(structured_gru_layers),
                        structured_bow_step_scale=float(structured_bow_step_scale),
                        bow_reversal_sharpen_tau=float(bow_reversal_sharpen_tau),
                        predict_bow_root_bridge_distance=bool(
                            predict_bow_root_bridge_distance
                        ),
                        bow_root_bridge_head_mode=bow_root_bridge_head_mode,
                        validity_bce_weight=float(validity_bce_weight),
                        continuous_feature_weights=continuous_feature_weights,
                        contact_class_weights=contact_weights,
                        bow_class_weights=bow_weights,
                        validity_pos_weight=validity_pos_weight,
                        temporal_config=temporal_config,
                        train_mode=train_data.get("mode", "train"),
                        selection="periodic_fixed_step",
                    )
                    accelerator.save(payload, str(output_path / f"checkpoint-{global_step}.pt"))

            if global_step >= int(max_train_steps):
                break

    accelerator.wait_for_everyone()
    if hasattr(accelerator, "trackers"):
        accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Temporal Stage-1 training YAML.",
    )
    args = parser.parse_args()
    main(**OmegaConf.load(args.config))
