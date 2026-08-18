#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pinned GPU-free attention replacement for the normalized Evo-1 oracle.

This module intentionally implements only the exact stateless attention topology
used by ``togethercomputer/evo-1-131k-base`` at revision
``c206aab77ae5967a069c4200ecb1858588528c9d``.  It preserves the upstream
``MHA`` parameter names while replacing its CUDA-only kernel with dense CPU
PyTorch operations.  It is oracle code and must never import evo.cpp runtime
math.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class PortableAttentionError(RuntimeError):
    """Raised when a caller leaves the frozen Evo-1 attention contract."""


def _require_f32(tensor: torch.Tensor, label: str) -> None:
    if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
        raise PortableAttentionError(f"{label} must be CPU float32")


class RotaryEmbedding(nn.Module):
    """Small API-compatible subset of flash-attn 2.5.5 RotaryEmbedding."""

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        interleaved: bool = False,
        scale_base: Optional[float] = None,
        pos_idx_in_fp32: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if dim <= 0 or dim % 2 != 0 or interleaved or scale_base is not None:
            raise PortableAttentionError("unsupported rotary topology")
        if not pos_idx_in_fp32:
            raise PortableAttentionError("rotary positions must be float32")
        self.dim = dim
        self.base = float(base)
        self.interleaved = False
        self.scale_base = None
        self.pos_idx_in_fp32 = True
        self.register_buffer(
            "inv_freq", self._compute_inv_freq(device=device), persistent=False
        )
        self.register_buffer("scale", None, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None
        self._cos_k_cached: Optional[torch.Tensor] = None
        self._sin_k_cached: Optional[torch.Tensor] = None

    def _compute_inv_freq(
        self, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        positions = torch.arange(
            0, self.dim, 2, device=device, dtype=torch.float32
        )
        return 1.0 / (self.base ** (positions / self.dim))

    def _update_cos_sin_cache(
        self,
        seqlen: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if dtype != torch.float32:
            raise PortableAttentionError("rotary cache must be float32")
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
        ):
            self._seq_len_cached = seqlen
            positions = torch.arange(
                seqlen, device=device, dtype=torch.float32
            )
            inv_freq = (
                self._compute_inv_freq(device=device)
                if self.inv_freq.dtype != torch.float32
                else self.inv_freq
            )
            frequencies = torch.outer(positions, inv_freq)
            self._cos_cached = torch.cos(frequencies).to(dtype)
            self._sin_cached = torch.sin(frequencies).to(dtype)

    @staticmethod
    def _rotate(
        tensor: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor
    ) -> torch.Tensor:
        half = tensor.shape[-1] // 2
        first = tensor[..., :half]
        second = tensor[..., half:]
        return torch.cat(
            (first * cosine - second * sine, first * sine + second * cosine),
            dim=-1,
        )

    def forward(
        self,
        qkv: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        seqlen_offset: int = 0,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        _require_f32(qkv, "qkv")
        if kv is not None or seqlen_offset != 0 or max_seqlen is not None:
            raise PortableAttentionError("only frozen stateless self-attention is supported")
        if qkv.ndim != 5 or qkv.shape[2] != 3 or qkv.shape[-1] < self.dim:
            raise PortableAttentionError("qkv layout differs")
        self._update_cos_sin_cache(
            qkv.shape[1], device=qkv.device, dtype=qkv.dtype
        )
        assert self._cos_cached is not None and self._sin_cached is not None
        cosine = self._cos_cached[: qkv.shape[1]][None, :, None, :]
        sine = self._sin_cached[: qkv.shape[1]][None, :, None, :]
        rotary = qkv[..., : self.dim]
        tail = qkv[..., self.dim :]
        query = self._rotate(rotary[:, :, 0], cosine, sine)
        key = self._rotate(rotary[:, :, 1], cosine, sine)
        value = rotary[:, :, 2]
        rotated = torch.stack((query, key, value), dim=2)
        return torch.cat((rotated, tail), dim=-1) if tail.shape[-1] else rotated


class MHA(nn.Module):
    """Parameter-compatible, dense causal F32 attention for three Evo layers."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_heads_kv: Optional[int] = None,
        cross_attn: bool = False,
        qkv_proj_bias: bool = True,
        out_proj_bias: bool = True,
        dropout: float = 0.0,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        layer_idx: Optional[int] = None,
        dwconv: bool = False,
        rotary_emb_dim: int = 0,
        rotary_emb_base: float = 10000.0,
        rotary_emb_scale_base: Optional[float] = None,
        rotary_emb_interleaved: bool = False,
        use_alibi: bool = False,
        window_size: tuple[int, int] = (-1, -1),
        fused_bias_fc: bool = False,
        use_flash_attn: bool = False,
        return_residual: bool = False,
        checkpointing: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        heads_kv = num_heads if num_heads_kv is None else num_heads_kv
        if (
            embed_dim <= 0
            or num_heads <= 0
            or embed_dim % num_heads != 0
            or heads_kv != num_heads
            or cross_attn
            or not qkv_proj_bias
            or not out_proj_bias
            or dropout != 0.0
            or softmax_scale is not None
            or not causal
            or layer_idx not in (8, 16, 24)
            or dwconv
            or rotary_emb_dim != embed_dim // num_heads
            or float(rotary_emb_base) != 10000.0
            or rotary_emb_scale_base is not None
            or rotary_emb_interleaved
            or use_alibi
            or window_size != (-1, -1)
            or fused_bias_fc
            or not use_flash_attn
            or return_residual
            or checkpointing
        ):
            raise PortableAttentionError("MHA construction left the Evo-1 contract")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_heads_kv = heads_kv
        self.head_dim = embed_dim // num_heads
        self.cross_attn = False
        self.causal = True
        self.layer_idx = layer_idx
        self.dwconv = False
        self.rotary_emb_dim = rotary_emb_dim
        self.use_flash_attn = False
        self.return_residual = False
        self.checkpointing = False
        self.Wqkv = nn.Linear(
            embed_dim, 3 * embed_dim, bias=True, device=device, dtype=dtype
        )
        self.rotary_emb = RotaryEmbedding(
            rotary_emb_dim,
            base=rotary_emb_base,
            interleaved=False,
            scale_base=None,
            pos_idx_in_fp32=True,
            device=device,
        )
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=True, device=device, dtype=dtype
        )

    def forward(
        self,
        x: torch.Tensor,
        x_kv: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        mixer_subset: Optional[torch.Tensor] = None,
        inference_params: object = None,
        **kwargs: object,
    ) -> torch.Tensor:
        _require_f32(x, "attention input")
        _require_f32(self.Wqkv.weight, "Wqkv.weight")
        _require_f32(self.out_proj.weight, "out_proj.weight")
        if (
            x.ndim != 3
            or x.shape[-1] != self.embed_dim
            or x_kv is not None
            or key_padding_mask is not None
            or cu_seqlens is not None
            or max_seqlen is not None
            or mixer_subset is not None
            or inference_params is not None
            or kwargs
        ):
            raise PortableAttentionError("only frozen stateless dense attention is supported")
        batch, rows, _ = x.shape
        qkv = self.Wqkv(x).reshape(
            batch, rows, 3, self.num_heads, self.head_dim
        )
        qkv = self.rotary_emb(qkv)
        query, key, value = qkv.unbind(dim=2)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.einsum("bthd,bshd->bhts", query, key * scale)
        row_index = torch.arange(rows, device=x.device, dtype=torch.long)[:, None]
        column_index = torch.arange(rows, device=x.device, dtype=torch.long)
        scores = scores.masked_fill(column_index > row_index, -10000.0)
        attention = torch.softmax(scores, dim=-1, dtype=value.dtype)
        context = torch.einsum("bhts,bshd->bthd", attention, value)
        return self.out_proj(context.reshape(batch, rows, self.embed_dim))
