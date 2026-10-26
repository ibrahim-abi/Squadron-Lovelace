#!/usr/bin/env python3
"""
01_MultiHead_Attention_Core.py — Scaled Dot-Product & Multi-Head Self-Attention from Scratch
=============================================================================================

USSE Noosphere // Squadron Lovelace (Group 01) // Mission 3 (ps03) // Part 1
Station: Systems Architect (Lead Engineer) — Muhammad Ibrahim
Sector SEC-03-M45 (Pleiades Open Cluster) // AURA Subsystem 03: Distributed Multi-Head Attention Mesh

What this file provides (ps03 Part 1, items 1-4):
  1. MultiHeadSelfAttention(nn.Module) with learned W_Q, W_K, W_V in R^{d_model x d_model} and W_O,
     written from first principles: no nn.MultiheadAttention, no F.scaled_dot_product_attention.
  2. Multi-head reshaping (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim).
  3. Lower-triangular causal masking (-inf above the diagonal) for autoregressive decoding.
  4. An empirical variance-preservation test: for q, k with zero mean and unit variance,
     Var(q . k) ~= d_k, and Var(q . k / sqrt(d_k)) ~= 1.0 before the softmax.

Supporting work used by 04_Memory_Hierarchy_and_Safety_Audit.md:
  * flash_attention_tiled(): exact attention computed block by block with the online softmax of
    FlashAttention, plus an HBM read/write counter, so the IO derivation can be checked numerically.

Run:   python 01_MultiHead_Attention_Core.py          (CPU or CUDA, about 10 s)
Tests: python -m pytest                               (from the mission-03 folder)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

SEED = 2026


# ════════════════════════════════════════════════════════════════════════════
# 1-3. Multi-head self-attention from first principles
# ════════════════════════════════════════════════════════════════════════════


def causal_mask(seq_len: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Additive mask M with M[i, j] = -inf for j > i (future positions) and 0 elsewhere."""
    return torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype), diagonal=1)


def softmax(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax written out: exp(s - max) / sum(exp(s - max))."""
    shifted = scores - scores.amax(dim=dim, keepdim=True)
    e = torch.exp(shifted)
    return e / e.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + M) V.

    q: (..., N_q, d_k), k: (..., N_k, d_k), v: (..., N_k, d_v); mask is additive and broadcastable.
    Returns (output, attention weights).
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    weights = softmax(scores, dim=-1)
    return torch.matmul(weights, v), weights


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention, MHA(X) = Concat(head_1..head_h) W_O, head_i = Attention(X W_Q^i, X W_K^i, X W_V^i).

    The four projections are plain learned matrices (nn.Parameter), so every step of the computation is
    visible: project, split into heads, scale, mask, softmax, weight the values, merge heads, project out.
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.0, bias: bool = False) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model, self.num_heads = d_model, num_heads
        self.head_dim = d_model // num_heads
        # W_Q, W_K, W_V, W_O in R^{d_model x d_model}, Xavier-uniform as in Vaswani et al. (2017)
        self.W_Q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_K = nn.Parameter(torch.empty(d_model, d_model))
        self.W_V = nn.Parameter(torch.empty(d_model, d_model))
        self.W_O = nn.Parameter(torch.empty(d_model, d_model))
        self.b_O = nn.Parameter(torch.zeros(d_model)) if bias else None
        for w in (self.W_Q, self.W_K, self.W_V, self.W_O):
            nn.init.xavier_uniform_(w)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, d_model) -> (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)."""
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, num_heads, seq_len, head_dim) -> (batch, seq_len, d_model)."""
        b, h, n, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, n, h * d)

    def forward(self, x: torch.Tensor, causal: bool = True,
                padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (batch, seq_len, d_model). padding_mask: optional bool (batch, seq_len), True = real token.

        Returns (output (batch, seq_len, d_model), attention weights (batch, num_heads, seq_len, seq_len)).
        """
        b, n, _ = x.shape
        q = self.split_heads(x @ self.W_Q)
        k = self.split_heads(x @ self.W_K)
        v = self.split_heads(x @ self.W_V)
        mask = causal_mask(n, x.device, x.dtype) if causal else None
        if padding_mask is not None:
            pad = torch.zeros(b, 1, 1, n, device=x.device, dtype=x.dtype)
            pad = pad.masked_fill(~padding_mask[:, None, None, :], float("-inf"))
            mask = pad if mask is None else mask + pad
        context, weights = scaled_dot_product_attention(q, k, v, mask)
        context = self.dropout(context)
        out = self.merge_heads(context) @ self.W_O
        if self.b_O is not None:
            out = out + self.b_O
        return out, weights


# ════════════════════════════════════════════════════════════════════════════
# 4. Variance preservation
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class VarianceResult:
    d_k: int
    unscaled_var: float
    scaled_var: float
    unscaled_mean: float
    scaled_mean: float
    max_softmax_unscaled: float   # mean of the largest attention weight per row: saturation indicator
    max_softmax_scaled: float


def variance_preservation(d_k: int, n_samples: int = 200_000, seq_len: int = 128, seed: int = SEED) -> VarianceResult:
    """Draw q, k ~ N(0, I_dk). Then q . k = sum_i q_i k_i has mean 0 and variance d_k
    (each product q_i k_i has variance E[q_i^2] E[k_i^2] = 1, and the d_k terms are independent);
    dividing by sqrt(d_k) restores variance 1."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n_samples, d_k, generator=g)
    k = torch.randn(n_samples, d_k, generator=g)
    dots = (q * k).sum(-1)
    scaled = dots / math.sqrt(d_k)
    # softmax saturation over rows of seq_len scores
    rows = n_samples // seq_len
    s_un = dots[: rows * seq_len].view(rows, seq_len)
    s_sc = scaled[: rows * seq_len].view(rows, seq_len)
    return VarianceResult(d_k, dots.var().item(), scaled.var().item(), dots.mean().item(), scaled.mean().item(),
                          softmax(s_un).amax(-1).mean().item(), softmax(s_sc).amax(-1).mean().item())


# ════════════════════════════════════════════════════════════════════════════
# Supporting: FlashAttention-style tiling with an HBM IO counter (used by Part 4)
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class IOCount:
    reads: int = 0    # elements read from HBM
    writes: int = 0   # elements written to HBM

    @property
    def total(self) -> int:
        return self.reads + self.writes


def standard_attention_io(N: int, d: int) -> IOCount:
    """HBM traffic of the textbook three-kernel implementation (Dao et al., 2022, Algorithm 0):
    S = QK^T (read Q, K: 2Nd; write S: N^2) -> P = softmax(S) (read N^2, write N^2) -> O = PV (read P: N^2,
    read V: Nd; write O: Nd). Total = 4Nd + 4N^2 = Theta(Nd + N^2)."""
    return IOCount(reads=2 * N * d + N * N + N * N + N * d, writes=N * N + N * N + N * d)


def flash_attention_tiled(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_r: int = 128, block_c: int = 128,
                          causal: bool = False, io: Optional[IOCount] = None) -> torch.Tensor:
    """Exact attention for one head, q, k, v: (N, d), computed tile by tile with the online softmax.

    For each query block Q_i (B_r rows) we stream every key/value block (K_j, V_j) (B_c rows) "through SRAM",
    keeping only running statistics m_i (row max), l_i (row sum of exponentials) and the unnormalised output O_i.
    The N x N score matrix is never materialised. `io` counts the elements moved to/from HBM.
    """
    N, d = q.shape
    scale = 1.0 / math.sqrt(d)
    out = torch.empty_like(q)
    for i0 in range(0, N, block_r):
        i1 = min(i0 + block_r, N)
        q_i = q[i0:i1]
        if io:
            io.reads += q_i.numel()
        m_i = torch.full((i1 - i0,), float("-inf"), dtype=q.dtype, device=q.device)
        l_i = torch.zeros(i1 - i0, dtype=q.dtype, device=q.device)
        o_i = torch.zeros(i1 - i0, d, dtype=q.dtype, device=q.device)
        for j0 in range(0, N, block_c):
            if causal and j0 > i1 - 1:
                break
            j1 = min(j0 + block_c, N)
            k_j, v_j = k[j0:j1], v[j0:j1]
            if io:
                io.reads += k_j.numel() + v_j.numel()
            s_ij = (q_i @ k_j.T) * scale
            if causal:
                rows = torch.arange(i0, i1, device=q.device)[:, None]
                cols = torch.arange(j0, j1, device=q.device)[None, :]
                s_ij = s_ij.masked_fill(cols > rows, float("-inf"))
            m_new = torch.maximum(m_i, s_ij.amax(-1))
            p_ij = torch.exp(s_ij - m_new[:, None])
            alpha = torch.exp(m_i - m_new)                 # rescale what was accumulated so far
            l_i = alpha * l_i + p_ij.sum(-1)
            o_i = alpha[:, None] * o_i + p_ij @ v_j
            m_i = m_new
        out[i0:i1] = o_i / l_i[:, None]
        if io:
            io.writes += out[i0:i1].numel()
    return out


def flash_io_formula(N: int, d: int, block_r: int) -> int:
    """HBM element accesses of flash_attention_tiled (FlashAttention-2 loop order): Q and O once (2Nd), K and V
    once per query block (T_r * 2Nd). With block_r = Theta(M/d) this is Theta(Nd + N^2 d^2 / M), Theorem 2 of
    Dao et al. (2022)."""
    return 2 * N * d + math.ceil(N / block_r) * 2 * N * d


# ════════════════════════════════════════════════════════════════════════════
# Self-checks used by the tests and by main()
# ════════════════════════════════════════════════════════════════════════════


def reference_attention(x: torch.Tensor, mha: MultiHeadSelfAttention, causal: bool) -> torch.Tensor:
    """Same computation through PyTorch's fused kernel, used only to check our from-scratch module."""
    q, k, v = (mha.split_heads(x @ w) for w in (mha.W_Q, mha.W_K, mha.W_V))
    ctx = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return mha.merge_heads(ctx) @ mha.W_O


def main() -> int:
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bar = "═" * 78
    print(bar)
    print("  USSE NOOSPHERE // MISSION 3 (ps03) // PART 1: MULTI-HEAD SELF-ATTENTION FROM SCRATCH")
    print(f"  Squadron Lovelace // Systems Architect: Muhammad Ibrahim // torch {torch.__version__} on {device}")
    print(bar)

    print("\n[1-2] Projections and head split  (d_model=512, h=8, d_k=64, batch=2, N=16)")
    mha = MultiHeadSelfAttention(d_model=512, num_heads=8).to(device)
    x = torch.randn(2, 16, 512, device=device)
    q_heads = mha.split_heads(x @ mha.W_Q)
    print(f"  W_Q, W_K, W_V, W_O shapes: {[tuple(w.shape) for w in (mha.W_Q, mha.W_K, mha.W_V, mha.W_O)]}")
    print(f"  x {tuple(x.shape)} -> Q {tuple((x @ mha.W_Q).shape)} -> (b, N, h, d_k) "
          f"{tuple((x @ mha.W_Q).view(2, 16, 8, 64).shape)} -> (b, h, N, d_k) {tuple(q_heads.shape)}")
    out, w = mha(x, causal=True)
    print(f"  output {tuple(out.shape)}, attention weights {tuple(w.shape)}")

    print("\n[3] Causal masking (lower-triangular, -inf above the diagonal)")
    print("  mask for N=5:")
    for row in causal_mask(5).tolist():
        print("   " + " ".join(f"{v:>5.0f}" if v == 0 else " -inf" for v in row))
    upper = torch.triu(torch.ones(16, 16, dtype=torch.bool, device=device), diagonal=1)
    print(f"  attention weight on future positions: max {w[..., upper].abs().max().item():.1e} (must be 0)")
    print(f"  rows sum to 1: max |sum-1| = {(w.sum(-1) - 1).abs().max().item():.1e}")
    x_changed = x.clone()
    x_changed[:, 10:] = torch.randn_like(x_changed[:, 10:])
    leak = (mha(x_changed)[0][:, :10] - out[:, :10]).abs().max().item()
    print(f"  changing tokens 10..15 changes outputs 0..9 by {leak:.1e} (no information flows backwards)")
    ref = reference_attention(x, mha, causal=True)
    print(f"  max |ours - torch fused SDPA| = {(out - ref).abs().max().item():.1e}")

    print("\n[4] Variance preservation before the softmax (q, k ~ N(0, I), 200,000 samples)")
    print(f"  {'d_k':>5} {'Var(q.k)':>10} {'Var(q.k/sqrt(d_k))':>20} {'mean max-softmax (unscaled -> scaled), N=128':>46}")
    for d_k in (16, 64, 128, 256):
        r = variance_preservation(d_k)
        print(f"  {d_k:>5} {r.unscaled_var:>10.2f} {r.scaled_var:>20.4f} {r.max_softmax_unscaled:>22.3f} -> {r.max_softmax_scaled:.3f}")
    r = variance_preservation(64)
    ok = abs(r.scaled_var - 1.0) < 0.05
    print(f"  [{'PASS' if ok else 'FAIL'}] d_k=64: Var = {r.scaled_var:.4f} (target 1.0 ± 0.05); without scaling the softmax "
          f"puts {r.max_softmax_unscaled:.0%} of the weight on one key on average, with scaling {r.max_softmax_scaled:.0%}.")

    print("\n[Part 4 support] FlashAttention-style tiling: exactness and HBM traffic (one head, d=64, fp32)")
    print(f"  {'N':>6} {'max |tiled - naive|':>20} {'HBM elems standard':>20} {'HBM elems tiled':>16} {'reduction':>10}")
    for N in (512, 1024, 2048, 4096):
        qh, kh, vh = (torch.randn(N, 64, device=device) for _ in range(3))
        io = IOCount()
        o_t = flash_attention_tiled(qh, kh, vh, block_r=128, block_c=128, io=io)
        o_n, _ = scaled_dot_product_attention(qh, kh, vh)
        std = standard_attention_io(N, 64)
        print(f"  {N:>6} {(o_t - o_n).abs().max().item():>20.1e} {std.total:>20,} {io.total:>16,} {std.total / io.total:>9.1f}x")
    print("  (tiled traffic = 2*N*d for Q and O plus 2*N*d per query block for K and V. The N x N matrix never "
          "reaches HBM, so the extra memory is O(N); the traffic is still quadratic, divided by ~2*B_r/d = 4x here.)")
    print("\n" + bar)
    print("  PART 1 COMPLETE: MultiHeadSelfAttention verified (shapes, causality, exactness, variance).")
    print(bar)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
