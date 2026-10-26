#!/usr/bin/env python3
"""
02_RoPE_Positional_Engine.py — Rotary Positional Embeddings (RoPE) Engine
=========================================================================

USSE Noosphere // Squadron Lovelace (Group 01) // Mission 3 (ps03) // Part 2
Station: Communications & Protocol Officer — Anas Asif Hussain Shaik
Sector SEC-03-M45 (Pleiades Open Cluster) // AURA Subsystem 03: Distributed Multi-Head Attention Mesh

What this file provides (ps03 Part 2, items 1-3, plus the directive's extrapolation evaluation):
  1. precompute_freqs_cis() and apply_rope(x, freqs_cis): rotary frequencies theta_i = base^(-2(i-1)/d),
     i = 1..d/2, applied to Query and Key tensors as complex 2D planar rotations (Su et al., 2021).
  2. Relative invariance: a short proof (below) and an empirical check over random q, k, m, n, delta that
     <RoPE(q, m), RoPE(k, n)> = <RoPE(q, m + delta), RoPE(k, n + delta)>.
  3. Figures: the attention-score / attention-weight decay with distance for RoPE versus ALiBi's linear
     penalty, and side-by-side attention heatmaps (figures/*.png).
  4. Context-length extrapolation: which frequency bands see rotation angles at inference that they never saw
     in training, and how Position Interpolation and NTK-aware base scaling change that.

Proof of relative invariance. Pair the coordinates of q as complex numbers q_c[i] = q[2i] + j q[2i+1].
RoPE multiplies pair i at position m by e^{j m theta_i}. For real vectors, <a, b> = Re(sum_i a_c[i] conj(b_c[i])), so
    <RoPE(q, m), RoPE(k, n)> = Re( sum_i q_c[i] e^{j m theta_i} conj(k_c[i] e^{j n theta_i}) )
                             = Re( sum_i q_c[i] conj(k_c[i]) e^{j (m - n) theta_i} ) = g(q, k, m - n).
Adding delta to both m and n leaves m - n, and therefore the score, unchanged.

Run:   python 02_RoPE_Positional_Engine.py
"""

from __future__ import annotations

import math
import os
from typing import Tuple

import torch

SEED = 2026
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")


# ════════════════════════════════════════════════════════════════════════════
# 1. Rotary frequencies and apply_rope
# ════════════════════════════════════════════════════════════════════════════


def rope_frequencies(dim: int, base: float = 10000.0) -> torch.Tensor:
    """theta_i = base^(-2(i-1)/dim) for i = 1..dim/2 (equivalently base^(-2i/dim) for i = 0..dim/2-1)."""
    if dim % 2:
        raise ValueError("RoPE needs an even head dimension (coordinates are rotated in pairs)")
    i = torch.arange(1, dim // 2 + 1, dtype=torch.float64)
    return base ** (-2.0 * (i - 1) / dim)


def precompute_freqs_cis(dim: int, max_positions: int, base: float = 10000.0,
                         position_scale: float = 1.0) -> torch.Tensor:
    """cis(m * theta_i) = e^{j m theta_i} for m = 0..max_positions-1 -> complex tensor (max_positions, dim/2).

    position_scale < 1 implements Position Interpolation (Chen et al., 2023): m is replaced by m * scale.
    """
    theta = rope_frequencies(dim, base)
    m = torch.arange(max_positions, dtype=torch.float64) * position_scale
    angles = torch.outer(m, theta)
    return torch.polar(torch.ones_like(angles), angles).to(torch.complex64)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Rotate every consecutive coordinate pair of x by its position's angle.

    x: (..., seq_len, head_dim) real, e.g. Q or K of shape (batch, heads, seq_len, head_dim).
    freqs_cis: (seq_len, head_dim/2) complex, the rows for the positions of x.
    The rotation of pair (x_{2i}, x_{2i+1}) by angle phi is the complex product (x_{2i} + j x_{2i+1}) e^{j phi},
    i.e. the 2x2 matrix [[cos phi, -sin phi], [sin phi, cos phi]].
    """
    if x.size(-1) % 2:
        raise ValueError("head_dim must be even")
    if freqs_cis.shape != (x.size(-2), x.size(-1) // 2):
        raise ValueError(f"freqs_cis must be (seq_len, head_dim/2) = {(x.size(-2), x.size(-1) // 2)}, "
                         f"got {tuple(freqs_cis.shape)}")
    x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2).contiguous())
    rotated = x_c * freqs_cis.to(x_c.device)
    return torch.view_as_real(rotated).flatten(-2).type_as(x)


def apply_rope_qk(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to Query and Key (Values are never rotated)."""
    return apply_rope(q, freqs_cis), apply_rope(k, freqs_cis)


def rope_rotation_matrix(dim: int, m: int, base: float = 10000.0) -> torch.Tensor:
    """The block-diagonal real matrix R^d_{Theta,m} of Su et al., used to cross-check apply_rope."""
    theta = rope_frequencies(dim, base)
    R = torch.zeros(dim, dim, dtype=torch.float64)
    for i, t in enumerate(theta.tolist()):
        c, s = math.cos(m * t), math.sin(m * t)
        R[2 * i:2 * i + 2, 2 * i:2 * i + 2] = torch.tensor([[c, -s], [s, c]], dtype=torch.float64)
    return R


# ════════════════════════════════════════════════════════════════════════════
# 2. Relative invariance check
# ════════════════════════════════════════════════════════════════════════════


def score_at(q: torch.Tensor, k: torch.Tensor, m: int, n: int, table: torch.Tensor) -> float:
    """<RoPE(q, m), RoPE(k, n)> for single vectors q, k of shape (dim,)."""
    qm = apply_rope(q[None, :], table[m:m + 1])[0]
    kn = apply_rope(k[None, :], table[n:n + 1])[0]
    return float((qm.double() * kn.double()).sum())


def invariance_trials(trials: int = 1000, dim: int = 128, max_pos: int = 8192, seed: int = SEED) -> dict:
    g = torch.Generator().manual_seed(seed)
    table = precompute_freqs_cis(dim, max_pos * 2)
    worst = 0.0
    worst_rel = 0.0
    examples = []
    for t in range(trials):
        q, k = torch.randn(dim, generator=g), torch.randn(dim, generator=g)
        m, n, delta = (int(v) for v in torch.randint(0, max_pos, (3,), generator=g))
        a, b = score_at(q, k, m, n, table), score_at(q, k, m + delta, n + delta, table)
        worst = max(worst, abs(a - b))
        worst_rel = max(worst_rel, abs(a - b) / max(1e-9, abs(a)))
        if t < 3:
            examples.append((m, n, delta, a, b))
    return {"trials": trials, "max_abs_diff": worst, "max_rel_diff": worst_rel, "examples": examples}


# ════════════════════════════════════════════════════════════════════════════
# 3. RoPE vs ALiBi distance decay
# ════════════════════════════════════════════════════════════════════════════


def rope_score_vs_distance(dim: int, distances: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Expected score for identical unit-norm q = k as a function of distance r:
    (1/(d/2)) * sum_i cos(r theta_i). This is the long-term decay curve of Su et al. (2021, §3.4.3)."""
    theta = rope_frequencies(dim, base)
    return torch.cos(torch.outer(distances.double(), theta)).mean(-1)


def alibi_slopes(num_heads: int) -> torch.Tensor:
    """ALiBi head slopes m_h = 2^(-8h/H), h = 1..H (Press et al., 2022)."""
    h = torch.arange(1, num_heads + 1, dtype=torch.float64)
    return 2.0 ** (-8.0 * h / num_heads)


def attention_weights_rope(seq_len: int, dim: int = 64, temperature: float = 4.0) -> torch.Tensor:
    """Causal attention weights when every token has the same content vector: the pattern is then driven by
    position alone. Scores = temperature * rope_score(i - j), softmax over j <= i."""
    i = torch.arange(seq_len)
    dist = (i[:, None] - i[None, :]).clamp(min=0)
    s = temperature * rope_score_vs_distance(dim, dist.flatten()).view(seq_len, seq_len)
    s = s.masked_fill(i[None, :] > i[:, None], float("-inf"))
    return torch.softmax(s, -1)


def attention_weights_alibi(seq_len: int, slope: float) -> torch.Tensor:
    """Causal attention weights with equal content scores and the ALiBi bias -slope * (i - j)."""
    i = torch.arange(seq_len, dtype=torch.float64)
    s = -slope * (i[:, None] - i[None, :])
    s = s.masked_fill(i[None, :] > i[:, None], float("-inf"))
    return torch.softmax(s, -1)


def make_figures(out_dir: str = FIG_DIR) -> list[str]:
    import matplotlib
    import matplotlib.pyplot as plt     # the backend is left alone, so notebooks keep inline plots

    os.makedirs(out_dir, exist_ok=True)
    paths = []

    # (a) score vs distance and attention-weight vs distance
    r = torch.arange(0, 4097)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for dim in (64, 128):
        ax[0].plot(r, rope_score_vs_distance(dim, r), lw=1, label=f"RoPE, d={dim}, base 10000")
    ax0b = ax[0].twinx()
    for h, sl, col in zip((1, 4, 8), alibi_slopes(8)[[0, 3, 7]].tolist(), ("tab:red", "tab:purple", "tab:green")):
        ax0b.plot(r, -sl * r.double(), ls="--", lw=1.2, color=col, label=f"ALiBi bias, head {h} (slope 2^-{h})")
    ax0b.set_yscale("symlog", linthresh=1)
    ax0b.set_ylabel("ALiBi bias -slope·|m-n| (dashed)")
    ax[0].set_xscale("symlog", linthresh=10)
    ax[0].set_xlabel("relative distance |m - n| (tokens)")
    ax[0].set_ylabel("RoPE score, q = k (solid)")
    ax[0].set_title("Positional score vs distance")
    h1, l1 = ax[0].get_legend_handles_labels()
    h2, l2 = ax0b.get_legend_handles_labels()
    ax[0].legend(h1 + h2, l1 + l2, fontsize=7, loc="lower left")
    ax[0].grid(alpha=.3)
    seq = 256
    w_rope = attention_weights_rope(seq)[-1].flip(0)          # weights of the last query by distance
    ax[1].plot(range(seq), w_rope, label="RoPE (d=64, equal content)")
    for sl, col in zip(alibi_slopes(8)[[1, 3, 7]].tolist(), ("tab:red", "tab:purple", "tab:green")):
        ax[1].plot(range(seq), attention_weights_alibi(seq, sl)[-1].flip(0), ls="--", color=col,
                   label=f"ALiBi slope {sl:.3g}")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("distance from the query (tokens)")
    ax[1].set_ylabel("attention weight (log)")
    ax[1].set_title(f"Attention weight decay, last query of {seq}")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "rope_vs_alibi_decay.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)

    # (b) heatmaps
    seq = 64
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
    for a, (title, w) in zip(ax, (("RoPE (d=64): oscillating decay", attention_weights_rope(seq)),
                                  ("ALiBi (slope 1/16): monotone linear penalty",
                                   attention_weights_alibi(seq, 1 / 16)))):
        im = a.imshow(w.numpy(), cmap="magma", norm=matplotlib.colors.LogNorm(vmin=1e-4, vmax=1))
        a.set_title(title, fontsize=10)
        a.set_xlabel("key position j")
        a.set_ylabel("query position i")
    fig.colorbar(im, ax=ax, shrink=.85, label="attention weight")
    p = os.path.join(out_dir, "rope_alibi_attention_heatmaps.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)

    # (c) extrapolation: rotation angles of the lowest-frequency pair
    fig, ax = plt.subplots(figsize=(7, 3.8))
    dim, train = 128, 4096
    theta = rope_frequencies(dim)
    for name, base, scale, ls in (("RoPE (base 10000)", 10000.0, 1.0, "-"),
                                  ("Position Interpolation x4", 10000.0, 0.25, "-"),
                                  ("NTK-aware base x4^(d/(d-2)) (same line for this pair)", ntk_base(10000.0, 4.0, dim),
                                   1.0, (0, (4, 3)))):
        th = rope_frequencies(dim, base)
        positions = torch.arange(0, 4 * train, 64, dtype=torch.float64)
        ax.plot(positions, positions * scale * th[-1], ls=ls, lw=2, label=name)
    ax.axhline(float(train * theta[-1]), color="k", ls=":", lw=1, label="largest angle seen in training (4096 tokens)")
    ax.axvline(train, color="grey", ls=":", lw=1)
    ax.set_xlabel("position m")
    ax.set_ylabel("angle of the slowest pair (rad)")
    ax.set_title("Extrapolation: unseen angles beyond the training length")
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "rope_extrapolation_angles.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)
    return paths


# ════════════════════════════════════════════════════════════════════════════
# 4. Context-length extrapolation
# ════════════════════════════════════════════════════════════════════════════


def ntk_base(base: float, factor: float, dim: int) -> float:
    """NTK-aware scaled RoPE base: base * factor^(d/(d-2)) (bloc97, 2023; used by YaRN and Code Llama)."""
    return base * factor ** (dim / (dim - 2))


def unseen_angle_fraction(dim: int, train_len: int, test_len: int, base: float = 10000.0,
                          position_scale: float = 1.0) -> float:
    """Fraction of frequency pairs whose largest angle at test time exceeds the largest angle seen in training
    *without* having completed a full turn in training (pairs that completed a turn have seen every angle)."""
    theta = rope_frequencies(dim, 10000.0)
    theta_test = rope_frequencies(dim, base)
    seen = train_len * theta
    test = test_len * position_scale * theta_test
    unseen = (seen < 2 * math.pi) & (test > seen)
    return unseen.double().mean().item()


def main() -> int:
    bar = "═" * 78
    print(bar)
    print("  USSE NOOSPHERE // MISSION 3 (ps03) // PART 2: ROTARY POSITIONAL EMBEDDINGS (RoPE)")
    print("  Squadron Lovelace // Communications & Protocol Officer: Anas Asif Hussain Shaik")
    print(bar)

    print("\n[1] Rotary frequencies theta_i = base^(-2(i-1)/d), d=128, base=10000")
    th = rope_frequencies(128)
    print(f"  theta_1 = {th[0]:.4f} rad/token (period {2 * math.pi / th[0]:.1f} tokens) ... "
          f"theta_64 = {th[-1]:.3e} rad/token (period {2 * math.pi / th[-1]:,.0f} tokens)")
    q = torch.randn(2, 8, 16, 128)
    table = precompute_freqs_cis(128, 16)
    qr = apply_rope(q, table)
    R = rope_rotation_matrix(128, m=5)
    print(f"  apply_rope on Q (2, 8, 16, 128) -> {tuple(qr.shape)}; norms preserved: "
          f"max |‖Rq‖-‖q‖| = {(qr.norm(dim=-1) - q.norm(dim=-1)).abs().max().item():.1e}")
    print(f"  equals the block-diagonal matrix R_(Theta,m=5): max diff "
          f"{(qr[0, 0, 5].double() - R @ q[0, 0, 5].double()).abs().max().item():.1e}")

    print("\n[2] Relative invariance <RoPE(q,m),RoPE(k,n)> = <RoPE(q,m+δ),RoPE(k,n+δ)> (1000 random trials, d=128)")
    res = invariance_trials()
    for m, n, d, a, b in res["examples"]:
        print(f"  m={m:>5}, n={n:>5}, δ={d:>5}: {a:+.6f} vs {b:+.6f}")
    ok = res["max_rel_diff"] < 1e-4
    print(f"  [{'PASS' if ok else 'FAIL'}] max |difference| = {res['max_abs_diff']:.2e} "
          f"(relative {res['max_rel_diff']:.1e}; float32 rotation, positions up to 16,384)")

    print("\n[3] Distance decay: RoPE vs ALiBi")
    print(f"  {'distance':>9} {'RoPE score d=64':>16} {'RoPE score d=128':>17} {'ALiBi bias (slope 1/2)':>23} {'(slope 1/256)':>14}")
    for r in (0, 1, 4, 16, 64, 256, 1024, 4096):
        rr = torch.tensor([r])
        print(f"  {r:>9} {rope_score_vs_distance(64, rr).item():>16.3f} {rope_score_vs_distance(128, rr).item():>17.3f} "
              f"{-0.5 * r:>23.1f} {-r / 256:>14.2f}")
    print("  RoPE: the positional contribution falls from 1 towards 0 with distance, oscillating (cos of many "
          "frequencies). ALiBi: a strict linear penalty per head; far tokens are suppressed exponentially in weight.")
    import matplotlib
    matplotlib.use("Agg")               # script mode: write files, open no windows
    for p in make_figures():
        print(f"  figure written: figures/{os.path.basename(p)}")

    print("\n[4] Context-length extrapolation (trained on 4,096 tokens, d=128)")
    print(f"  {'test length':>11} {'plain RoPE':>11} {'Position Interp.':>17} {'NTK-aware base':>15}  (share of frequency pairs at unseen angles)")
    for L in (4096, 8192, 16384, 32768):
        f = L / 4096
        print(f"  {L:>11,} {unseen_angle_fraction(128, 4096, L):>11.1%} "
              f"{unseen_angle_fraction(128, 4096, L, position_scale=1 / f):>17.1%} "
              f"{unseen_angle_fraction(128, 4096, L, base=ntk_base(10000.0, f, 128)):>15.1%}")
    print("  Plain RoPE: the slow pairs (periods longer than the training length) meet rotation angles they never saw,")
    print("  which is why it degrades past its training length. Interpolation squeezes positions back into the")
    print("  trained range; NTK scaling stretches only the slow frequencies. ALiBi has no angles to leave the")
    print("  trained range, which is why it extrapolates zero-shot (but it cannot represent oscillating patterns).")
    print("\n" + bar)
    print("  PART 2 COMPLETE: apply_rope verified (rotation, norm, relative invariance); decay figures written.")
    print(bar)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
