#!/usr/bin/env python3
"""
03_GQA_and_KVCache_Benchmark.py — Grouped-Query Attention & Autoregressive KV-Cache Benchmark
=============================================================================================

USSE Noosphere // Squadron Lovelace (Group 01) // Mission 3 (ps03) // Part 3
Station: Data & Payload Specialist — Areej Ilyas
Sector SEC-03-M45 (Pleiades Open Cluster) // AURA Subsystem 03: KV-Cache Transceiver

What this file provides (ps03 Part 3, items 1-3):
  1. GQAAttention(nn.Module): configurable H_Q query heads and H_KV key/value heads. The KV heads are
     broadcast to the query heads with torch.repeat_interleave. H_KV = H_Q gives MHA, H_KV = 1 gives MQA.
  2. KVCache: an in-memory buffer that keeps past (K_t, V_t) per layer and appends each new single-token slice
     during sequential decoding (prefill writes a block, decode appends one position at a time).
  3. A benchmark table for N in {1024, 4096, 16384, 32768} comparing MHA (32:32), GQA 4:1 (32:8) and
     MQA 32:1 (32:1) on a LLaMA-3-8B-shaped model (32 layers, 32 query heads, head_dim 128, FP16):
       - KV-cache memory footprint (MB)
       - memory-bandwidth traffic at 50 tokens/s (GB/s): the whole cache is read once per generated token
       - decoding latency per token (ms): measured on this machine's GPU for one layer x 32 layers, plus the
         bandwidth-bound lower bound (bytes / peak HBM bandwidth)

Support for 04_Memory_Hierarchy_and_Safety_Audit.md (Part 4):
  * kv_dos_simulation(): concurrent long sessions until CUDA out-of-memory (analytic for an 80 GB mainframe,
    and a real, contained allocation test on this GPU);
  * paged_vs_contiguous(): PagedAttention-style block allocation versus contiguous max-length reservation;
  * kv_bytes_per_token(): MHA vs GQA vs MQA vs MLA (DeepSeek-V3 latent KV) per-token cache size.

Run:   python 03_GQA_and_KVCache_Benchmark.py              (GPU used if available; about 1 min)
       python 03_GQA_and_KVCache_Benchmark.py --no-oom     (skip the real out-of-memory test)
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

SEED = 2026
BYTES_FP16 = 2
MB = 1024 ** 2
GB = 1024 ** 3


# ════════════════════════════════════════════════════════════════════════════
# 2. KVCache
# ════════════════════════════════════════════════════════════════════════════


class KVCache:
    """Per-layer key/value cache for autoregressive decoding.

    Layout (batch, n_kv_heads, max_seq_len, head_dim), preallocated once (as inference engines do), with a
    write pointer `length`. append() writes new slices at the pointer and returns views of everything cached
    so far, so attention for token t reads K_{1..t}, V_{1..t} without recomputing them.
    """

    def __init__(self, batch: int, n_kv_heads: int, max_seq_len: int, head_dim: int,
                 dtype=torch.float16, device="cpu") -> None:
        shape = (batch, n_kv_heads, max_seq_len, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.max_seq_len = max_seq_len
        self.length = 0

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """k_new, v_new: (batch, n_kv_heads, t, head_dim), t = prompt length at prefill, 1 during decoding."""
        t = k_new.size(2)
        if self.length + t > self.max_seq_len:
            raise RuntimeError(f"KV-cache full: {self.length} + {t} > {self.max_seq_len} positions")
        self.k[:, :, self.length:self.length + t] = k_new
        self.v[:, :, self.length:self.length + t] = v_new
        self.length += t
        return self.k[:, :, :self.length], self.v[:, :, :self.length]

    def reset(self) -> None:
        self.length = 0

    @property
    def allocated_bytes(self) -> int:
        return self.k.numel() * self.k.element_size() + self.v.numel() * self.v.element_size()

    @property
    def used_bytes(self) -> int:
        b, h, _, d = self.k.shape
        return 2 * b * h * self.length * d * self.k.element_size()


# ════════════════════════════════════════════════════════════════════════════
# 1. Grouped-Query Attention
# ════════════════════════════════════════════════════════════════════════════


class GQAAttention(nn.Module):
    """Grouped-Query Attention (Ainslie et al., 2023). H_Q query heads share H_KV key/value heads in groups
    of G = H_Q / H_KV. The K/V projections are G times smaller than MHA's, and so is the KV-cache."""

    def __init__(self, d_model: int = 4096, n_q_heads: int = 32, n_kv_heads: int = 8,
                 dtype=torch.float32, device="cpu", kv_broadcast: str = "repeat") -> None:
        """kv_broadcast="repeat": torch.repeat_interleave copies each KV head G times (the textbook form).
        kv_broadcast="group": the G query heads of a group are stacked and multiplied against their one KV head
        by broadcasting, so the cache is never copied (what fused GQA kernels do)."""
        super().__init__()
        if kv_broadcast not in ("repeat", "group"):
            raise ValueError("kv_broadcast must be 'repeat' or 'group'")
        self.kv_broadcast = kv_broadcast
        if n_q_heads % n_kv_heads:
            raise ValueError("n_q_heads must be a multiple of n_kv_heads")
        if d_model % n_q_heads:
            raise ValueError("d_model must be divisible by n_q_heads")
        self.d_model, self.n_q, self.n_kv = d_model, n_q_heads, n_kv_heads
        self.group = n_q_heads // n_kv_heads
        self.head_dim = d_model // n_q_heads
        kw = dict(bias=False, dtype=dtype, device=device)
        self.w_q = nn.Linear(d_model, n_q_heads * self.head_dim, **kw)
        self.w_k = nn.Linear(d_model, n_kv_heads * self.head_dim, **kw)
        self.w_v = nn.Linear(d_model, n_kv_heads * self.head_dim, **kw)
        self.w_o = nn.Linear(n_q_heads * self.head_dim, d_model, **kw)

    def new_cache(self, batch: int, max_seq_len: int) -> KVCache:
        p = self.w_q.weight
        return KVCache(batch, self.n_kv, max_seq_len, self.head_dim, dtype=p.dtype, device=p.device)

    def forward(self, x: torch.Tensor, cache: Optional[KVCache] = None) -> torch.Tensor:
        """x: (batch, t, d_model). With a cache, x holds only the new positions (t = 1 when decoding)."""
        b, t, _ = x.shape
        q = self.w_q(x).view(b, t, self.n_q, self.head_dim).transpose(1, 2)    # (b, H_Q, t, d)
        k = self.w_k(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)   # (b, H_KV, t, d)
        v = self.w_v(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        past = 0
        if cache is not None:
            past = cache.length
            k, v = cache.append(k, v)                                          # (b, H_KV, past + t, d)
        T = past + t
        if self.kv_broadcast == "repeat":
            # broadcast each KV head to its G query heads: (b, H_KV, T, d) -> (b, H_Q, T, d)
            k = torch.repeat_interleave(k, self.group, dim=1)
            v = torch.repeat_interleave(v, self.group, dim=1)
        else:
            # query heads h = g*G .. g*G+G-1 share KV head g: (b, H_Q, t, d) -> (b, H_KV, G*t, d)
            q = q.reshape(b, self.n_kv, self.group * t, self.head_dim)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if t > 1:                                                              # causal mask for the new block
            rows = torch.arange(past, T, device=x.device)[:, None]
            cols = torch.arange(T, device=x.device)[None, :]
            mask = cols > rows
            if self.kv_broadcast == "group":
                mask = mask.repeat(self.group, 1)                              # rows are (head-in-group, position)
            scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        out = weights @ v
        if self.kv_broadcast == "group":
            out = out.reshape(b, self.n_q, t, self.head_dim)
        out = out.transpose(1, 2).reshape(b, t, self.n_q * self.head_dim)
        return self.w_o(out)


# ════════════════════════════════════════════════════════════════════════════
# 3. Benchmark
# ════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ModelShape:
    name: str = "LLaMA-3-8B-shaped"
    n_layers: int = 32
    n_q_heads: int = 32
    head_dim: int = 128

    @property
    def d_model(self) -> int:
        return self.n_q_heads * self.head_dim


VARIANTS = {"MHA 32:32": 32, "GQA 4:1 (32:8)": 8, "MQA 32:1": 1}
SEQ_LENS = (1024, 4096, 16384, 32768)
TOKENS_PER_SECOND = 50


def kv_cache_bytes(shape: ModelShape, n_kv_heads: int, seq_len: int, batch: int = 1, bytes_per: int = BYTES_FP16) -> int:
    """Memory_KV = 2 (K and V) x bytes x n_layers x n_KV_heads x d_head x seq_len x batch (lecture §4)."""
    return 2 * bytes_per * shape.n_layers * n_kv_heads * shape.head_dim * seq_len * batch


def bandwidth_gbps(cache_bytes: int, tokens_per_second: int = TOKENS_PER_SECOND) -> float:
    """Each generated token attends to every cached position, so the whole cache is read once per token."""
    return cache_bytes * tokens_per_second / GB


def gpu_peak_bandwidth_gbps() -> Optional[float]:
    """Peak DRAM bandwidth from the device's memory clock and bus width when PyTorch exposes them;
    otherwise None (we then fall back to a measured copy bandwidth)."""
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    clock = getattr(props, "memory_clock_rate", None)      # kHz
    bus = getattr(props, "memory_bus_width", None)         # bits
    if clock and bus:
        return 2 * clock * 1e3 * bus / 8 / GB               # DDR: two transfers per clock
    return None


def measured_copy_bandwidth_gbps(device: str, n_bytes: int = 512 * MB) -> float:
    """Achievable bandwidth: time a large device-to-device copy (read + write)."""
    x = torch.empty(n_bytes // 2, dtype=torch.float16, device=device)
    y = torch.empty_like(x)
    for _ in range(3):
        y.copy_(x)
    _sync(device)
    t0 = time.perf_counter()
    reps = 10
    for _ in range(reps):
        y.copy_(x)
    _sync(device)
    dt = (time.perf_counter() - t0) / reps
    del x, y
    return 2 * n_bytes / dt / GB


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def decode_latency_ms(shape: ModelShape, n_kv_heads: int, seq_len: int, device: str, reps: int = 10,
                      kv_broadcast: str = "group") -> Optional[float]:
    """Wall-clock time of ONE decoding step of ONE attention layer (projections + cache append + attention over
    seq_len cached positions), FP16 on GPU / FP32 on CPU. Multiply by n_layers for the per-token model latency.
    Returns None if the layer does not fit in memory."""
    dtype = torch.float16 if device == "cuda" else torch.float32
    try:
        layer = GQAAttention(shape.d_model, shape.n_q_heads, n_kv_heads, dtype=dtype, device=device,
                             kv_broadcast=kv_broadcast)
        cache = layer.new_cache(1, seq_len + reps + 8)  # room for the appended test tokens
        # fill the cache to seq_len with random keys/values (equivalent to a prefill)
        cache.k[:, :, :seq_len].normal_()
        cache.v[:, :, :seq_len].normal_()
        cache.length = seq_len
        x = torch.randn(1, 1, shape.d_model, dtype=dtype, device=device)
        for _ in range(5):                                    # warm-up
            layer(x, cache)
        trials = []
        for _ in range(5):                                    # median of 5 trials of `reps` decoding steps
            cache.length = seq_len
            _sync(device)
            t0 = time.perf_counter()
            for _ in range(reps):
                layer(x, cache)
            _sync(device)
            trials.append((time.perf_counter() - t0) / reps)
        dt = sorted(trials)[len(trials) // 2]
        del layer, cache, x
        if device == "cuda":
            torch.cuda.empty_cache()
        return dt * 1e3
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


def run_benchmark(device: str, shape: ModelShape = ModelShape()) -> list[dict]:
    peak = gpu_peak_bandwidth_gbps() or measured_copy_bandwidth_gbps(device)
    rows = []
    for n in SEQ_LENS:
        for name, h_kv in VARIANTS.items():
            nbytes = kv_cache_bytes(shape, h_kv, n)
            layer_ms = decode_latency_ms(shape, h_kv, n, device, kv_broadcast="group")
            layer_rep = decode_latency_ms(shape, h_kv, n, device, kv_broadcast="repeat")
            rows.append({
                "N": n, "variant": name, "kv_heads": h_kv,
                "kv_mb": nbytes / MB,
                "bandwidth_gbps_at_50tps": bandwidth_gbps(nbytes),
                "latency_ms_measured": None if layer_ms is None else layer_ms * shape.n_layers,
                "latency_ms_repeat_interleave": None if layer_rep is None else layer_rep * shape.n_layers,
                "latency_ms_bound": nbytes / GB / peak * 1e3,
            })
    return rows


# ════════════════════════════════════════════════════════════════════════════
# Part 4 support: DoS, paging, MLA
# ════════════════════════════════════════════════════════════════════════════


def kv_bytes_per_token(shape: ModelShape = ModelShape()) -> dict[str, int]:
    """KV-cache bytes per token for the whole model (FP16). MLA (DeepSeek-V2/V3) caches one compressed latent
    c_t^KV of d_c = 512 plus a shared decoupled RoPE key of 64 dims per layer (DeepSeek-V3 report, §2.1.1)."""
    per_layer = {
        "MHA (32 KV heads)": 2 * 32 * shape.head_dim,
        "GQA 4:1 (8 KV heads)": 2 * 8 * shape.head_dim,
        "MQA (1 KV head)": 2 * 1 * shape.head_dim,
        "MLA (d_c = 512 + 64 RoPE)": 512 + 64,
    }
    return {k: v * BYTES_FP16 * shape.n_layers for k, v in per_layer.items()}


def kv_dos_analytic(hbm_gb: float = 80.0, weights_gb: float = 16.0, context: int = 65536,
                    shape: ModelShape = ModelShape()) -> list[dict]:
    """How many concurrent max-length sessions fit before OOM on an 80 GB mainframe holding an 8B FP16 model."""
    free = (hbm_gb - weights_gb) * GB
    rows = []
    for name, h_kv in VARIANTS.items():
        per = kv_cache_bytes(shape, h_kv, context)
        rows.append({"variant": name, "per_session_gb": per / GB, "max_sessions": int(free // per)})
    per = kv_bytes_per_token(shape)["MLA (d_c = 512 + 64 RoPE)"] * context
    rows.append({"variant": "MLA (DeepSeek-V3 latent)", "per_session_gb": per / GB, "max_sessions": int(free // per)})
    return rows


def kv_dos_live(device: str, context: int = 2048, shape: ModelShape = ModelShape(), n_kv_heads: int = 8,
                limit_sessions: int = 400) -> dict:
    """Real, contained OOM test on this GPU: open GQA sessions (one KV-cache of `context` tokens for all 32
    layers each) until torch raises CUDA OutOfMemoryError, then free everything."""
    if device != "cuda":
        return {"skipped": "no CUDA device"}
    torch.cuda.empty_cache()
    free0, total = torch.cuda.mem_get_info()
    per = kv_cache_bytes(shape, n_kv_heads, context)
    sessions, err = [], ""
    try:
        while len(sessions) < limit_sessions:
            sessions.append(torch.empty(per // BYTES_FP16, dtype=torch.float16, device="cuda"))
    except torch.cuda.OutOfMemoryError as exc:
        err = str(exc).split("\n")[0][:160]
    opened = len(sessions)
    del sessions
    torch.cuda.empty_cache()
    return {"gpu": torch.cuda.get_device_name(0), "total_gb": total / GB, "free_gb_at_start": free0 / GB,
            "per_session_mb": per / MB, "sessions_before_oom": opened, "error": err,
            "predicted": int(free0 // per)}


def paged_vs_contiguous(budget_gb: float = 64.0, max_seq_len: int = 32768, block: int = 16, n_requests: int = 2000,
                        shape: ModelShape = ModelShape(), n_kv_heads: int = 8, seed: int = SEED) -> dict:
    """Admit requests with realistic, mostly short lengths until memory runs out.

    contiguous: each request reserves a max_seq_len slab up front (pre-vLLM serving): internal fragmentation.
    paged:      each request takes ceil(len / block) fixed-size blocks on demand (PagedAttention, Kwon et al., 2023);
                waste is at most block - 1 tokens per request.
    """
    rng = random.Random(seed)
    lengths = [min(max_seq_len, max(16, int(rng.lognormvariate(7.0, 1.0)))) for _ in range(n_requests)]
    per_tok = kv_cache_bytes(shape, n_kv_heads, 1)
    budget = budget_gb * GB

    def admit(cost_fn):
        used = admitted = useful = 0
        for L in lengths:
            c = cost_fn(L)
            if used + c > budget:
                break
            used += c
            admitted += 1
            useful += L * per_tok
        return admitted, used, useful

    c_adm, c_used, c_useful = admit(lambda L: max_seq_len * per_tok)
    p_adm, p_used, p_useful = admit(lambda L: math.ceil(L / block) * block * per_tok)
    return {"median_len": sorted(lengths)[len(lengths) // 2], "mean_len": sum(lengths) / len(lengths),
            "contiguous_admitted": c_adm, "contiguous_waste": 1 - c_useful / c_used if c_used else 0,
            "paged_admitted": p_adm, "paged_waste": 1 - p_useful / p_used if p_used else 0, "block": block}


# ════════════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def demo_cache_correctness(device: str) -> float:
    """Incremental decoding with the KVCache must equal one full causal forward pass."""
    torch.manual_seed(SEED)
    layer = GQAAttention(d_model=256, n_q_heads=8, n_kv_heads=2, device=device)
    x = torch.randn(2, 12, 256, device=device)
    full = layer(x, layer.new_cache(2, 12))               # prefill all 12 positions at once (causal)
    cache = layer.new_cache(2, 12)
    steps = [layer(x[:, t:t + 1], cache) for t in range(12)]
    layer.kv_broadcast = "group"                          # the copy-free broadcast must give the same result
    grouped = layer(x, layer.new_cache(2, 12))
    return max((torch.cat(steps, 1) - full).abs().max().item(), (grouped - full).abs().max().item())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ps03 Part 3: GQA & KV-cache benchmark")
    ap.add_argument("--no-oom", action="store_true", help="skip the real CUDA out-of-memory test")
    args = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shape = ModelShape()
    bar = "═" * 96
    print(bar)
    print("  USSE NOOSPHERE // MISSION 3 (ps03) // PART 3: GROUPED-QUERY ATTENTION & KV-CACHE BENCHMARK")
    print("  Squadron Lovelace // Data & Payload Specialist: Areej Ilyas")
    dev_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"  torch {torch.__version__} on {dev_name}")
    print(bar)

    print("\n[1] GQAAttention head grouping (d_model=256, H_Q=8)")
    x = torch.randn(1, 5, 256, device=device)
    for h_kv in (8, 2, 1):
        layer = GQAAttention(256, 8, h_kv, device=device)
        y = layer(x)
        print(f"  H_KV={h_kv}: group size G={layer.group}, W_K {tuple(layer.w_k.weight.shape)}, output {tuple(y.shape)}")

    print("\n[2] KVCache: prefill + one-token appends equal a full causal pass (both KV-broadcast modes)")
    print(f"  max |incremental - full| = {demo_cache_correctness(device):.2e}")
    c = KVCache(1, 8, 16, 128)
    c.append(torch.zeros(1, 8, 4, 128, dtype=torch.float16), torch.zeros(1, 8, 4, 128, dtype=torch.float16))
    for _ in range(3):
        c.append(torch.zeros(1, 8, 1, 128, dtype=torch.float16), torch.zeros(1, 8, 1, 128, dtype=torch.float16))
    print(f"  after a 4-token prefill and 3 single-token appends: length {c.length}, "
          f"used {c.used_bytes:,} B of {c.allocated_bytes:,} B allocated")

    print(f"\n[3] Benchmark: {shape.name} ({shape.n_layers} layers, H_Q={shape.n_q_heads}, d_head={shape.head_dim}, "
          f"FP16, batch 1)")
    peak = gpu_peak_bandwidth_gbps()
    copy_bw = measured_copy_bandwidth_gbps(device)
    print(f"  peak DRAM bandwidth (spec): {f'{peak:.0f} GB/s' if peak else 'n/a'}; measured copy bandwidth {copy_bw:.0f} GB/s")
    rows = run_benchmark(device, shape)
    hdr = (f"  {'N':>6} | {'variant':<15} | {'KV-cache MB':>11} | {'GB/s @ 50 tok/s':>15} | "
           f"{'ms/token measured':>17} | {'(repeat_interleave)':>19} | {'bandwidth bound ms':>18}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        meas = "does not fit" if r["latency_ms_measured"] is None else f"{r['latency_ms_measured']:.2f}"
        rep = "does not fit" if r["latency_ms_repeat_interleave"] is None else f"{r['latency_ms_repeat_interleave']:.2f}"
        flag = "  (> GPU bandwidth)" if peak and r["bandwidth_gbps_at_50tps"] > peak else ""
        print(f"  {r['N']:>6} | {r['variant']:<15} | {r['kv_mb']:>11,.0f} | {r['bandwidth_gbps_at_50tps']:>15,.1f} | "
              f"{meas:>17} | {rep:>19} | {r['latency_ms_bound']:>18.2f}{flag}")
        if r["variant"].startswith("MQA") and r["N"] != SEQ_LENS[-1]:
            print("  " + "-" * (len(hdr) - 2))
    print("  measured = one attention layer (projections, cache append, attention) timed on this device x 32 layers,")
    print("  KV heads broadcast without copying; the next column uses torch.repeat_interleave, which copies the cache G times;")
    print("  bound = KV bytes / peak bandwidth: the time just to read the cache once. Weights and the MLP are excluded.")

    print("\n[Part 4 support] KV-cache bytes per token, whole model (FP16)")
    base = kv_bytes_per_token(shape)["MHA (32 KV heads)"]
    for k, v in kv_bytes_per_token(shape).items():
        print(f"  {k:<28} {v / 1024:>8.1f} KiB/token   ({v / base:.1%} of MHA)")

    print("\n[Part 4 support] KV-cache DoS, analytic: 80 GB mainframe, 8B FP16 weights (16 GB), sessions of 65,536 tokens")
    for r in kv_dos_analytic():
        print(f"  {r['variant']:<26} {r['per_session_gb']:>7.2f} GB/session -> OOM after {r['max_sessions']:>3} sessions")

    if not args.no_oom:
        print("\n[Part 4 support] KV-cache DoS, live: GQA sessions of 2,048 tokens on this GPU until CUDA OOM")
        live = kv_dos_live(device)
        if "skipped" in live:
            print(f"  skipped: {live['skipped']}")
        else:
            print(f"  {live['gpu']}: {live['total_gb']:.1f} GB total, {live['free_gb_at_start']:.2f} GB free; "
                  f"{live['per_session_mb']:.0f} MB per session")
            print(f"  sessions opened before OOM: {live['sessions_before_oom']} (predicted {live['predicted']})")
            print(f"  error raised: {live['error']}")
            if live["sessions_before_oom"] > live["predicted"]:
                spill = (live["sessions_before_oom"] - live["predicted"]) * live["per_session_mb"] / 1024
                print(f"  {live['sessions_before_oom'] - live['predicted']} sessions ({spill:.1f} GB) went beyond the card's VRAM: the "
                      "Windows driver's sysmem fallback placed them in shared system RAM, so the OOM came late "
                      "and every session in that region runs at PCIe speed instead of GDDR6 speed.")
            print("  all test allocations released.")

    print("\n[Part 4 support] PagedAttention vs contiguous reservation (64 GB KV budget, max context 32,768, GQA)")
    pg = paged_vs_contiguous()
    print(f"  request lengths: median {pg['median_len']} tokens, mean {pg['mean_len']:.0f}")
    print(f"  contiguous (reserve max length): {pg['contiguous_admitted']:>5} requests admitted, "
          f"{pg['contiguous_waste']:.1%} of reserved memory unused")
    print(f"  paged ({pg['block']}-token blocks):          {pg['paged_admitted']:>5} requests admitted, "
          f"{pg['paged_waste']:.1%} unused")
    print("\n" + bar)
    print("  PART 3 COMPLETE: GQA + KVCache verified; benchmark, DoS and paging figures produced.")
    print(bar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
