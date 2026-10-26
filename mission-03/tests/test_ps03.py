"""Tests for the three ps03 scripts (run with `python -m pytest` from the mission-03 folder).

The deliverable file names start with digits, so they cannot be imported normally; they are loaded by path.
"""

import contextlib
import importlib.util
import inspect
import io
import math
import pathlib
import sys
import tempfile

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


mha_core = _load("mha_core", "01_MultiHead_Attention_Core.py")
rope = _load("rope_engine", "02_RoPE_Positional_Engine.py")
gqa = _load("gqa_kvcache", "03_GQA_and_KVCache_Benchmark.py")


# ── Part 1: MultiHeadSelfAttention ─────────────────────────────────────────────


def test_projection_and_head_shapes():
    m = mha_core.MultiHeadSelfAttention(d_model=128, num_heads=8)
    for w in (m.W_Q, m.W_K, m.W_V, m.W_O):
        assert tuple(w.shape) == (128, 128)
    x = torch.randn(3, 10, 128)
    assert tuple(m.split_heads(x @ m.W_Q).shape) == (3, 8, 10, 16)
    out, w = m(x)
    assert tuple(out.shape) == (3, 10, 128) and tuple(w.shape) == (3, 8, 10, 10)
    assert torch.allclose(m.merge_heads(m.split_heads(x)), x)


def test_no_high_level_attention_wrappers_in_the_module():
    src = inspect.getsource(mha_core.MultiHeadSelfAttention) + inspect.getsource(mha_core.scaled_dot_product_attention)
    assert "MultiheadAttention" not in src
    assert "functional.scaled_dot_product_attention" not in src and "F.scaled_dot_product_attention" not in src


def test_causal_mask_blocks_the_future():
    torch.manual_seed(0)
    m = mha_core.MultiHeadSelfAttention(d_model=64, num_heads=4)
    x = torch.randn(2, 9, 64)
    out, w = m(x, causal=True)
    upper = torch.triu(torch.ones(9, 9, dtype=torch.bool), diagonal=1)
    assert w[..., upper].abs().max() == 0
    assert torch.allclose(w.sum(-1), torch.ones(2, 4, 9), atol=1e-6)
    x2 = x.clone()
    x2[:, 5:] = torch.randn_like(x2[:, 5:])
    assert torch.allclose(m(x2)[0][:, :5], out[:, :5], atol=1e-6)
    mask = mha_core.causal_mask(4)
    assert mask[0, 1] == float("-inf") and mask[1, 0] == 0 and mask[3, 3] == 0


def test_matches_pytorch_reference():
    torch.manual_seed(1)
    m = mha_core.MultiHeadSelfAttention(d_model=96, num_heads=6)
    x = torch.randn(2, 11, 96)
    for causal in (True, False):
        assert torch.allclose(m(x, causal=causal)[0], mha_core.reference_attention(x, m, causal), atol=1e-5)


def test_padding_mask():
    torch.manual_seed(2)
    m = mha_core.MultiHeadSelfAttention(d_model=32, num_heads=4)
    x = torch.randn(1, 6, 32)
    pad = torch.tensor([[True, True, True, True, False, False]])
    _, w = m(x, causal=False, padding_mask=pad)
    assert w[..., 4:].abs().max() == 0


@pytest.mark.parametrize("d_k", [16, 64, 128])
def test_variance_preservation(d_k):
    r = mha_core.variance_preservation(d_k)
    assert abs(r.unscaled_var / d_k - 1) < 0.05          # Var(q.k) ~= d_k
    assert abs(r.scaled_var - 1.0) < 0.05                 # Var(q.k / sqrt(d_k)) ~= 1
    assert abs(r.scaled_mean) < 0.02
    assert r.max_softmax_scaled < r.max_softmax_unscaled  # scaling prevents softmax saturation


@pytest.mark.parametrize("causal", [False, True])
def test_tiled_flash_attention_is_exact_and_counts_io(causal):
    torch.manual_seed(3)
    N, d = 300, 32
    q, k, v = (torch.randn(N, d, dtype=torch.float64) for _ in range(3))
    mask = mha_core.causal_mask(N, dtype=torch.float64) if causal else None
    ref, _ = mha_core.scaled_dot_product_attention(q, k, v, mask)
    io = mha_core.IOCount()
    out = mha_core.flash_attention_tiled(q, k, v, block_r=64, block_c=48, causal=causal, io=io)
    assert torch.allclose(out, ref, atol=1e-10)
    if not causal:
        assert io.total == mha_core.flash_io_formula(N, d, 64)
        assert io.total < mha_core.standard_attention_io(N, d).total


# ── Part 2: RoPE ───────────────────────────────────────────────────────────────


def test_frequency_formula():
    th = rope.rope_frequencies(8, base=10000.0)
    expected = [10000 ** (-2 * (i - 1) / 8) for i in range(1, 5)]
    assert torch.allclose(th, torch.tensor(expected, dtype=torch.float64))


def test_apply_rope_is_the_rotation_matrix_and_preserves_norm():
    torch.manual_seed(4)
    x = torch.randn(1, 2, 12, 16)
    table = rope.precompute_freqs_cis(16, 12)
    y = rope.apply_rope(x, table)
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5)
    for m in (0, 3, 11):
        R = rope.rope_rotation_matrix(16, m)
        assert torch.allclose(y[0, 1, m].double(), R @ x[0, 1, m].double(), atol=1e-5)
    assert torch.allclose(y[..., 0, :], x[..., 0, :])        # position 0 is not rotated


def test_relative_invariance():
    r = rope.invariance_trials(trials=300, dim=64, max_pos=4096)
    assert r["max_rel_diff"] < 1e-4


def test_score_depends_only_on_offset():
    torch.manual_seed(5)
    table = rope.precompute_freqs_cis(32, 200)
    q, k = torch.randn(32), torch.randn(32)
    scores = [rope.score_at(q, k, m, m + 7, table) for m in (0, 50, 120)]
    assert max(scores) - min(scores) < 1e-4


def test_apply_rope_validates_shapes():
    with pytest.raises(ValueError):
        rope.apply_rope(torch.randn(2, 5, 7), rope.precompute_freqs_cis(8, 5))
    with pytest.raises(ValueError):
        rope.apply_rope(torch.randn(2, 5, 8), rope.precompute_freqs_cis(8, 6))


def test_decay_and_alibi():
    r = torch.tensor([0, 1000])
    s = rope.rope_score_vs_distance(128, r)
    assert s[0] == pytest.approx(1.0) and abs(s[1]) < 0.5
    assert torch.allclose(rope.alibi_slopes(8), torch.tensor([2.0 ** -h for h in range(1, 9)], dtype=torch.float64))
    w = rope.attention_weights_alibi(16, 0.5)
    assert torch.allclose(w.sum(-1), torch.ones(16, dtype=torch.float64))
    assert (w[-1, :-1].diff() > 0).all()                   # weight grows towards the query (monotone)


def test_extrapolation_metrics():
    assert rope.unseen_angle_fraction(128, 4096, 4096) == 0.0
    assert rope.unseen_angle_fraction(128, 4096, 16384) > 0.1
    assert rope.unseen_angle_fraction(128, 4096, 16384, position_scale=0.25) == 0.0


def test_figures_are_written():
    with tempfile.TemporaryDirectory() as tmp:
        paths = rope.make_figures(tmp)
        assert len(paths) == 3 and all(pathlib.Path(p).stat().st_size > 10_000 for p in paths)


# ── Part 3: GQA and KV-cache ───────────────────────────────────────────────────


def test_gqa_grouping_shapes():
    layer = gqa.GQAAttention(d_model=64, n_q_heads=8, n_kv_heads=2)
    assert layer.group == 4 and tuple(layer.w_k.weight.shape) == (16, 64)
    assert tuple(layer(torch.randn(2, 5, 64)).shape) == (2, 5, 64)
    with pytest.raises(ValueError):
        gqa.GQAAttention(d_model=64, n_q_heads=8, n_kv_heads=3)


def test_gqa_with_all_kv_heads_equals_mha():
    torch.manual_seed(6)
    layer = gqa.GQAAttention(d_model=32, n_q_heads=4, n_kv_heads=4)
    x = torch.randn(1, 6, 32)
    q = layer.w_q(x).view(1, 6, 4, 8).transpose(1, 2)
    k = layer.w_k(x).view(1, 6, 4, 8).transpose(1, 2)
    v = layer.w_v(x).view(1, 6, 4, 8).transpose(1, 2)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = layer.w_o(ref.transpose(1, 2).reshape(1, 6, 32))
    assert torch.allclose(layer(x), ref, atol=1e-5)


def test_incremental_decoding_equals_full_pass():
    assert gqa.demo_cache_correctness("cpu") < 1e-5


def test_kv_cache_appends_and_overflows():
    c = gqa.KVCache(1, 2, 5, 4, dtype=torch.float32)
    k, v = c.append(torch.ones(1, 2, 3, 4), torch.ones(1, 2, 3, 4))
    assert c.length == 3 and tuple(k.shape) == (1, 2, 3, 4)
    k, v = c.append(2 * torch.ones(1, 2, 1, 4), 2 * torch.ones(1, 2, 1, 4))
    assert c.length == 4 and k[0, 0, 3, 0] == 2 and k[0, 0, 0, 0] == 1
    assert c.used_bytes == 2 * 2 * 4 * 4 * 4
    c.append(torch.ones(1, 2, 1, 4), torch.ones(1, 2, 1, 4))
    with pytest.raises(RuntimeError):
        c.append(torch.ones(1, 2, 1, 4), torch.ones(1, 2, 1, 4))


def test_memory_and_bandwidth_formulas():
    s = gqa.ModelShape()
    mb = {name: gqa.kv_cache_bytes(s, h, 32768) / gqa.MB for name, h in gqa.VARIANTS.items()}
    assert mb == {"MHA 32:32": 16384, "GQA 4:1 (32:8)": 4096, "MQA 32:1": 512}
    assert gqa.bandwidth_gbps(gqa.kv_cache_bytes(s, 32, 32768)) == pytest.approx(800.0)
    # agrees with the course portal's KV-cache lab: 4 * L * H_KV * d * N * B bytes
    assert gqa.kv_cache_bytes(s, 8, 8192, batch=4) == 4 * 32 * 8 * 128 * 8192 * 4


def test_security_simulations():
    per = gqa.kv_bytes_per_token()
    assert per["MHA (32 KV heads)"] / per["MLA (d_c = 512 + 64 RoPE)"] == pytest.approx(8192 / 576)
    dos = {r["variant"]: r["max_sessions"] for r in gqa.kv_dos_analytic()}
    assert dos["MHA 32:32"] == 2 and dos["GQA 4:1 (32:8)"] == 8 and dos["MQA 32:1"] == 64
    pg = gqa.paged_vs_contiguous()
    assert pg["paged_admitted"] > 5 * pg["contiguous_admitted"]
    assert pg["paged_waste"] < 0.02 < 0.5 < pg["contiguous_waste"]


# ── scripts run end to end ─────────────────────────────────────────────────────


@pytest.mark.parametrize("module", [mha_core, rope])
def test_scripts_run(module):
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        assert module.main() == 0
    assert "COMPLETE" in buf.getvalue()
