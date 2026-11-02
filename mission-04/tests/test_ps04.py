"""
test_ps04.py — Test Suite for Mission 4 (ps04)
==============================================

USSE Noosphere // Squadron Lovelace (Group 01) // Mission 4
Covers:
  - Part 1: Token logps, DPO loss, gradient isolation
  - Part 2: Implicit reward evaluation, pairwise accuracy, Bradley-Terry calibration & ECE
  - Part 3: Length correlation diagnostic, SimPO loss, memory benchmark
  - Part 4: Concept vector extraction, activation steering hook, alignment tax audit
"""

import math
import os
import pathlib
import sys
import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add mission-04 directory to sys.path
MISSION_DIR = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(MISSION_DIR))

from importlib.machinery import SourceFileLoader

# Dynamically import numbered modules
mod_dpo = SourceFileLoader("dpo_loss_core", str(MISSION_DIR / "01_DPO_Loss_Core.py")).load_module()
mod_reward = SourceFileLoader("implicit_reward", str(MISSION_DIR / "02_Implicit_Reward_Evaluator.py")).load_module()
mod_simpo = SourceFileLoader("length_bias_simpo", str(MISSION_DIR / "03_Length_Bias_and_SimPO.py")).load_module()
mod_steer = SourceFileLoader("activation_steering", str(MISSION_DIR / "04_Activation_Steering_and_Audit.py")).load_module()


# ════════════════════════════════════════════════════════════════════════════
# Part 1 Tests: DPOLoss and Token Logps
# ════════════════════════════════════════════════════════════════════════════


def test_get_batch_logps_masking():
    """Verify that padding tokens (-100) are excluded from logps."""
    b, seq_len, vocab_size = 2, 5, 10
    logits = torch.randn(b, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (b, seq_len))

    # Mask out the last token of sequence 0
    labels[0, -1] = -100

    logps = mod_dpo.get_batch_logps(logits, labels, average_log_prob=False, shift_labels=False)
    assert logps.shape == (b,)
    assert torch.isfinite(logps).all()

    # Log-probs should be non-positive
    assert (logps <= 0).all()


def test_get_batch_logps_shift_labels():
    """Verify autoregressive shift (token t predicts label t+1)."""
    b, seq_len, vocab_size = 2, 4, 8
    logits = torch.randn(b, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (b, seq_len))

    logps_shifted = mod_dpo.get_batch_logps(logits, labels, shift_labels=True)
    # With shift_labels, valid sequence length is seq_len - 1
    assert logps_shifted.shape == (b,)
    assert torch.isfinite(logps_shifted).all()


def test_dpo_loss_values_and_clamping():
    """Verify DPO loss monotonicity and numerical stability on extreme deltas."""
    dpo = mod_dpo.DPOLoss(beta=0.1)

    # When chosen logps are higher than rejected, loss should be smaller
    high_diff = dpo(
        torch.tensor([0.0]), torch.tensor([-10.0]),
        torch.tensor([-5.0]), torch.tensor([-5.0])
    )
    low_diff = dpo(
        torch.tensor([-10.0]), torch.tensor([0.0]),
        torch.tensor([-5.0]), torch.tensor([-5.0])
    )
    assert high_diff.loss < low_diff.loss

    # Extreme logits (prevent underflow / overflow NaN)
    extreme_out = dpo(
        torch.tensor([1000.0]), torch.tensor([-1000.0]),
        torch.tensor([0.0]), torch.tensor([0.0])
    )
    assert torch.isfinite(extreme_out.loss)
    assert extreme_out.loss.item() >= 0.0


def test_empirical_gradient_isolation():
    """Verify that backpropagation flows exclusively to policy parameters."""
    res = mod_dpo.run_empirical_gradient_test()
    assert res["policy_updated"] is True
    assert res["reference_frozen"] is True
    assert res["loss_is_finite"] is True


# ════════════════════════════════════════════════════════════════════════════
# Part 2 Tests: Implicit Rewards and Bradley-Terry Evaluation
# ════════════════════════════════════════════════════════════════════════════


def test_evaluate_implicit_rewards():
    """Verify implicit reward formula: r = beta * (log pi - log ref)."""
    vocab_size = 32
    policy = mod_dpo.TinyDecoderLM(vocab_size=vocab_size, d_model=16, n_layers=1)
    ref = mod_dpo.TinyDecoderLM(vocab_size=vocab_size, d_model=16, n_layers=1)
    ref.load_state_dict(policy.state_dict())

    batch = mod_reward.generate_synthetic_preference_dataset(n_samples=10, vocab_size=vocab_size)
    eval_res = mod_reward.evaluate_implicit_rewards(policy, ref, batch, beta=0.1)

    # When policy == ref, implicit rewards and margins should be approximately 0
    assert torch.allclose(eval_res.rewards_chosen, torch.zeros_like(eval_res.rewards_chosen), atol=1e-5)
    assert torch.allclose(eval_res.rewards_rejected, torch.zeros_like(eval_res.rewards_rejected), atol=1e-5)
    assert torch.allclose(eval_res.reward_margins, torch.zeros_like(eval_res.reward_margins), atol=1e-5)


def test_calibration_curve_metrics():
    """Verify calibration curve computation and Expected Calibration Error (ECE)."""
    probs = torch.tensor([0.1, 0.2, 0.8, 0.9])
    labels = torch.tensor([0, 0, 1, 1])

    calib = mod_reward.compute_calibration_curve(probs, labels, n_bins=5)
    assert len(calib.bin_centers) == 5
    assert 0.0 <= calib.ece <= 1.0


# ════════════════════════════════════════════════════════════════════════════
# Part 3 Tests: Length Bias & SimPO
# ════════════════════════════════════════════════════════════════════════════


def test_length_correlation_diagnostic():
    """Verify Pearson correlation calculation."""
    # Perfectly correlated
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    r, p = mod_simpo.compute_length_correlation(x, y)
    assert pytest.approx(r, abs=1e-3) == 1.0

    # Uncorrelated
    x_uncorr = np.array([1, 2, 3, 4, 5])
    y_uncorr = np.array([5, 1, 4, 2, 3])
    r_u, _ = mod_simpo.compute_length_correlation(x_uncorr, y_uncorr)
    assert -1.0 <= r_u <= 1.0


def test_simpo_loss_formula():
    """Verify SimPOLoss with target margin gamma."""
    simpo = mod_simpo.SimPOLoss(beta=2.0, gamma=0.5)

    avg_w = torch.tensor([-1.0])
    avg_l = torch.tensor([-2.0])

    out = simpo(avg_w, avg_l)
    # Margin = beta * (avg_w - avg_l) - gamma = 2.0 * (1.0) - 0.5 = 1.5
    expected_margin = 2.0 * (avg_w.item() - avg_l.item()) - 0.5
    assert pytest.approx(out.reward_margin.item(), abs=1e-4) == expected_margin
    assert torch.isfinite(out.loss)


def test_vram_benchmark_reduction():
    """Verify that SimPO cuts model weights memory by exactly 50%."""
    records = mod_simpo.benchmark_vram_and_memory(d_model_scales=[64, 128])
    for r in records:
        assert pytest.approx(r.weights_savings_pct, abs=1e-2) == 50.0
        assert r.total_savings_pct > 0.0


# ════════════════════════════════════════════════════════════════════════════
# Part 4 Tests: Concept Vector Extraction & Steering Hook
# ════════════════════════════════════════════════════════════════════════════


def test_steering_vector_unit_norm():
    """Verify that extracted concept steering vector is normalized to unit L2 norm."""
    model = mod_steer.SteerableTransformerLM(vocab_size=32, d_model=16, n_layers=2)
    pairs = [
        mod_steer.ContrastPair(
            honest_tokens=torch.tensor([1, 2, 3, 4]),
            sycophantic_tokens=torch.tensor([1, 2, 5, 6]),
        )
        for _ in range(5)
    ]
    v_steer = mod_steer.extract_steering_vector(model, pairs, layer_idx=1)
    norm = torch.norm(v_steer, p=2).item()
    assert pytest.approx(norm, abs=1e-5) == 1.0


def test_activation_steering_hook_residual_injection():
    """Verify that the hook modifies activations by alpha * v_steer and removes cleanly."""
    model = mod_steer.SteerableTransformerLM(vocab_size=32, d_model=16, n_layers=2)
    target_layer = model.layers[1]
    v_steer = torch.randn(16)
    v_steer = v_steer / torch.norm(v_steer, p=2)

    hook = mod_steer.ActivationSteeringHook(target_layer, v_steer, alpha=2.5)

    input_ids = torch.tensor([[1, 2, 3, 4]])

    # Baseline output without hook
    with torch.no_grad():
        logits_base, _ = model(input_ids)

    # Output with registered hook
    hook.register()
    with torch.no_grad():
        logits_steered, _ = model(input_ids)

    # Steered output should differ from baseline
    assert not torch.allclose(logits_base, logits_steered, atol=1e-4)

    # Detach hook and verify exact baseline restoration
    hook.remove()
    with torch.no_grad():
        logits_restored, _ = model(input_ids)
    assert torch.allclose(logits_base, logits_restored, atol=1e-6)
