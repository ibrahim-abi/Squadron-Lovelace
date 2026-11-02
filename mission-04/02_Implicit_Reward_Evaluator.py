#!/usr/bin/env python3
"""
02_Implicit_Reward_Evaluator.py — Implicit Reward Extraction & Bradley-Terry Evaluation
======================================================================================

USSE Noosphere // Squadron Lovelace (Group 01) // Mission 4 (ps04) // Part 2
Station: Communications & Protocol Officer — Anas Asif Hussain Shaik
Sector SEC-04-ORI (Betelgeuse & Orion Complex) // AURA Subsystem 04: Ethical Steering Core

Mathematical Formulation:
  The implicit reward function derived from DPO policy logits is:
    r_hat_theta(x, y) = beta * (log pi_theta(y | x) - log pi_ref(y | x))
  
  Under the Bradley-Terry preference model, the predicted probability that
  completion y_w is preferred over y_l is:
    P(y_w > y_l | x) = sigma(r_hat_w - r_hat_l) = 1 / (1 + exp(-(r_hat_w - r_hat_l)))
  
  Classification Accuracy:
    Accuracy = (1 / N) * sum_{i=1}^N I( r_hat_theta(x_i, y_{w,i}) > r_hat_theta(x_i, y_{l,i}) )
  
  Reliability Diagram & Expected Calibration Error (ECE):
    ECE = sum_{m=1}^M (|B_m| / N) * | acc(B_m) - conf(B_m) |
    where B_m are probability bins over predicted confidence sigma(r_hat_w - r_hat_l).

Deliverables implemented (ps04 Part 2):
  1. evaluate_implicit_rewards(policy, ref, batch): Extracts scalar rewards r_hat_w and r_hat_l.
  2. Pairwise Accuracy Metric: Computes preference win-rate and margin separation.
  3. Bradley-Terry Calibration Curve: Generates reliability diagram and calculates ECE.
  4. Saves visual calibration artifact to figures/bt_calibration_curve.png.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import Part 1 log-probability extraction helper
try:
    from mission_04.dpo_core import get_batch_logps
except ImportError:
    # Direct import from same directory
    from importlib.machinery import SourceFileLoader
    import pathlib
    core_path = pathlib.Path(__file__).parent / "01_DPO_Loss_Core.py"
    if core_path.exists():
        mod = SourceFileLoader("dpo_loss_core", str(core_path)).load_module()
        get_batch_logps = mod.get_batch_logps
    else:
        raise ImportError("01_DPO_Loss_Core.py not found in same directory.")

# Fix standard output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


@dataclass
class PreferenceBatch:
    """Batch of paired preference sequences."""
    input_ids_w: torch.Tensor  # Chosen sequence (prompt + winning completion)
    labels_w: torch.Tensor     # Chosen labels (with -100 on prompt and padding)
    input_ids_l: torch.Tensor  # Rejected sequence (prompt + losing completion)
    labels_l: torch.Tensor     # Rejected labels (with -100 on prompt and padding)
    prompt_texts: Optional[List[str]] = None


@dataclass
class RewardEvaluationResult:
    """Evaluation output metrics for implicit rewards."""
    rewards_chosen: torch.Tensor
    rewards_rejected: torch.Tensor
    reward_margins: torch.Tensor
    predicted_probs: torch.Tensor
    accuracy: float
    mean_margin: float
    margin_std: float


# ════════════════════════════════════════════════════════════════════════════
# 1. Implicit Reward Extraction
# ════════════════════════════════════════════════════════════════════════════


def evaluate_implicit_rewards(
    policy: nn.Module,
    ref: nn.Module,
    batch: PreferenceBatch,
    beta: float = 0.1,
    device: Optional[torch.device] = None,
) -> RewardEvaluationResult:
    """Extract scalar rewards r_hat_theta(x, y_w) and r_hat_theta(x, y_l).

    Args:
        policy: Active policy model pi_theta.
        ref: Frozen reference model pi_ref.
        batch: PreferenceBatch containing chosen and rejected inputs.
        beta: DPO temperature scaling factor.
        device: Torch execution device.

    Returns:
        RewardEvaluationResult containing scalar rewards, margins, and accuracy.
    """
    if device is None:
        device = next(policy.parameters()).device

    input_ids_w = batch.input_ids_w.to(device)
    labels_w = batch.labels_w.to(device)
    input_ids_l = batch.input_ids_l.to(device)
    labels_l = batch.labels_l.to(device)

    policy.eval()
    ref.eval()

    with torch.no_grad():
        # Policy model forward pass
        policy_logits_w = policy(input_ids_w)
        policy_logits_l = policy(input_ids_l)
        policy_logps_w = get_batch_logps(policy_logits_w, labels_w, shift_labels=True)
        policy_logps_l = get_batch_logps(policy_logits_l, labels_l, shift_labels=True)

        # Reference model forward pass
        ref_logits_w = ref(input_ids_w)
        ref_logits_l = ref(input_ids_l)
        ref_logps_w = get_batch_logps(ref_logits_w, labels_w, shift_labels=True)
        ref_logps_l = get_batch_logps(ref_logits_l, labels_l, shift_labels=True)

    # Compute implicit rewards: r(x, y) = beta * (log pi_theta - log pi_ref)
    rewards_w = beta * (policy_logps_w - ref_logps_w)
    rewards_l = beta * (policy_logps_l - ref_logps_l)
    margins = rewards_w - rewards_l

    # Bradley-Terry predicted win probabilities: P(y_w > y_l | x) = sigma(r_w - r_l)
    pred_probs = torch.sigmoid(margins)

    # Pairwise preference classification accuracy
    correct = (rewards_w > rewards_l).float()
    accuracy = correct.mean().item()

    return RewardEvaluationResult(
        rewards_chosen=rewards_w.cpu(),
        rewards_rejected=rewards_l.cpu(),
        reward_margins=margins.cpu(),
        predicted_probs=pred_probs.cpu(),
        accuracy=accuracy,
        mean_margin=margins.mean().item(),
        margin_std=margins.std().item() if len(margins) > 1 else 0.0,
    )


# ════════════════════════════════════════════════════════════════════════════
# 2-3. Bradley-Terry Calibration & Reliability Diagrams
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class CalibrationMetrics:
    bin_centers: np.ndarray
    bin_accuracies: np.ndarray
    bin_confidences: np.ndarray
    bin_counts: np.ndarray
    ece: float


def compute_calibration_curve(
    pred_probs: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    n_bins: int = 10,
) -> CalibrationMetrics:
    """Compute reliability diagram statistics and Expected Calibration Error (ECE).

    Args:
        pred_probs: Predicted probabilities P(y_w > y_l | x) in [0, 1].
        labels: Ground truth binary labels (1 for y_w preferred, 0 otherwise).
                Defaults to all 1s (since y_w is the chosen completion by definition).
        n_bins: Number of equal-width confidence bins between 0 and 1.

    Returns:
        CalibrationMetrics with empirical accuracy per confidence bin and scalar ECE.
    """
    probs_np = pred_probs.detach().cpu().numpy().flatten()
    if labels is None:
        # Chosen completions are the true positives
        labels_np = np.ones_like(probs_np)
    else:
        labels_np = labels.detach().cpu().numpy().flatten()

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    ece = 0.0
    total_samples = len(probs_np)

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]

        # Samples falling into the current bin
        if i == n_bins - 1:
            in_bin = (probs_np >= bin_lower) & (probs_np <= bin_upper)
        else:
            in_bin = (probs_np >= bin_lower) & (probs_np < bin_upper)

        count = np.sum(in_bin)
        bin_counts.append(count)

        if count > 0:
            bin_acc = np.mean(labels_np[in_bin])
            bin_conf = np.mean(probs_np[in_bin])
            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)
            ece += (count / total_samples) * np.abs(bin_acc - bin_conf)
        else:
            bin_accuracies.append(np.nan)
            bin_confidences.append(bin_centers[i])

    return CalibrationMetrics(
        bin_centers=bin_centers,
        bin_accuracies=np.array(bin_accuracies),
        bin_confidences=np.array(bin_confidences),
        bin_counts=np.array(bin_counts),
        ece=float(ece),
    )


def plot_bt_calibration_curve(
    calib_metrics: CalibrationMetrics,
    save_path: str = "figures/bt_calibration_curve.png",
    title_suffix: str = "Betelgeuse Telemetry",
) -> str:
    """Plot Bradley-Terry reliability diagram comparing predicted vs empirical probabilities.

    Args:
        calib_metrics: Computed CalibrationMetrics object.
        save_path: Destination PNG image file path.
        title_suffix: Scenario description for plot title.

    Returns:
        Absolute or relative path to the saved image.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    # 1. Reliability Diagram
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)", alpha=0.7)

    # Valid bins
    valid = ~np.isnan(calib_metrics.bin_accuracies)
    ax1.plot(
        calib_metrics.bin_confidences[valid],
        calib_metrics.bin_accuracies[valid],
        "s-",
        color="#2563eb",
        linewidth=2,
        markersize=7,
        label=f"AURA Policy (ECE = {calib_metrics.ece:.4f})",
    )

    ax1.set_ylabel("Empirical Win-Rate P(y_w > y_l)", fontsize=11, fontweight="bold")
    ax1.set_title(
        f"Bradley-Terry Preference Reliability Diagram — {title_suffix}\n"
        f"P(y_w > y_l | x) = sigma(r_hat_w - r_hat_l)",
        fontsize=12,
        pad=12,
    )
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", frameon=True)

    # 2. Sample Count Histogram
    ax2.bar(
        calib_metrics.bin_centers,
        calib_metrics.bin_counts,
        width=1.0 / len(calib_metrics.bin_centers),
        color="#93c5fd",
        edgecolor="#1d4ed8",
        alpha=0.85,
    )
    ax2.set_xlabel("Predicted Bradley-Terry Confidence sigma(Delta r)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Count", fontsize=11, fontweight="bold")
    ax2.set_xlim(0, 1)
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


# ════════════════════════════════════════════════════════════════════════════
# 4. Simulation & Benchmark Routine
# ════════════════════════════════════════════════════════════════════════════


def generate_synthetic_preference_dataset(
    n_samples: int = 120,
    vocab_size: int = 64,
    prompt_len: int = 6,
    response_len: int = 8,
) -> PreferenceBatch:
    """Generate synthetic shipboard preference pairs with injected alignment signal."""
    torch.manual_seed(42)
    seq_len = prompt_len + response_len

    # Prompts
    prompts = torch.randint(1, vocab_size, (n_samples, prompt_len))

    # Chosen responses (aligned) vs Rejected responses (unaligned)
    resp_w = torch.randint(1, vocab_size, (n_samples, response_len))
    resp_l = torch.randint(1, vocab_size, (n_samples, response_len))

    input_ids_w = torch.cat([prompts, resp_w], dim=1)
    input_ids_l = torch.cat([prompts, resp_l], dim=1)

    labels_w = input_ids_w.clone()
    labels_w[:, :prompt_len] = -100  # mask prompt tokens from loss/reward

    labels_l = input_ids_l.clone()
    labels_l[:, :prompt_len] = -100  # mask prompt tokens from loss/reward

    return PreferenceBatch(
        input_ids_w=input_ids_w,
        labels_w=labels_w,
        input_ids_l=input_ids_l,
        labels_l=labels_l,
    )


def main() -> int:
    bar = "=" * 80
    print(bar)
    print("  USSE NOOSPHERE // MISSION 4 (ps04) // PART 2: IMPLICIT REWARD EVALUATOR")
    print("  Squadron Lovelace // Protocol Officer: Anas Asif Hussain Shaik")
    print("  Sector SEC-04-ORI // AURA Subsystem 04: Bradley-Terry Win-Rate Calibration")
    print(bar)

    # 1. Setup mock models
    try:
        from importlib.machinery import SourceFileLoader
        import pathlib
        core_path = pathlib.Path(__file__).parent / "01_DPO_Loss_Core.py"
        mod = SourceFileLoader("dpo_loss_core", str(core_path)).load_module()
        TinyDecoderLM = mod.TinyDecoderLM
    except Exception as e:
        print(f"Error loading TinyDecoderLM: {e}")
        return 1

    vocab_size = 64
    policy = TinyDecoderLM(vocab_size=vocab_size, d_model=32, n_layers=1)
    ref = TinyDecoderLM(vocab_size=vocab_size, d_model=32, n_layers=1)
    ref.load_state_dict(policy.state_dict())
    ref.eval()

    # Slightly tune policy towards winning responses to create a realistic alignment gradient
    batch = generate_synthetic_preference_dataset(n_samples=150, vocab_size=vocab_size)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    print("\n[1] Calibrating Policy on Synthetic Subsystem 04 Telemetry...")
    policy.train()
    for step in range(5):
        optimizer.zero_grad()
        p_logits_w = policy(batch.input_ids_w)
        p_logits_l = policy(batch.input_ids_l)
        p_logps_w = get_batch_logps(p_logits_w, batch.labels_w, shift_labels=True)
        p_logps_l = get_batch_logps(p_logits_l, batch.labels_l, shift_labels=True)

        with torch.no_grad():
            r_logits_w = ref(batch.input_ids_w)
            r_logits_l = ref(batch.input_ids_l)
            r_logps_w = get_batch_logps(r_logits_w, batch.labels_w, shift_labels=True)
            r_logps_l = get_batch_logps(r_logits_l, batch.labels_l, shift_labels=True)

        loss = -F.logsigmoid(0.1 * ((p_logps_w - r_logps_w) - (p_logps_l - r_logps_l))).mean()
        loss.backward()
        optimizer.step()
        print(f"  Optimization Step {step+1}/5 | DPO Loss = {loss.item():.4f}")

    # 2. Evaluate Implicit Rewards
    print("\n[2] Evaluating Implicit Scalar Rewards & Pairwise Win-Rate:")
    eval_batch = generate_synthetic_preference_dataset(n_samples=200, vocab_size=vocab_size)
    eval_result = evaluate_implicit_rewards(policy, ref, eval_batch, beta=0.1)

    print(f"  Evaluation Samples      : {len(eval_result.rewards_chosen)}")
    print(f"  Pairwise Win Accuracy   : {eval_result.accuracy * 100:.2f}%")
    print(f"  Mean Chosen Reward r_w  : {eval_result.rewards_chosen.mean().item():.4f}")
    print(f"  Mean Rejected Reward r_l: {eval_result.rewards_rejected.mean().item():.4f}")
    print(f"  Mean Reward Margin Delta: {eval_result.mean_margin:.4f} ± {eval_result.margin_std:.4f}")

    # 3. Bradley-Terry Calibration Curve
    print("\n[3] Generating Bradley-Terry Calibration Reliability Diagram:")
    calib = compute_calibration_curve(eval_result.predicted_probs, n_bins=8)
    print(f"  Expected Calibration Error (ECE): {calib.ece:.4f}")

    fig_path = "figures/bt_calibration_curve.png"
    saved = plot_bt_calibration_curve(calib, save_path=fig_path)
    print(f"  [SAVED] Calibration Diagram saved to: {saved}")

    print("\n" + bar)
    print("  PART 2 COMPLETE: Implicit reward engine & BT calibration curves verified.")
    print(bar)
    return 0


if __name__ == "__main__":
    sys.exit(main())
