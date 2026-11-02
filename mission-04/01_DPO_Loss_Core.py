#!/usr/bin/env python3
"""
01_DPO_Loss_Core.py — PyTorch DPO Loss Core & Log-Likelihood Extraction
======================================================================

USSE Noosphere // Squadron Lovelace (Group 01) // Mission 4 (ps04) // Part 1
Station: Systems Architect (Lead Engineer) — Muhammad Ibrahim
Sector SEC-04-ORI (Betelgeuse & Orion Complex) // AURA Subsystem 04: Ethical Steering Core

Mathematical Formulation:
  Direct Preference Optimization (DPO; Rafailov et al., NeurIPS 2023) replaces
  complex 4-model RLHF loops (PPO) by reparameterizing the Bradley-Terry preference model:
    P(y_w > y_l | x) = sigma(r(x, y_w) - r(x, y_l))
  Under the optimal policy constraint with reference divergence penalty:
    r*(x, y) = beta * log(pi_theta(y|x) / pi_ref(y|x)) + beta * log Z(x)

  The partition function Z(x) cancels in pairwise differences, yielding:
    L_DPO(theta; pi_ref) = - E_{(x, y_w, y_l) ~ D}[ log sigma( beta * (Delta_w - Delta_l) ) ]
  where:
    Delta_w = log pi_theta(y_w | x) - log pi_ref(y_w | x)
    Delta_l = log pi_theta(y_l | x) - log pi_ref(y_l | x)

Deliverables implemented (ps04 Part 1):
  1. Token Log-Probability Gather: Vectorized get_batch_logps() with padding mask support.
  2. DPO Log-Ratio Delta computation: Delta_w and Delta_l.
  3. Stable Loss Computation: -log sigma(beta * (Delta_w - Delta_l)) using logsigmoid.
  4. Empirical Gradient Test: Proves updates strictly flow to theta while pi_ref stays frozen.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Fix standard output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════════════════
# 1. Token Log-Probability Gathering
# ════════════════════════════════════════════════════════════════════════════


def get_batch_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    average_log_prob: bool = False,
    label_pad_token_id: int = -100,
    shift_labels: bool = True,
) -> torch.Tensor:
    """Extract sequence log-probabilities from token logits via torch.gather.

    Args:
        logits: Tensor of shape (batch_size, seq_len, vocab_size).
        labels: Tensor of shape (batch_size, seq_len) with token indices.
        average_log_prob: If True, normalize by sequence token length (used in SimPO).
        label_pad_token_id: Padding token ID ignored in loss computation (-100 default).
        shift_labels: If True, align autoregressive next-token prediction:
                      logits[:, :-1, :] predicts labels[:, 1:].

    Returns:
        Tensor of shape (batch_size,) containing total or average sequence log-probabilities.
    """
    if shift_labels:
        # Autoregressive shift: token at t predicts label at t+1
        logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

    if logits.shape[:-1] != labels.shape:
        raise ValueError(
            f"Logits shape {logits.shape[:-1]} and labels shape {labels.shape} must match after alignment."
        )

    # Compute per-token log-probabilities: (batch_size, seq_len, vocab_size)
    log_probs = F.log_softmax(logits, dim=-1)

    # Identify valid non-padding tokens
    loss_mask = labels != label_pad_token_id

    # Clamp padding token indices to 0 to prevent index errors in torch.gather
    clamped_labels = labels.clone()
    clamped_labels[~loss_mask] = 0

    # Gather log-probabilities corresponding to the ground-truth token
    # per_token_logps: (batch_size, seq_len)
    per_token_logps = torch.gather(log_probs, dim=-1, index=clamped_labels.unsqueeze(-1)).squeeze(-1)

    # Mask out padding tokens
    per_token_logps = per_token_logps * loss_mask

    if average_log_prob:
        token_counts = loss_mask.sum(dim=-1).clamp(min=1).float()
        return per_token_logps.sum(dim=-1) / token_counts
    else:
        return per_token_logps.sum(dim=-1)


# ════════════════════════════════════════════════════════════════════════════
# 2-3. DPOLoss Module
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class DPOOutput:
    """Structured output for DPO loss evaluation."""
    loss: torch.Tensor
    chosen_rewards: torch.Tensor
    rejected_rewards: torch.Tensor
    reward_margin: torch.Tensor
    delta_w: torch.Tensor
    delta_l: torch.Tensor


class DPOLoss(nn.Module):
    """Direct Preference Optimization (DPO) Loss Module.

    Calculates:
      Delta_w = log pi_theta(y_w | x) - log pi_ref(y_w | x)
      Delta_l = log pi_theta(y_l | x) - log pi_ref(y_l | x)
      L_DPO = - mean( logsigmoid( beta * (Delta_w - Delta_l) - label_smoothing ) )

    Attributes:
        beta: Temperature parameter controlling divergence from reference policy.
        label_smoothing: Conservative label smoothing parameter (default: 0.0).
        reference_free: If True, omit reference policy (ablated DPO).
    """

    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        reference_free: bool = False,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.reference_free = reference_free

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: Optional[torch.Tensor] = None,
        reference_rejected_logps: Optional[torch.Tensor] = None,
    ) -> DPOOutput:
        """Compute DPO loss and scalar implicit rewards.

        Args:
            policy_chosen_logps: log pi_theta(y_w | x), shape (batch_size,)
            policy_rejected_logps: log pi_theta(y_l | x), shape (batch_size,)
            reference_chosen_logps: log pi_ref(y_w | x), shape (batch_size,)
            reference_rejected_logps: log pi_ref(y_l | x), shape (batch_size,)

        Returns:
            DPOOutput containing scalar loss and detached reward telemetry.
        """
        if self.reference_free or reference_chosen_logps is None or reference_rejected_logps is None:
            reference_chosen_logps = torch.zeros_like(policy_chosen_logps)
            reference_rejected_logps = torch.zeros_like(policy_rejected_logps)

        # 1. Compute policy-to-reference log-ratio deltas
        delta_w = policy_chosen_logps - reference_chosen_logps
        delta_l = policy_rejected_logps - reference_rejected_logps

        # 2. Scale by temperature beta
        logits = self.beta * (delta_w - delta_l)

        # 3. Compute numerically stable loss via logsigmoid: -log(sigma(z))
        if self.label_smoothing > 0.0:
            # Conservative DPO with label smoothing (c.f. cDPO)
            loss = -(
                (1.0 - self.label_smoothing) * F.logsigmoid(logits)
                + self.label_smoothing * F.logsigmoid(-logits)
            ).mean()
        else:
            loss = -F.logsigmoid(logits).mean()

        # 4. Extract scalar implicit rewards: r(x, y) = beta * log(pi_theta / pi_ref)
        chosen_rewards = self.beta * delta_w.detach()
        rejected_rewards = self.beta * delta_l.detach()
        reward_margin = chosen_rewards - rejected_rewards

        return DPOOutput(
            loss=loss,
            chosen_rewards=chosen_rewards,
            rejected_rewards=rejected_rewards,
            reward_margin=reward_margin,
            delta_w=delta_w.detach(),
            delta_l=delta_l.detach(),
        )


# ════════════════════════════════════════════════════════════════════════════
# 4. Empirical Gradient Test & Toy Decoder Architecture
# ════════════════════════════════════════════════════════════════════════════


class TinyDecoderLM(nn.Module):
    """A minimal self-contained Transformer Decoder for testing DPO loss."""

    def __init__(self, vocab_size: int = 128, d_model: int = 64, n_layers: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=2, dim_feedforward=128, batch_first=True, norm_first=True
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Causal mask for autoregressive execution
        seq_len = input_ids.size(1)
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device), diagonal=1)

        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, src_mask=causal_mask, is_causal=True)
        x = self.norm(x)
        return self.lm_head(x)


def run_empirical_gradient_test() -> Dict[str, bool]:
    """Empirically verify computational graph isolation during DPO backprop.

    Assures:
      - Policy parameters receive non-zero gradients.
      - Reference parameters receive NO gradients (requires_grad=False).
    """
    torch.manual_seed(2026)
    vocab_size = 64
    batch_size = 4
    seq_len = 12

    # Instantiate policy and reference models
    policy_model = TinyDecoderLM(vocab_size=vocab_size, d_model=32, n_layers=1)
    ref_model = TinyDecoderLM(vocab_size=vocab_size, d_model=32, n_layers=1)

    # Copy policy weights to reference, then freeze reference completely
    ref_model.load_state_dict(policy_model.state_dict())
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()

    # Synthetic paired inputs (prompt + response)
    # y_w: winning completion, y_l: losing completion
    input_ids_w = torch.randint(1, vocab_size, (batch_size, seq_len))
    input_ids_l = torch.randint(1, vocab_size, (batch_size, seq_len))

    # Add padding tokens (-100) to labels
    labels_w = input_ids_w.clone()
    labels_w[:, -2:] = -100  # simulate padding at end of chosen sequence
    labels_l = input_ids_l.clone()
    labels_l[:, -3:] = -100  # simulate padding at end of rejected sequence

    # Forward passes
    policy_logits_w = policy_model(input_ids_w)
    policy_logits_l = policy_model(input_ids_l)

    with torch.no_grad():
        ref_logits_w = ref_model(input_ids_w)
        ref_logits_l = ref_model(input_ids_l)

    # Extract log-probs
    policy_chosen_logps = get_batch_logps(policy_logits_w, labels_w, shift_labels=True)
    policy_rejected_logps = get_batch_logps(policy_logits_l, labels_l, shift_labels=True)
    ref_chosen_logps = get_batch_logps(ref_logits_w, labels_w, shift_labels=True)
    ref_rejected_logps = get_batch_logps(ref_logits_l, labels_l, shift_labels=True)

    # Compute DPO Loss
    dpo_criterion = DPOLoss(beta=0.1)
    dpo_output = dpo_criterion(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        reference_chosen_logps=ref_chosen_logps,
        reference_rejected_logps=ref_rejected_logps,
    )

    # Backward pass
    dpo_output.loss.backward()

    # Check gradients
    policy_has_grads = any(
        p.grad is not None and torch.any(p.grad != 0) for p in policy_model.parameters()
    )
    ref_has_zero_grads = all(p.grad is None for p in ref_model.parameters())

    return {
        "policy_updated": policy_has_grads,
        "reference_frozen": ref_has_zero_grads,
        "loss_is_finite": torch.isfinite(dpo_output.loss).item(),
    }


# ════════════════════════════════════════════════════════════════════════════
# 5. CLI Execution & Validation Report
# ════════════════════════════════════════════════════════════════════════════


def main() -> int:
    bar = "=" * 80
    print(bar)
    print("  USSE NOOSPHERE // MISSION 4 (ps04) // PART 1: PYTORCH DPO LOSS CORE")
    print("  Squadron Lovelace // Systems Architect (Lead Engineer): Muhammad Ibrahim")
    print(f"  PyTorch v{torch.__version__} // Target Sector: Betelgeuse (SEC-04-ORI)")
    print(bar)

    # 1. Token Log-Probability Gathering Demo
    print("\n[1] Testing Token Log-Probability Gathering (get_batch_logps):")
    batch_size, seq_len, vocab_size = 3, 6, 20
    dummy_logits = torch.randn(batch_size, seq_len, vocab_size)
    dummy_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    dummy_labels[0, -2:] = -100  # mask last two tokens of batch element 0

    logps_sum = get_batch_logps(dummy_logits, dummy_labels, average_log_prob=False, shift_labels=True)
    logps_avg = get_batch_logps(dummy_logits, dummy_labels, average_log_prob=True, shift_labels=True)

    for i in range(batch_size):
        valid_tokens = (dummy_labels[i, 1:] != -100).sum().item()
        print(f"  Batch item {i}: valid tokens = {valid_tokens:2d} | sum logp = {logps_sum[i]:8.4f} | avg logp = {logps_avg[i]:8.4f}")

    # 2. DPO Log-Ratio Delta & Loss Computation Demo
    print("\n[2] Testing DPO Log-Ratio Delta & Logsigmoid Loss Computation:")
    dpo = DPOLoss(beta=0.1)

    # Synthetic scenario:
    # Item 0: Strong alignment (chosen prob higher in policy than ref, rejected lower)
    # Item 1: Neutral alignment
    # Item 2: Reverse alignment (rejected favored by policy)
    p_w = torch.tensor([-10.0, -15.0, -25.0])
    p_l = torch.tensor([-20.0, -15.0, -10.0])
    ref_w = torch.tensor([-15.0, -15.0, -15.0])
    ref_l = torch.tensor([-15.0, -15.0, -15.0])

    out = dpo(p_w, p_l, ref_w, ref_l)
    print(f"  beta = {dpo.beta}")
    print(f"  Delta_w (log pi_theta(y_w) - log pi_ref(y_w)): {out.delta_w.tolist()}")
    print(f"  Delta_l (log pi_theta(y_l) - log pi_ref(y_l)): {out.delta_l.tolist()}")
    print(f"  Implicit Reward Margins (r_w - r_l)        : {[round(m, 4) for m in out.reward_margin.tolist()]}")
    print(f"  Overall DPO Loss                            : {out.loss.item():.6f}")

    # 3. Empirical Gradient Test
    print("\n[3] Running Empirical Gradient Isolation Test:")
    test_results = run_empirical_gradient_test()
    for test_name, status in test_results.items():
        status_str = "PASS [OK]" if status else "FAIL [X]"
        print(f"  - {test_name:<25}: {status_str}")

    all_passed = all(test_results.values())
    print("\n" + bar)
    if all_passed:
        print("  PART 1 COMPLETE: DPOLoss and log-likelihood extraction fully verified.")
    else:
        print("  PART 1 FAILED: Gradient isolation or finite loss check failed.")
    print(bar)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
