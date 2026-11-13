"""
Mission 06 — Part 3: Data & Payload Specialist
MDP Trajectory Logger, Empirical Discounted Return & Transition Probability Engine

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-06-CAR (Eta Carinae & Homunculus Nebula)
Subsystem 06: Autonomous Warp State Machine & Deliberation Loop
Cadet Assigned: Areej Ilyas (Data & Payload Specialist)
Squadron: Squadron Lovelace
"""

import math
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any, Optional
import os
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TrajectoryStep:
    """Represents a single discrete transition step in an MDP trajectory tau."""
    step_idx: int
    state: str
    action: str
    reward: float
    next_state: str
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


class MDPTrajectoryLogger:
    """
    Records and monitors state-action-reward trajectories:
    tau = (s0, a0, r0, s1, a1, r1, ..., sT)
    """
    def __init__(self):
        self.trajectory: List[TrajectoryStep] = []

    def record_transition(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str,
        done: bool = False,
        info: Optional[Dict[str, Any]] = None
    ) -> TrajectoryStep:
        """Appends a transition tuple to the active episodic trajectory."""
        step_idx = len(self.trajectory)
        record = TrajectoryStep(
            step_idx=step_idx,
            state=state,
            action=action,
            reward=float(reward),
            next_state=next_state,
            done=done,
            info=info or {}
        )
        self.trajectory.append(record)
        return record

    def get_rewards(self) -> List[float]:
        """Returns the chronological sequence of episodic scalar rewards."""
        return [step.reward for step in self.trajectory]

    def clear(self):
        """Clears logged trajectory records."""
        self.trajectory.clear()


def compute_discounted_return(rewards: List[float], gamma: float = 0.95) -> float:
    """
    Calculates the empirical cumulative discounted return G_0 for a sequence of rewards:
    G_0 = sum_{t=0}^T (gamma^t * r_t)
    """
    discounted_sum = 0.0
    for t, r in enumerate(rewards):
        discounted_sum += (gamma ** t) * float(r)
    return discounted_sum


def compute_transition_probability_matrix(
    historical_transitions: List[Tuple[str, str, str]]
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Computes empirical transition probabilities P(s' | s, a) from historical jump data.
    P(s' | s, a) = Count(s, a, s') / sum_{s''} Count(s, a, s'')
    """
    transition_counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)

    for s, a, next_s in historical_transitions:
        transition_counts[(s, a)][next_s] += 1

    probabilities: Dict[Tuple[str, str], Dict[str, float]] = {}
    for (s, a), counter in transition_counts.items():
        total = sum(counter.values())
        probabilities[(s, a)] = {next_s: count / total for next_s, count in counter.items()}

    return probabilities


def plot_transition_heatmap(
    probabilities: Dict[Tuple[str, str], Dict[str, float]],
    save_path: str = "figures/transition_matrix_heatmap.png"
):
    """Visualizes transition probability matrix for the Homunculus Nebula sectors."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    pairs = list(probabilities.keys())
    all_next_states = sorted(list(set(ns for counts in probabilities.values() for ns in counts.keys())))
    
    matrix = np.zeros((len(pairs), len(all_next_states)))
    row_labels = [f"{s} + {a}" for s, a in pairs]
    
    for i, pair in enumerate(pairs):
        for j, ns in enumerate(all_next_states):
            matrix[i, j] = probabilities[pair].get(ns, 0.0)

    plt.figure(figsize=(10, 6))
    plt.imshow(matrix, cmap="Blues", aspect="auto", vmin=0.0, vmax=1.0)
    plt.colorbar(label="Empirical Probability P(s' | s, a)")
    plt.xticks(range(len(all_next_states)), all_next_states, rotation=45, ha="right")
    plt.yticks(range(len(pairs)), row_labels)
    plt.title("Homunculus Nebula Empirical Transition Probabilities P(s' | s, a)")
    plt.xlabel("Destination Sector State (s')")
    plt.ylabel("Source State + Navigation Action (s, a)")
    
    for i in range(len(pairs)):
        for j in range(len(all_next_states)):
            val = matrix[i, j]
            if val > 0:
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", color="black" if val < 0.6 else "white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Data Specialist] Transition heatmap saved to {save_path}")


# Historical jump telemetry in Sector SEC-06-CAR (Homunculus Nebula)
HISTORICAL_NEBULA_JUMPS = [
    ("Orbit_Stable", "navigate_warp_vector", "Dust_Shear_Zone"),
    ("Orbit_Stable", "navigate_warp_vector", "Dust_Shear_Zone"),
    ("Orbit_Stable", "navigate_warp_vector", "Escape_Corridor"),
    ("Dust_Shear_Zone", "adjust_plasma_shields", "Dust_Shear_Zone"),
    ("Dust_Shear_Zone", "adjust_plasma_shields", "Radiation_Pocket"),
    ("Dust_Shear_Zone", "query_sensor_array", "Escape_Corridor"),
    ("Dust_Shear_Zone", "query_sensor_array", "Escape_Corridor"),
    ("Radiation_Pocket", "adjust_plasma_shields", "Dust_Shear_Zone"),
    ("Radiation_Pocket", "adjust_plasma_shields", "Escape_Corridor"),
    ("Escape_Corridor", "navigate_warp_vector", "Interstellar_Vacuum"),
    ("Escape_Corridor", "navigate_warp_vector", "Interstellar_Vacuum"),
    ("Escape_Corridor", "navigate_warp_vector", "Escape_Corridor"),
]


if __name__ == "__main__":
    print("=== Subsystem 06: MDP Trajectory & Return Evaluator Simulation ===")
    logger = MDPTrajectoryLogger()

    # 1. Log test episode
    logger.record_transition("Orbit_Stable", "query_sensor_array", 10.0, "Dust_Shear_Zone")
    logger.record_transition("Dust_Shear_Zone", "adjust_plasma_shields", 0.0, "Dust_Shear_Zone")
    logger.record_transition("Dust_Shear_Zone", "navigate_warp_vector", -5.0, "Radiation_Pocket")
    logger.record_transition("Radiation_Pocket", "navigate_warp_vector", 20.0, "Escape_Corridor", done=True)

    test_rewards = logger.get_rewards()
    gamma_val = 0.95
    G0 = compute_discounted_return(test_rewards, gamma=gamma_val)
    print(f"Logged Rewards: {test_rewards}")
    print(f"Calculated Discounted Cumulative Return G_0 (gamma={gamma_val}): {G0:.4f}")

    # 2. Transition matrix
    trans_matrix = compute_transition_probability_matrix(HISTORICAL_NEBULA_JUMPS)
    print("\nEmpirical Transition Probabilities P(s' | s, a):")
    for (src, act), dsts in trans_matrix.items():
        dst_str = ", ".join(f"{dst}: {prob:.2f}" for dst, prob in dsts.items())
        print(f"  P(s' | {src}, {act}) => [{dst_str}]")

    # Generate plot
    plot_transition_heatmap(trans_matrix)
