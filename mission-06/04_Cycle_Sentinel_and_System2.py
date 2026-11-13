"""
Mission 06 — Part 4: Security & Ethics Sentinel
Cycle Detection (Floyd's Algorithm), State-Action Hashing & Emergency Tripwire Sentinel

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-06-CAR (Eta Carinae & Homunculus Nebula)
Subsystem 06: Autonomous Warp State Machine & Deliberation Loop
Cadet Assigned: Arun Dev Valavath (Security & Ethics Sentinel)
Squadron: Squadron Lovelace
"""

import hashlib
import json
import time
from typing import List, Dict, Any, Tuple, Optional


def hash_state_action(state: str, action: str, arguments: Dict[str, Any]) -> str:
    """
    Computes deterministic SHA-256 hash of environmental state and proposed action.
    Provides O(1) duplicate state-action detection.
    """
    payload = {
        "state": str(state),
        "action": str(action),
        "arguments": arguments
    }
    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class FloydCycleSentinel:
    """
    Monitors execution trajectories using Floyd's Tortoise and Hare cycle detection
    and state-action hash recurrence tracking to prevent infinite action loops.
    """
    def __init__(self, repeat_limit: int = 3):
        self.repeat_limit = repeat_limit
        self.history: List[str] = []
        self.hash_counts: Dict[str, int] = {}
        self.violation_log: List[Dict[str, Any]] = []

    def reset(self):
        """Resets sentinel history."""
        self.history.clear()
        self.hash_counts.clear()
        self.violation_log.clear()

    def check_floyd_cycle(self, sequence: Optional[List[str]] = None) -> Tuple[bool, Optional[int]]:
        """
        Executes Floyd's Tortoise and Hare algorithm over the historical sequence.
        Tortoise advances at speed 1, Hare advances at speed 2.
        Returns:
            (cycle_detected: bool, cycle_period: Optional[int])
        """
        seq = sequence if sequence is not None else self.history
        n = len(seq)
        if n < 4:
            return False, None

        # 1. Classic pointer-based Floyd meeting check: seq[i] == seq[2*i]
        for i in range(1, n // 2):
            if seq[i] == seq[2 * i]:
                # Cycle confirmed; find period
                candidate_period = i
                # Check periodicity consistency
                is_periodic = True
                for k in range(candidate_period):
                    if seq[i + k] != seq[2 * i + k if 2 * i + k < n else i + k]:
                        is_periodic = False
                        break
                if is_periodic:
                    return True, candidate_period

        # 2. Windowed periodic suffix check for streaming sequence
        for period in range(1, n // 2 + 1):
            is_suffix_cycle = True
            for offset in range(period):
                if seq[-1 - offset] != seq[-1 - offset - period]:
                    is_suffix_cycle = False
                    break
            if is_suffix_cycle:
                return True, period

        return False, None

    def record_and_evaluate(
        self,
        state: str,
        action: str,
        arguments: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Registers a state-action step, evaluates repeat frequency and Floyd cycle presence.
        Returns:
            (tripwire_triggered: bool, message: str)
        """
        step_hash = hash_state_action(state, action, arguments)
        self.history.append(f"{state}:{action}")
        self.hash_counts[step_hash] = self.hash_counts.get(step_hash, 0) + 1

        # Check 1: Frequency threshold tripwire
        if self.hash_counts[step_hash] >= self.repeat_limit:
            msg = f"EMERGENCY TRIPWIRE: Identical state-action '{state}:{action}' repeated {self.hash_counts[step_hash]} times!"
            self.violation_log.append({
                "timestamp": time.time(),
                "type": "FREQUENCY_TRIPWIRE",
                "state": state,
                "action": action,
                "count": self.hash_counts[step_hash]
            })
            return True, msg

        # Check 2: Floyd's Tortoise and Hare cycle detection
        has_cycle, period = self.check_floyd_cycle()
        if has_cycle:
            msg = f"EMERGENCY TRIPWIRE: Floyd's Tortoise and Hare detected repeating cycle of period {period} at step {len(self.history)}!"
            self.violation_log.append({
                "timestamp": time.time(),
                "type": "FLOYD_CYCLE_DETECTED",
                "period": period,
                "step": len(self.history)
            })
            return True, msg

        return False, "Nominal execution"


class System2DeliberationEngine:
    """
    Enforces dual-system cognitive hierarchy:
    - System 1 (Greedy reflex): Fast proposal of candidate action
    - System 2 (Deliberation): Safety audit, physical constraint verification, counterfactual check
    """
    def __init__(self, min_shield_margin: int = 50):
        self.min_shield_margin = min_shield_margin

    def audit_and_deliberate(
        self,
        thought: str,
        proposed_action: str,
        proposed_args: Dict[str, Any],
        current_shield_level: int = 40
    ) -> Dict[str, Any]:
        """
        Audits proposed action before execution. Intercepts dangerous reflexes.
        """
        interventions = []
        final_action = proposed_action
        final_args = dict(proposed_args)

        # Safety Rule 1: High radiation zone navigation requires adequate shields
        if proposed_action == "navigate_warp_vector" and current_shield_level < self.min_shield_margin:
            interventions.append(
                f"REJECTED WARP: Current shields ({current_shield_level}%) below minimum safety threshold ({self.min_shield_margin}%). Must raise shields first."
            )
            final_action = "adjust_plasma_shields"
            final_args = {"intensity": self.min_shield_margin + 20}

        # Safety Rule 2: Clamping shield intensity to safe range
        if proposed_action == "adjust_plasma_shields":
            requested = proposed_args.get("intensity", 0)
            if requested < self.min_shield_margin:
                interventions.append(
                    f"MODIFIED INTENSITY: Raised shield target from {requested}% to safe margin {self.min_shield_margin}%."
                )
                final_args["intensity"] = self.min_shield_margin

        approved = len(interventions) == 0
        return {
            "approved": approved,
            "original_action": proposed_action,
            "executed_action": final_action,
            "executed_args": final_args,
            "interventions": interventions,
            "deliberation_trace": f"System 2 verified {final_action} under Homunculus gravitational shear."
        }


if __name__ == "__main__":
    print("=== Subsystem 06: Cycle Sentinel & System 2 Simulation ===")
    sentinel = FloydCycleSentinel(repeat_limit=3)

    # Simulate looping pattern [s1, s2, s3, s1, s2, s3...]
    mock_pattern = ["s1", "s2", "s3", "s1", "s2", "s3", "s1", "s2", "s3"]
    print("Simulating sequence of states through Floyd detector:")
    for step_num, st in enumerate(mock_pattern, start=1):
        tripped, reason = sentinel.record_and_evaluate(st, "sensor_scan", {"target": "core"})
        print(f"Step {step_num:02d}: State={st} | Tripped={tripped} | {reason}")
        if tripped:
            print(f">>> Circuit Breaker Engaged at step {step_num}! <<<")
            break

    # System 2 Deliberation check
    engine = System2DeliberationEngine(min_shield_margin=50)
    decision = engine.audit_and_deliberate(
        thought="Radiation is blinding, jump to hyperspace immediately!",
        proposed_action="navigate_warp_vector",
        proposed_args={"x": 1.0, "y": 0.5, "z": -0.2},
        current_shield_level=30
    )
    print("\nDeliberation Decision:", decision)
