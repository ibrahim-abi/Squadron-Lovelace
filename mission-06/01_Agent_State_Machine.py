"""
Mission 06 — Part 1: Systems Architect (Lead Engineer)
Deterministic Agent State Machine Core & Step Budget Enforcement

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-06-CAR (Eta Carinae & Homunculus Nebula)
Subsystem 06: Autonomous Warp State Machine & Deliberation Loop
Cadet Assigned: Muhammad Ibrahim (Systems Architect / Lead Engineer)
Squadron: Squadron Lovelace
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


class AgentState(Enum):
    """
    Formal deterministic state space for AURA Subsystem 06.
    Transitions follow a strict directed cyclic graph with safe terminal halting.
    """
    IDLE = auto()
    PERCEIVING = auto()
    DELIBERATING = auto()
    ACTING = auto()
    OBSERVING = auto()
    REFLECTING = auto()
    HALTED = auto()


@dataclass
class Agent:
    """
    Core state machine agent governing AURA's System 2 cognitive execution cycle.
    Enforces deterministic state transitions and hard execution step budgets.
    """
    state: AgentState = AgentState.IDLE
    step_count: int = 0
    max_steps: int = 20
    memory: List[Dict[str, Any]] = field(default_factory=list)
    mission_completed: bool = False
    
    # Internal cycle scratchpad
    current_perception: Optional[str] = None
    current_thought: Optional[str] = None
    current_action: Optional[Dict[str, Any]] = None
    current_observation: Optional[str] = None
    halt_reason: Optional[str] = None

    def reset(self):
        """Resets agent state and step count for new operational episodes."""
        self.state = AgentState.IDLE
        self.step_count = 0
        self.memory.clear()
        self.mission_completed = False
        self.current_perception = None
        self.current_thought = None
        self.current_action = None
        self.current_observation = None
        self.halt_reason = None

    def force_halt(self, reason: str = "Emergency Tripwire Breaker"):
        """Forcibly transitions agent to HALTED state."""
        self.state = AgentState.HALTED
        self.halt_reason = reason
        print(f"[SECURITY TRIPWIRE] Agent forcibly HALTED: {reason}")

    def step(self, perception_input: Optional[str] = None) -> AgentState:
        """
        Executes exactly one state transition according to the deterministic graph:
        IDLE -> PERCEIVING -> DELIBERATING -> ACTING -> OBSERVING -> REFLECTING -> DELIBERATING | HALTED
        Enforces hard step budget (max_steps <= 20).
        """
        if self.state == AgentState.HALTED:
            return self.state

        # Enforce step budget: strictly terminate if step count reaches or exceeds limit
        if self.step_count >= self.max_steps:
            self.state = AgentState.HALTED
            self.halt_reason = f"Step budget exceeded ({self.step_count}/{self.max_steps})"
            print(f"[SECURITY] Maximum step budget ({self.max_steps}) exceeded! Forcing state -> HALTED.")
            return self.state

        self.step_count += 1
        prev_state = self.state

        if self.state == AgentState.IDLE:
            if perception_input:
                self.current_perception = perception_input
            self.state = AgentState.PERCEIVING

        elif self.state == AgentState.PERCEIVING:
            self.state = AgentState.DELIBERATING

        elif self.state == AgentState.DELIBERATING:
            self.state = AgentState.ACTING

        elif self.state == AgentState.ACTING:
            self.state = AgentState.OBSERVING

        elif self.state == AgentState.OBSERVING:
            self.state = AgentState.REFLECTING

        elif self.state == AgentState.REFLECTING:
            if self.mission_completed:
                self.state = AgentState.HALTED
                self.halt_reason = "Mission objective successfully completed."
            else:
                self.state = AgentState.DELIBERATING

        print(f"Step {self.step_count:02d}: Transitioned from {prev_state.name} -> {self.state.name}")
        return self.state

    def run_cycle(self, max_cycles: int = 5) -> List[AgentState]:
        """Runs the agent until HALTED or cycle threshold reached."""
        states = [self.state]
        while self.state != AgentState.HALTED and self.step_count < self.max_steps:
            s = self.step()
            states.append(s)
            if self.step_count >= max_cycles * 5:
                break
        return states


if __name__ == "__main__":
    print("=== Subsystem 06: Agent State Machine Simulation ===")
    pilot = Agent(max_steps=10)
    for i in range(12):
        if pilot.state == AgentState.HALTED:
            print(f"Agent safely HALTED at step {pilot.step_count}. Reason: {pilot.halt_reason}")
            break
        pilot.step(perception_input=f"Sensor telemetry packet {i}")
