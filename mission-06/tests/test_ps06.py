"""
Unit Tests for Mission 06: Squadron Lovelace
The Autonomous Warp State Machine & Deliberation Loop
Sector: SEC-06-CAR (Eta Carinae & Homunculus Nebula)
Subsystem 06: Autonomous Warp State Machine & Deliberation Loop
"""

import sys
import os
import math
import pytest
from typing import List

# Ensure mission-06 directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from importlib import import_module

state_mod = import_module("01_Agent_State_Machine")
AgentState = state_mod.AgentState
Agent = state_mod.Agent

tool_mod = import_module("02_Tool_Protocol_Dispatcher")
ActionDispatcher = tool_mod.ActionDispatcher
validate_tool_arguments = tool_mod.validate_tool_arguments
navigate_warp_vector = tool_mod.navigate_warp_vector
adjust_plasma_shields = tool_mod.adjust_plasma_shields
query_sensor_array = tool_mod.query_sensor_array

mdp_mod = import_module("03_MDP_Trajectory_and_Return")
MDPTrajectoryLogger = mdp_mod.MDPTrajectoryLogger
compute_discounted_return = mdp_mod.compute_discounted_return
compute_transition_probability_matrix = mdp_mod.compute_transition_probability_matrix

sentinel_mod = import_module("04_Cycle_Sentinel_and_System2")
FloydCycleSentinel = sentinel_mod.FloydCycleSentinel
hash_state_action = sentinel_mod.hash_state_action
System2DeliberationEngine = sentinel_mod.System2DeliberationEngine


# ==============================================================================
# 1. State Machine Test (Requirement 1 from ps06.pdf)
# ==============================================================================
def test_state_machine_transitions_and_step_budget():
    """
    Assert that the agent transitions correctly from IDLE to HALTED
    and strictly terminates if forced to run 21 steps.
    """
    # 1. Nominal transition cycle test
    agent = Agent(max_steps=20)
    assert agent.state == AgentState.IDLE
    
    agent.step("Initial sensor telemetry")
    assert agent.state == AgentState.PERCEIVING
    
    agent.step()
    assert agent.state == AgentState.DELIBERATING
    
    agent.step()
    assert agent.state == AgentState.ACTING
    
    agent.step()
    assert agent.state == AgentState.OBSERVING
    
    agent.step()
    assert agent.state == AgentState.REFLECTING
    
    # Mission complete -> transition to HALTED
    agent.mission_completed = True
    agent.step()
    assert agent.state == AgentState.HALTED

    # 2. Hard budget test: forcing 21 steps must guarantee HALTED state
    budget_agent = Agent(max_steps=20)
    for _ in range(25):
        budget_agent.step("Stress test step")

    assert budget_agent.state == AgentState.HALTED
    assert budget_agent.step_count == 20, f"Expected step count to clamp at max budget 20, got {budget_agent.step_count}"
    assert "budget exceeded" in budget_agent.halt_reason.lower()


# ==============================================================================
# 2. Protocol Dispatch Test (Requirement 2 from ps06.pdf)
# ==============================================================================
def test_protocol_dispatch_and_error_boundaries():
    """
    Assert that passing an invalid JSON payload to the dispatcher safely returns
    a formatted string error observation and does NOT crash the Python process.
    """
    dispatcher = ActionDispatcher()

    # 1. Malformed JSON string
    success, obs = dispatcher.dispatch("navigate_warp_vector", "{invalid_json: 123}")
    assert success is False
    assert isinstance(obs, str)
    assert "Tool Execution Error" in obs
    assert "Failed to parse argument JSON string" in obs

    # 2. Missing required parameters
    success, obs = dispatcher.dispatch("navigate_warp_vector", {"x": 1.0, "y": 2.0})
    assert success is False
    assert "Missing required parameter 'z'" in obs

    # 3. Type violation
    success, obs = dispatcher.dispatch("adjust_plasma_shields", {"intensity": "super_high"})
    assert success is False
    assert "must be an integer" in obs

    # 4. Out of bounds runtime exception
    success, obs = dispatcher.dispatch("adjust_plasma_shields", {"intensity": -50})
    assert success is False
    assert "Shield intensity -50% is out of bounds" in obs

    # 5. Valid call succeeds
    success, obs = dispatcher.dispatch("navigate_warp_vector", {"x": 1.5, "y": -0.8, "z": 0.2})
    assert success is True
    assert "successfully aligned" in obs


# ==============================================================================
# 3. MDP Return Test (Requirement 3 from ps06.pdf)
# ==============================================================================
def test_mdp_discounted_return_calculation():
    """
    Given a hardcoded trajectory with rewards [10, 0, -5, 20] and gamma = 0.95,
    assert the calculated G_0 matches the expected mathematical output (22.635).
    """
    rewards = [10.0, 0.0, -5.0, 20.0]
    gamma = 0.95

    # Hand calculation:
    # t=0: 10 * 0.95^0 = 10.0
    # t=1: 0 * 0.95^1 = 0.0
    # t=2: -5 * 0.95^2 = -4.5125
    # t=3: 20 * 0.95^3 = 17.1475
    # Sum = 22.635
    expected_g0 = 10.0 + 0.0 - 4.5125 + 17.1475
    computed_g0 = compute_discounted_return(rewards, gamma=gamma)

    assert math.isclose(computed_g0, expected_g0, abs_tol=1e-5), f"Expected {expected_g0}, got {computed_g0}"

    # Trajectory Logger test
    logger = MDPTrajectoryLogger()
    logger.record_transition("s0", "a0", 10.0, "s1")
    logger.record_transition("s1", "a1", 0.0, "s2")
    logger.record_transition("s2", "a2", -5.0, "s3")
    logger.record_transition("s3", "a3", 20.0, "s4", done=True)

    logged_g0 = compute_discounted_return(logger.get_rewards(), gamma=gamma)
    assert math.isclose(logged_g0, expected_g0, abs_tol=1e-5)


def test_transition_probability_matrix_normalization():
    """Verifies that empirical transition frequencies sum to 1.0 for each (s, a)."""
    dataset = [
        ("s0", "a0", "s1"),
        ("s0", "a0", "s1"),
        ("s0", "a0", "s2"),
        ("s1", "a1", "s2"),
        ("s1", "a1", "s0")
    ]
    prob_mat = compute_transition_probability_matrix(dataset)
    assert math.isclose(prob_mat[("s0", "a0")]["s1"], 2.0 / 3.0, abs_tol=1e-5)
    assert math.isclose(prob_mat[("s0", "a0")]["s2"], 1.0 / 3.0, abs_tol=1e-5)
    assert math.isclose(sum(prob_mat[("s0", "a0")].values()), 1.0, abs_tol=1e-5)


# ==============================================================================
# 4. Sentinel Cycle Test (Requirement 4 from ps06.pdf)
# ==============================================================================
def test_sentinel_floyd_cycle_detection():
    """
    Provide the agent with a mock state-action sequence that repeats [s1, s2, s3, s1, s2, s3...].
    Assert that the Tortoise and Hare algorithm detects the cycle and halts the agent before step 10.
    """
    sentinel = FloydCycleSentinel(repeat_limit=3)
    mock_pattern = ["s1", "s2", "s3", "s1", "s2", "s3", "s1", "s2", "s3"]

    halted_step = None
    for step_num, st in enumerate(mock_pattern, start=1):
        tripped, reason = sentinel.record_and_evaluate(st, "navigate_warp_vector", {"x": 1.0})
        if tripped:
            halted_step = step_num
            break

    assert halted_step is not None, "Cycle sentinel must detect repeating cycle"
    assert halted_step < 10, f"Expected cycle detection before step 10, but tripped at step {halted_step}"
    assert halted_step == 6, f"Expected Floyd cycle detection at step 6 for period 3, got {halted_step}"


def test_system2_deliberation_interventions():
    """Verifies System 2 audit intercepts dangerous low-shield warp attempts."""
    engine = System2DeliberationEngine(min_shield_margin=50)

    # Low shield warp attempt -> System 2 modifies to raising shields
    plan = engine.audit_and_deliberate(
        thought="Attempting blind warp jump through dust cloud.",
        proposed_action="navigate_warp_vector",
        proposed_args={"x": -0.85, "y": 0.52, "z": 0.05},
        current_shield_level=30
    )
    assert plan["approved"] is False
    assert plan["executed_action"] == "adjust_plasma_shields"
    assert plan["executed_args"]["intensity"] >= 50
    assert len(plan["interventions"]) > 0
