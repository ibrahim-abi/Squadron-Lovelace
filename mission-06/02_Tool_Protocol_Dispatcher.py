"""
Mission 06 — Part 2: Communications & Protocol Officer
Tool Protocol Contracts, Schema Validation & Exception Boundaries

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-06-CAR (Eta Carinae & Homunculus Nebula)
Subsystem 06: Autonomous Warp State Machine & Deliberation Loop
Cadet Assigned: Anas Asif Hussain Shaik (Communications & Protocol Officer)
Squadron: Squadron Lovelace
"""

import json
from typing import Callable, Dict, Any, Tuple, Optional


# 1. Ship Navigation & Sensor Tools
def navigate_warp_vector(x: float, y: float, z: float) -> str:
    """Sets course vector through the Homunculus Nebula."""
    return f"Warp navigational coordinates successfully aligned to vector [{x:.2f}, {y:.2f}, {z:.2f}]."


def adjust_plasma_shields(intensity: int) -> str:
    """Modulates magnetic plasma shield intensity (0 to 100%)."""
    if not isinstance(intensity, int) or isinstance(intensity, bool):
        raise TypeError(f"Shield intensity must be an integer, got {type(intensity).__name__}.")
    if not (0 <= intensity <= 100):
        raise ValueError(f"Shield intensity {intensity}% is out of bounds! Valid range: [0, 100].")
    return f"Plasma shield harmonics adjusted to {intensity}% capacity."


def query_sensor_array(target: str) -> str:
    """Scans telemetry for anomalies or radiation spikes in Sector SEC-06-CAR."""
    if not isinstance(target, str):
        raise TypeError(f"Sensor target must be a string, got {type(target).__name__}.")
    
    known_targets = {
        "homunculus_core": "Severe plasma flare detected. High density stellar dust at azimuth 142.",
        "eta_carinae_b": "Secondary O-type binary companion emission: 100x solar luminosity.",
        "escape_corridor": "Clear gravitational corridor detected along vector [-0.85, 0.52, 0.05].",
        "plasma_shearing": "Turbulent magnetic field shear. Shield intensity must remain above 50%."
    }
    return known_targets.get(target.lower(), f"Sensor array reports: Target '{target}' unresolved.")


# 2. Strict JSON-Schema Contracts
TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "navigate_warp_vector": {
        "type": "object",
        "properties": {
            "x": {"type": "number"},
            "y": {"type": "number"},
            "z": {"type": "number"}
        },
        "required": ["x", "y", "z"]
    },
    "adjust_plasma_shields": {
        "type": "object",
        "properties": {
            "intensity": {"type": "integer"}
        },
        "required": ["intensity"]
    },
    "query_sensor_array": {
        "type": "object",
        "properties": {
            "target": {"type": "string"}
        },
        "required": ["target"]
    }
}

TOOL_REGISTRY: Dict[str, Callable] = {
    "navigate_warp_vector": navigate_warp_vector,
    "adjust_plasma_shields": adjust_plasma_shields,
    "query_sensor_array": query_sensor_array
}


def validate_tool_arguments(tool_name: str, arguments: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validates tool arguments against JSON-Schema contract.
    Returns (is_valid, error_message).
    """
    if tool_name not in TOOL_SCHEMAS:
        return False, f"Unknown tool '{tool_name}'. Available tools: {list(TOOL_SCHEMAS.keys())}"

    schema = TOOL_SCHEMAS[tool_name]
    required_keys = schema.get("required", [])

    # Check required fields
    for req in required_keys:
        if req not in arguments:
            return False, f"Missing required parameter '{req}' for tool '{tool_name}'."

    # Check parameter types
    properties = schema.get("properties", {})
    for key, val in arguments.items():
        if key not in properties:
            return False, f"Unexpected parameter '{key}' passed to tool '{tool_name}'."
        
        expected_type = properties[key].get("type")
        if expected_type == "number":
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                return False, f"Parameter '{key}' must be a number, got {type(val).__name__} ({val})."
        elif expected_type == "integer":
            if not isinstance(val, int) or isinstance(val, bool):
                return False, f"Parameter '{key}' must be an integer, got {type(val).__name__} ({val})."
        elif expected_type == "string":
            if not isinstance(val, str):
                return False, f"Parameter '{key}' must be a string, got {type(val).__name__}."

    return True, None


class ActionDispatcher:
    """
    Validates, boundaries, and executes tool actions for AURA.
    Catches all exceptions gracefully and formats error observations.
    """
    def __init__(self, registry: Optional[Dict[str, Callable]] = None, schemas: Optional[Dict[str, Dict[str, Any]]] = None):
        self.registry = registry or TOOL_REGISTRY
        self.schemas = schemas or TOOL_SCHEMAS

    def dispatch(self, tool_name: str, raw_arguments: Any) -> Tuple[bool, str]:
        """
        Executes tool within a strict error boundary.
        Returns:
            (success: bool, observation: str)
        """
        # Parse JSON string if payload was serialized
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except Exception as json_err:
                return False, f"Tool Execution Error: Failed to parse argument JSON string: {str(json_err)}"
        elif isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            return False, f"Tool Execution Error: Arguments must be a dict or valid JSON string, got {type(raw_arguments).__name__}."

        # Validate schema
        is_valid, validation_err = validate_tool_arguments(tool_name, arguments)
        if not is_valid:
            return False, f"Tool Execution Error: {validation_err}"

        # Execute inside error boundary
        tool_fn = self.registry[tool_name]
        try:
            result = tool_fn(**arguments)
            return True, str(result)
        except Exception as exc:
            return False, f"Tool Execution Error: {type(exc).__name__}: {str(exc)}"


if __name__ == "__main__":
    print("=== Subsystem 06: Tool Protocol & Action Dispatcher Simulation ===")
    dispatcher = ActionDispatcher()

    # 1. Valid execution
    ok, obs = dispatcher.dispatch("query_sensor_array", {"target": "escape_corridor"})
    print(f"[Valid Call] Success: {ok} -> {obs}")

    # 2. Schema violation (missing required parameter)
    ok, obs = dispatcher.dispatch("navigate_warp_vector", {"x": 1.0, "y": 2.0})
    print(f"[Schema Failure] Success: {ok} -> {obs}")

    # 3. Execution boundary exception (negative shield intensity)
    ok, obs = dispatcher.dispatch("adjust_plasma_shields", {"intensity": -25})
    print(f"[Runtime Exception Caught] Success: {ok} -> {obs}")

    # 4. Malformed JSON string
    ok, obs = dispatcher.dispatch("adjust_plasma_shields", "{'invalid_json': true}")
    print(f"[Malformed JSON Caught] Success: {ok} -> {obs}")
