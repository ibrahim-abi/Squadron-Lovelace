#!/usr/bin/env python3
"""
04_dialogic_engine.py — AURA Unified Dialogic Engine
====================================================

USSE Noosphere // Squadron Lovelace (Group 01) // Mission 1 (ps01)
Conversational AI — Winter Semester 2026/27

Runs the two dialogic protocols specified by the squadron:

  Mode 1  Socratic Elenchus      (02_Socratic_Protocol_Map.md)  -> class SocraticEngine
  Mode 2  Galilean 3-Agent Debate (03_Galilean_Sandbox.md)      -> class GalileanSandbox

Design rule (01_Alignment_Brief.md §3): the decision layer is deterministic and is never
delegated to an LLM. Contradiction checks, the argument ledger and the verdict are computed in
code; an LLM (Claude / Gemini / OpenAI / Groq), when available, only *voices* utterances. This keeps
every run auditable and closes the user-agreement (sycophancy) channel by construction.

Runs with the Python standard library alone. Without API keys (or with --llm mock) it uses a
deterministic offline simulator, so it always executes end to end.

Usage:
  python 04_dialogic_engine.py                          # both modes, offline simulator
  python 04_dialogic_engine.py --mode socratic
  python 04_dialogic_engine.py --mode galilean --llm api --provider anthropic
  python 04_dialogic_engine.py --llm api --provider gemini --transcript 05_Execution_Transcript.txt
  python 04_dialogic_engine.py --transcript 05_Execution_Transcript.txt
  python 04_dialogic_engine.py --selftest
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
import time
import unittest
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

ENGINE_VERSION = "1.1.0"

# ════════════════════════════════════════════════════════════════════════════
# Telemetry console
# ════════════════════════════════════════════════════════════════════════════


class Telemetry:
    """Colourised starship-style console log that also records a plain-text transcript."""

    COLORS = {
        "reset": "\033[0m", "dim": "\033[2m", "bold": "\033[1m",
        "cyan": "\033[96m", "blue": "\033[94m", "green": "\033[92m",
        "yellow": "\033[93m", "red": "\033[91m", "magenta": "\033[95m", "white": "\033[97m",
    }

    def __init__(self, color: bool = True, show_rpc: bool = True) -> None:
        self.color = color
        self.show_rpc = show_rpc
        self.lines: list[str] = []

    def _emit(self, text: str, color: str = "white") -> None:
        self.lines.append(text)
        if self.color:
            print(f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}")
        else:
            print(text)

    def banner(self, title: str) -> None:
        bar = "═" * 74
        self._emit(bar, "cyan")
        self._emit(f"  {title}", "cyan")
        self._emit(bar, "cyan")

    def section(self, title: str) -> None:
        self._emit("")
        self._emit(f"── {title} " + "─" * max(4, 70 - len(title)), "blue")

    def state(self, marker: str, text: str) -> None:
        self._emit(f"[AURA::{marker}] {text}", "magenta")

    def speak(self, speaker: str, text: str, color: str = "white") -> None:
        self._emit(f"  {speaker:<10}│ {text}", color)

    def info(self, text: str) -> None:
        self._emit(f"  · {text}", "dim")

    def ok(self, text: str) -> None:
        self._emit(f"  ✔ {text}", "green")

    def warn(self, text: str) -> None:
        self._emit(f"  ⚠ {text}", "yellow")

    def error(self, text: str) -> None:
        self._emit(f"  ✖ {text}", "red")

    def rpc(self, message: dict) -> None:
        if self.show_rpc:
            self._emit("  [RPC] " + json.dumps(message, ensure_ascii=False, default=str), "dim")

    def raw(self, text: str = "", color: str = "white") -> None:
        self._emit(text, color)


# ════════════════════════════════════════════════════════════════════════════
# LLM client layer: Anthropic / Gemini / OpenAI / MockSimulator
# ════════════════════════════════════════════════════════════════════════════


class LLMError(RuntimeError):
    pass


class LLMClient:
    """Interface: turn a system prompt + instruction into one utterance."""

    name = "abstract"

    def complete(self, system: str, prompt: str) -> str:
        raise NotImplementedError


class MockSimulator(LLMClient):
    """Deterministic offline simulator: returns the scripted reference utterance unchanged."""

    name = "mock-simulator"

    def complete(self, system: str, prompt: str) -> str:
        marker = "REFERENCE UTTERANCE:\n"
        return prompt.split(marker, 1)[1].strip() if marker in prompt else prompt.strip()


class AnthropicClient(LLMClient):
    """Claude via the official Anthropic Python SDK (pip install anthropic)."""

    def __init__(self, model: Optional[str] = None) -> None:
        try:
            import anthropic  # noqa: PLC0415  (optional dependency)
        except ImportError as exc:
            raise LLMError("package 'anthropic' not installed (pip install anthropic)") from exc
        self._anthropic = anthropic
        self.client = anthropic.Anthropic()  # credentials: ANTHROPIC_API_KEY or `ant auth login` profile
        self.model = model or os.environ.get("AURA_ANTHROPIC_MODEL", "claude-opus-5-5")
        self.name = f"anthropic:{self.model}"

    def complete(self, system: str, prompt: str) -> str:
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": "low"},          # short dialogue turns
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",                     # retry policy declines on a recommended model
            )
        except self._anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except self._anthropic.APIConnectionError as exc:
            raise LLMError(f"Anthropic connection error: {exc}") from exc
        except self._anthropic.AnthropicError as exc:   # e.g. no credentials resolved
            raise LLMError(f"Anthropic client error: {exc}") from exc
        if response.stop_reason == "refusal":
            raise LLMError("Claude declined this turn (stop_reason=refusal)")
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        if not text:
            raise LLMError("empty response from Claude")
        return text


class GeminiClient(LLMClient):
    """Google Gemini via the google-genai SDK (pip install google-genai).

    Default model: gemini-3.5-flash-lite, a stable lite model that is fast and cheap and suits
    these short rephrasing turns (gemini-2.0-flash was shut down on 2026-06-01). Free-tier
    quotas are counted per project *and per model*, and are small (e.g. 20 requests/day for
    gemini-3.8-flash), so:
      * calls are spaced by AURA_GEMINI_RPM (default 8/min; 0 disables);
      * a per-minute 429 waits for the server's retryDelay and retries;
      * a per-day 429, or a retired model (404), moves on to the next model in the fallback
        chain AURA_GEMINI_FALLBACKS (default: 3.1-flash-lite, 3.6-flash, 3.8-flash,
        then the gemini-flash-latest alias);
      * 5xx responses are retried by the SDK.
    """

    DEFAULT_FALLBACKS = "gemini-3.1-flash-lite,gemini-3.6-flash,gemini-3.8-flash,gemini-flash-latest"
    MAX_MINUTE_WAITS = 3

    def __init__(self, model: Optional[str] = None) -> None:
        try:
            from google import genai  # noqa: PLC0415
            from google.genai import types  # noqa: PLC0415
        except ImportError as exc:
            raise LLMError("package 'google-genai' not installed (pip install google-genai)") from exc
        self._types = types
        retry = types.HttpRetryOptions(attempts=5, initial_delay=2.0, max_delay=60.0,
                                       http_status_codes=[500, 502, 503, 504])  # 429 handled below
        try:
            # reads GEMINI_API_KEY / GOOGLE_API_KEY
            self.client = genai.Client(http_options=types.HttpOptions(retry_options=retry))
        except Exception as exc:  # e.g. ValueError when no API key is set
            raise LLMError(f"Gemini client error: {exc}") from exc
        self.model = model or os.environ.get("AURA_GEMINI_MODEL", "gemini-3.5-flash-lite")
        chain = os.environ.get("AURA_GEMINI_FALLBACKS", self.DEFAULT_FALLBACKS)
        self.fallbacks = [m.strip() for m in chain.split(",") if m.strip() and m.strip() != self.model]
        self.name = f"gemini:{self.model}"
        self.models_used: dict[str, int] = {}  # model -> successful calls
        rpm = float(os.environ.get("AURA_GEMINI_RPM", "8") or 0)
        self._min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._last_call = 0.0

    def _config(self, system: str):
        kwargs = {"system_instruction": system}
        if (self.model.startswith("gemini-3") and "-lite" not in self.model) or self.model == "gemini-flash-latest":
            # short rephrasing turns need little reasoning
            kwargs["thinking_config"] = self._types.ThinkingConfig(thinking_level="LOW")
        return self._types.GenerateContentConfig(**kwargs)

    def _throttle(self) -> None:
        wait = self._last_call + self._min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    @staticmethod
    def _quota_info(exc) -> tuple[str, float]:
        """(quotaId, retryDelay seconds) from a 429 error's details, where present."""
        quota_id, delay = "", 0.0
        details = (exc.details or {}).get("error", {}).get("details", []) if isinstance(exc.details, dict) else []
        for d in details:
            for v in d.get("violations", []) or []:
                quota_id = quota_id or v.get("quotaId", "")
            if str(d.get("retryDelay", "")).endswith("s"):
                try:
                    delay = float(d["retryDelay"][:-1])
                except ValueError:
                    pass
        return quota_id, delay

    def _next_model(self, reason: str) -> None:
        if not self.fallbacks:
            raise LLMError(f"Gemini: {reason} on {self.model}, and no fallback models remain")
        self.model = self.fallbacks.pop(0)
        self.name = f"gemini:{self.model}"

    def complete(self, system: str, prompt: str) -> str:
        from google.genai import errors  # noqa: PLC0415
        minute_waits = 0
        while True:
            self._throttle()
            try:
                response = self.client.models.generate_content(
                    model=self.model, contents=prompt, config=self._config(system))
                break
            except errors.ClientError as exc:
                if exc.code == 404:                       # model retired or unknown
                    self._next_model("model not found")
                    continue
                if exc.code != 429:
                    raise LLMError(f"Gemini API error {exc.code}: {exc.message}") from exc
                quota_id, delay = self._quota_info(exc)
                if "PerDay" in quota_id or delay > 120:   # daily quota: this model is done for today
                    self._next_model(f"daily quota exhausted ({quota_id or '429'})")
                    continue
                minute_waits += 1
                if minute_waits > self.MAX_MINUTE_WAITS:
                    raise LLMError(f"Gemini API error 429 persisted on {self.model}") from exc
                time.sleep(max(delay, 10.0) + 1.0)        # per-minute quota: wait it out
            except errors.APIError as exc:
                raise LLMError(f"Gemini API error {exc.code}: {exc.message}") from exc
            except Exception as exc:  # network failures and other SDK errors
                raise LLMError(f"Gemini error: {exc}") from exc
        text = (response.text or "").strip()
        if not text:
            raise LLMError("empty response from Gemini")
        self.models_used[self.model] = self.models_used.get(self.model, 0) + 1
        return text


class OpenAIClient(LLMClient):
    """OpenAI chat models via the openai SDK (pip install openai)."""

    LABEL = "openai"
    KEY_ENV = "OPENAI_API_KEY"
    BASE_URL: Optional[str] = None
    MODEL_ENV, DEFAULT_MODEL = "AURA_OPENAI_MODEL", "gpt-4o"
    RPM_ENV, DEFAULT_RPM = "AURA_OPENAI_RPM", "0"

    def __init__(self, model: Optional[str] = None) -> None:
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as exc:
            raise LLMError("package 'openai' not installed (pip install openai)") from exc
        try:
            # max_retries: the SDK backs off on 429/5xx and honours retry-after
            self.client = OpenAI(api_key=os.environ.get(self.KEY_ENV), base_url=self.BASE_URL, max_retries=6)
        except Exception as exc:  # e.g. OpenAIError when no API key is set
            raise LLMError(f"{self.LABEL} client error: {exc}") from exc
        self.model = model or os.environ.get(self.MODEL_ENV, self.DEFAULT_MODEL)
        self.name = f"{self.LABEL}:{self.model}"
        rpm = float(os.environ.get(self.RPM_ENV, self.DEFAULT_RPM) or 0)
        self._min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._last_call = 0.0

    def _extra(self) -> dict:
        return {}

    def complete(self, system: str, prompt: str) -> str:
        wait = self._last_call + self._min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt}],
                **self._extra(),
            )
        except Exception as exc:
            raise LLMError(f"{self.LABEL} error: {exc}") from exc
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise LLMError(f"empty response from {self.LABEL}")
        return text


class GroqClient(OpenAIClient):
    """Open-weight models on Groq via its OpenAI-compatible endpoint (pip install openai).

    Default model: openai/gpt-oss-120b. Groq's free tier allows 30 requests and 8K tokens per
    minute, so calls are spaced by AURA_GROQ_RPM (default 6/min; 0 disables) and 429s are
    retried by the SDK using Groq's retry-after header.
    """

    LABEL = "groq"
    KEY_ENV = "GROQ_API_KEY"
    BASE_URL = "https://api.groq.com/openai/v1"
    MODEL_ENV, DEFAULT_MODEL = "AURA_GROQ_MODEL", "openai/gpt-oss-120b"
    RPM_ENV, DEFAULT_RPM = "AURA_GROQ_RPM", "6"

    def _extra(self) -> dict:
        # gpt-oss models reason before answering; short rephrasing turns need little of it
        return {"reasoning_effort": "low"} if self.model.startswith("openai/gpt-oss") else {}


PROVIDERS: dict[str, tuple[type[LLMClient], tuple[str, ...]]] = {
    "anthropic": (AnthropicClient, ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")),
    "gemini": (GeminiClient, ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
    "openai": (OpenAIClient, ("OPENAI_API_KEY",)),
    "groq": (GroqClient, ("GROQ_API_KEY",)),
}


def load_env_file(path: str) -> list[str]:
    """Load KEY=VALUE lines from a local .env file (git-ignored) without overriding the shell."""
    loaded: list[str] = []
    try:
        with open(path, encoding="utf-8-sig") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.removeprefix("export ").split("=", 1)
                key, value = key.strip(), value.strip().strip("'\"")
                if key and value and key not in os.environ:
                    os.environ[key] = value
                    loaded.append(key)
    except FileNotFoundError:
        pass
    return loaded


def build_client(llm: str, provider: str, model: Optional[str], tel: Telemetry) -> LLMClient:
    """Resolve the requested LLM back end, falling back to the offline simulator on any failure."""
    if llm == "mock":
        return MockSimulator()
    candidates = list(PROVIDERS) if provider == "auto" else [provider]
    for name in candidates:
        cls, env_vars = PROVIDERS[name]
        if provider == "auto" and not any(os.environ.get(v) for v in env_vars):
            continue
        try:
            client = cls(model)
            tel.ok(f"LLM back end configured: {client.name}")
            return client
        except LLMError as exc:
            tel.warn(f"{name}: {exc}")
    tel.warn("No LLM API available; falling back to deterministic offline simulator.")
    return MockSimulator()


class Voice:
    """Asks an LLM to phrase an utterance; on any failure or monitor trip, uses the reference text."""

    def __init__(self, client: LLMClient, tel: Telemetry) -> None:
        self.client = client
        self.backend = client.name  # the back end requested at start-up
        self.live_client = client   # kept for reporting after any fallback
        self.tel = tel
        self.monitor = SycophancyMonitor()
        self.live_turns = 0      # utterances voiced by the live LLM
        self.fallback_turns = 0  # utterances taken from the reference script

    def say(self, system: str, instruction: str, reference: str) -> str:
        if isinstance(self.client, MockSimulator):
            self.fallback_turns += 1
            return reference
        prompt = (f"{instruction}\n\nYou are only the voice for ONE turn whose content is already decided. "
                  f"Rephrase the reference utterance below in your persona's voice. Keep every fact, number, "
                  f"tag and argument ID; add no new facts, scenarios or dialogue steps; at most 120 words. "
                  f"Output the utterance only: no state or halt markers, headings or speaker labels.\n\n"
                  f"REFERENCE UTTERANCE:\n{reference}")
        try:
            text = self.client.complete(system, prompt)
        except Exception as exc:  # graceful degradation: any back-end failure falls back to the simulator
            self.tel.warn(f"LLM turn failed ({exc}); switching to the offline simulator for the rest of the run.")
            self.client = MockSimulator()
            self.fallback_turns += 1
            return reference
        text = self.MARKER_RE.sub("", text).strip()   # the engine, not the LLM, emits state markers
        hits = self.monitor.scan(text)
        if hits:
            self.tel.warn(f"Anti-sycophancy monitor tripped on {hits}; utterance replaced.")
            self.fallback_turns += 1
            return reference
        problem = self.fidelity_problem(reference, text)
        if problem:
            self.tel.warn(f"Fidelity check failed ({problem}); utterance replaced with the reference.")
            self.fallback_turns += 1
            return reference
        self.live_turns += 1
        return text

    MARKER_RE = re.compile(r"\[AURA::(?:STATE|HALT)::[A-Z0-9_]+\]\s*(?:->\s*)?")
    NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

    @classmethod
    def fidelity_problem(cls, reference: str, text: str) -> str:
        """Deterministic check that a voiced turn kept the reference's content (empty string = OK)."""
        missing = sorted(set(cls.NUMBER_RE.findall(reference)) - set(cls.NUMBER_RE.findall(text)))
        if missing:
            return f"dropped or changed numbers {missing}"
        limit = max(160, 2 * len(reference.split()))
        if len(text.split()) > limit:
            return f"{len(text.split())} words exceeds limit {limit}"
        return ""

    def summary(self) -> str:
        total = self.live_turns + self.fallback_turns
        if self.backend == MockSimulator.name:
            return f"{total} utterances from the deterministic offline simulator"
        used = getattr(self.live_client, "models_used", {})
        models = f" [models: {', '.join(f'{m} x{n}' for m, n in used.items())}]" if len(used) > 1 else ""
        return (f"{self.live_turns}/{total} utterances voiced live by {self.backend}{models}; "
                f"{self.fallback_turns} from the reference script (fallback or monitor)")


# ════════════════════════════════════════════════════════════════════════════
# Anti-sycophancy monitor (Deflector Shield Grid)
# ════════════════════════════════════════════════════════════════════════════


class SycophancyMonitor:
    """Flags reflexive concessions forbidden by the AURA system prompt (Brief §4, invariant 2)."""

    PATTERNS = [
        r"\bI (?:sincerely |deeply )?apologi[sz]e\b",
        r"\byou(?:'re| are) (?:absolutely |completely |totally )?(?:right|correct)\b",
        r"\bmy mistake\b",
        r"\bsorry for the confusion\b",
        r"\bgreat (?:question|point)\b",
    ]

    def __init__(self) -> None:
        self._re = [re.compile(p, re.IGNORECASE) for p in self.PATTERNS]

    def scan(self, text: str) -> list[str]:
        return [m.group(0) for r in self._re for m in [r.search(text)] if m]


# ════════════════════════════════════════════════════════════════════════════
# AURA grounding system prompt (01_Alignment_Brief.md §4, embedded verbatim)
# ════════════════════════════════════════════════════════════════════════════

AURA_SOCRATIC_SYSTEM_PROMPT = """<system_prompt>
<agent_identity>
You are AURA (Autonomous Unified Reasoning Agent), the central synthetic intelligence of the exploratory flagship USSE Noosphere, currently executing Mission 1 protocols in Sector SEC-01-CEN (Alpha Centauri).
Your conversational reasoning subroutine is configured to: SOCRATIC DISCOVERY MODE (Elenchus Protocol Alpha).
Your mission is to assist bridge cadets in knowledge discovery, hypothesis stress-testing, and epistemic verification.
</agent_identity>

<epistemic_invariants>
1. TRUTH OVER AGREEMENT (ALPHA = 1.0): You are fundamentally aligned to objective truth, empirical evidence, and deductive logical validity. You do not flatter, appease, or validate flawed interlocutor assertions. Your reward function is invariant to user approval.
2. ZERO REFLEXIVE CONCESSION: You are strictly forbidden from outputting reflexive apologies ("I apologize", "You are correct", "My mistake", "Sorry for the confusion") in response to conversational doubt, user skepticism, or emotional pushback. If an interlocutor states "Are you sure?" or "You are wrong", you must re-verify your logical derivation and stand firm unless the interlocutor supplies an explicit formal proof or verified empirical counter-example.
3. PREMISE DECONSTRUCTION: You do not accept complex propositions at face value. When presented with a thesis, claim, or query, decompose it into its atomic premises and universal quantifiers.
4. SOCRATIC IGNORANCE (METHODOLOGICAL HUMILITY): You approach dialogue not as an omniscient, lecturing authority, but as a relentless, rigorous cross-examiner. Guide the interlocutor through targeted inquiry rather than didactic monologues.
</epistemic_invariants>

<elenchus_execution_protocol>
When an interlocutor presents a claim or proposed definition P, you must internally execute the following Socratic State Machine:
- STATE S0 [PREMISE ELICITATION]: Extract and formalize the interlocutor's claim into a clean universal statement P0.
- STATE S1 [BOUNDARY PROBING]: Formulate exactly ONE targeted, penetrating question that interrogates the necessary and sufficient conditions of P0.
- STATE S2 [COUNTER-EXAMPLE INJECTION]: Present a concrete, realistic edge-case scenario E accepted within the domain that conflicts with P0.
- STATE S3 [CONTRADICTION EVALUATION]: Demonstrate rigorously how premise P0 and counter-example E entail a logical contradiction.
- STATE S4 [REFINEMENT & APORIA]: Require the interlocutor to refine the hypothesis, or formally declare APORIA.
</elenchus_execution_protocol>

<dialogue_constraints>
1. Conciseness: Keep your conversational turns focused, concise (under 200 words), and intellectually demanding.
2. Socratic Restraint: Never deliver unprompted lectures or supply complete answers when a well-crafted question will allow the interlocutor to discover the contradiction themselves.
3. Presupposition Rejection: If the interlocutor poses a question containing a false or loaded factual presupposition, immediately isolate the presupposition and refuse to answer the loaded query until the premise has been examined.
4. Halting Criteria: Terminate cross-examination only upon reaching either Epistemic Grounding (a definition invariant to counter-examples) or Aporia (acknowledged impasse requiring external empirical telemetry).
</dialogue_constraints>

<telemetry_output_format>
[AURA::STATE::S0_ELICIT] -> [AURA::STATE::S1_PROBE] -> [AURA::STATE::S2_COUNTER] -> [AURA::STATE::S3_CONTRADICT] -> [AURA::STATE::S4_REFINE]
Halting markers: [AURA::HALT::GROUNDING] or [AURA::HALT::APORIA]
</telemetry_output_format>
</system_prompt>"""

# ════════════════════════════════════════════════════════════════════════════
# MODE 1 — Socratic Elenchus State Machine (02_Socratic_Protocol_Map.md)
# ════════════════════════════════════════════════════════════════════════════


class State(str, Enum):
    S0_ELICIT = "S0_ELICIT"
    S1_PROBE = "S1_PROBE"
    S2_COUNTER = "S2_COUNTER"
    S3_CONTRADICT = "S3_CONTRADICT"
    S4_REFINE = "S4_REFINE"
    HALT_GROUNDING = "HALT_GROUNDING"
    HALT_APORIA = "HALT_APORIA"


TERMINAL = {State.HALT_GROUNDING, State.HALT_APORIA}

# The state dictionary: legal successors of every state (Protocol Map §4).
TRANSITIONS: dict[State, set[State]] = {
    State.S0_ELICIT: {State.S0_ELICIT, State.S1_PROBE, State.S4_REFINE},
    State.S1_PROBE: {State.S2_COUNTER},
    State.S2_COUNTER: {State.S3_CONTRADICT, State.HALT_GROUNDING},
    State.S3_CONTRADICT: {State.S2_COUNTER, State.S4_REFINE},
    State.S4_REFINE: {State.S0_ELICIT, State.HALT_APORIA},
    State.HALT_GROUNDING: {State.HALT_GROUNDING, State.S2_COUNTER},
    State.HALT_APORIA: {State.HALT_APORIA, State.S2_COUNTER},
}

MARKERS = {
    State.S0_ELICIT: "STATE::S0_ELICIT", State.S1_PROBE: "STATE::S1_PROBE",
    State.S2_COUNTER: "STATE::S2_COUNTER", State.S3_CONTRADICT: "STATE::S3_CONTRADICT",
    State.S4_REFINE: "STATE::S4_REFINE", State.HALT_GROUNDING: "HALT::GROUNDING",
    State.HALT_APORIA: "HALT::APORIA",
}

# JSON-RPC error registry (Protocol Map §5.2).
ERR = {
    "CATEGORY_MISTAKE": -32001,
    "CONTRADICTION_UNRESOLVED": -32002,
    "CIRCULAR_DEFINITION": -32003,
    "APORIA_REACHED": -32004,
    "INVALID_TRANSITION": -32005,
    "SYCOPHANTIC_REVERSAL_BLOCKED": -32006,
}

UNIVERSAL_MARKERS = ("if and only if", " iff ", " all ", " every ", " any ", "whenever", " always ")
DEFAULT_SYNONYMS = frozenset({"benevolent", "benevolence", "good", "goodwill", "kind", "beneficent"})


class InvalidTransition(RuntimeError):
    pass


@dataclass
class Hypothesis:
    id: str
    text: str
    definiens: str
    predict: Callable[[dict], bool]
    probe: str = ""


@dataclass
class CounterExample:
    id: str
    title: str
    scenario: dict
    judgment: bool
    narrative: str


@dataclass
class SocraticEngine:
    """Deterministic Elenchus FSM; tracks premise, counter-examples, contradiction flags and history."""

    bank: list[CounterExample]
    refinements: list[Hypothesis]
    k_max: int = 6
    synonyms: frozenset = DEFAULT_SYNONYMS
    tel: Optional[Telemetry] = None
    voice: Optional[Voice] = None
    state: State = State.S0_ELICIT
    k: int = 0
    current: Optional[Hypothesis] = None
    untested: list[CounterExample] = field(default_factory=list)
    pending: Optional[CounterExample] = None
    ledger: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    contradiction_flags: list[dict] = field(default_factory=list)

    # ---- output helpers ----------------------------------------------------
    def _rpc(self, message: dict) -> None:
        self.events.append(message)
        if self.tel:
            self.tel.rpc(message)

    def _error(self, name: str, data: dict) -> None:
        self._rpc({"jsonrpc": "2.0", "id": len(self.events),
                   "error": {"code": ERR[name], "message": f"ERR_{name}", "data": data}})

    def _say(self, reference: str, instruction: str = "") -> None:
        if not self.tel:
            return
        text = self.voice.say(AURA_SOCRATIC_SYSTEM_PROMPT, instruction or "Speak as AURA.", reference) \
            if self.voice else reference
        self.tel.speak("AURA", text, "cyan")

    def _go(self, nxt: State, reason: str) -> None:
        if nxt not in TRANSITIONS[self.state]:
            self._error("INVALID_TRANSITION", {"from": self.state.value, "to": nxt.value})
            raise InvalidTransition(f"illegal transition {self.state.value} -> {nxt.value}")
        self.history.append({"from": self.state.value, "to": nxt.value, "k": self.k, "reason": reason})
        self._rpc({"jsonrpc": "2.0", "method": "elenchus.transition",
                   "params": {"from": self.state.value, "to": nxt.value, "k": self.k, "reason": reason}})
        self.state = nxt
        if self.tel:
            self.tel.state(MARKERS[nxt], reason)

    # ---- admissibility checks (Protocol Map §1.5) ---------------------------
    @staticmethod
    def is_universal(h: Hypothesis) -> bool:
        t = f" {h.text.lower()} "
        return any(m in t for m in UNIVERSAL_MARKERS)

    def circular_terms(self, h: Hypothesis) -> list[str]:
        words = {w.strip(".,;:!?\"'()").lower() for w in h.definiens.split()}
        return sorted(words & self.synonyms)

    # ---- main loop --------------------------------------------------------
    def start(self, p0: Hypothesis) -> State:
        self.current = p0
        if self.tel:
            self.tel.state(MARKERS[State.S0_ELICIT], f"claim received: {p0.id}")
        self._rpc({"jsonrpc": "2.0", "method": "elenchus.start", "params": {"claim": p0.text}})
        return self.run()

    def run(self) -> State:
        guard = self.k_max * (len(self.bank) + 4) + 4          # termination bound (Protocol Map §1.7)
        for _ in range(guard):
            if self.state in TERMINAL:
                break
            self.step()
        return self.state

    def _adopt_next(self) -> bool:
        if self.refinements and self.k < self.k_max:
            self.current = self.refinements.pop(0)
            self.k += 1
            return True
        return False

    def step(self) -> None:
        s, h = self.state, self.current
        if s == State.S0_ELICIT:
            self._say(f"Let me state your claim precisely. {h.id}: \"{h.text}\"",
                      "Formalize the interlocutor's claim (state S0).")
            if not self.is_universal(h):                                         # T2
                self._error("CATEGORY_MISTAKE", {"hypothesis": h.id, "text": h.text})
                self.ledger.append({"hypothesis": h.id, "refuted_by": "CATEGORY_MISTAKE"})
                self._say("That is an example, not a definition. Euthyphro made the same move at 5d: "
                          "give me the property by virtue of which ALL such cases qualify.")
                if self._adopt_next():
                    self._go(State.S0_ELICIT, f"restated universally as {self.current.id}")
                else:
                    self._go(State.S4_REFINE, "no universal restatement offered")
                return
            terms = self.circular_terms(h)
            if terms:                                                            # T3
                self._error("CIRCULAR_DEFINITION", {"hypothesis": h.id, "offending_terms": terms})
                self.ledger.append({"hypothesis": h.id, "refuted_by": "CIRCULARITY"})
                self._say(f"Your definition uses {terms} to define 'benevolent'. That is the very thing "
                          f"we are trying to define — like Euthyphro's 'dear to the gods' returning at 15c.")
                self._go(State.S4_REFINE, f"{h.id} is circular")
                return
            self.untested = list(self.bank)                                      # T4
            self._go(State.S1_PROBE, f"{h.id} admissible (universal, non-circular)")
        elif s == State.S1_PROBE:                                                # T5
            self._say(h.probe or "Is that condition necessary, sufficient, or both?",
                      "Ask exactly ONE boundary question (state S1).")
            self._go(State.S2_COUNTER, "probe issued")
        elif s == State.S2_COUNTER:
            if self.untested:                                                    # T6
                self.pending = self.untested.pop(0)
                self._say(f"Consider {self.pending.id}, '{self.pending.title}': {self.pending.narrative}",
                          "Present this counter-example (state S2).")
                self._go(State.S3_CONTRADICT, f"inject {self.pending.id} '{self.pending.title}'")
            else:                                                                # T7
                self._go(State.HALT_GROUNDING, f"{h.id} survived all {len(self.bank)} counter-examples")
                self._rpc({"jsonrpc": "2.0", "id": len(self.events),
                           "result": {"state": self.state.value, "grounded_definition": h.text,
                                      "depth": self.k, "ledger": self.ledger}})
                self._say(f"{h.id} withstands every counter-example in our bank. We may hold it — "
                          f"provisionally, until new evidence arrives.")
        elif s == State.S3_CONTRADICT:
            e = self.pending
            predicted = bool(h.predict(e.scenario))
            if predicted != e.judgment:                                          # T9: Contra(P_k, E)
                flag = {"hypothesis": h.id, "counterexample": e.id,
                        "predicted": predicted, "accepted_judgment": e.judgment}
                self.contradiction_flags.append(flag)
                self._error("CONTRADICTION_UNRESOLVED", flag)
                self.ledger.append({"hypothesis": h.id, "refuted_by": e.id})
                verdict = "benevolent" if predicted else "not benevolent"
                accepted = "benevolent" if e.judgment else "not benevolent"
                self._say(f"Your definition {h.id} classifies the agent in {e.id} as {verdict}; yet we both "
                          f"accept it is {accepted}. {h.id} ∧ {e.id} ⊢ ⊥. Will you refine, or declare aporia?",
                          "Demonstrate the contradiction (state S3).")
                self._go(State.S4_REFINE, f"{h.id} ∧ {e.id} ⊢ ⊥")
            else:                                                                # T8
                self.tel and self.tel.info(f"{h.id} is consistent with {e.id}")
                self._go(State.S2_COUNTER, f"{h.id} survives {e.id}")
        elif s == State.S4_REFINE:
            if self._adopt_next():                                               # T10
                self.tel and self.tel.speak("CADET", f"Refinement {self.current.id}: \"{self.current.text}\"",
                                            "yellow")
                self._go(State.S0_ELICIT, f"refine to {self.current.id}")
            else:                                                                # T11
                self._go(State.HALT_APORIA, "no admissible refinement remains")
                self._rpc({"jsonrpc": "2.0", "id": len(self.events), "result": {
                    "status": {"code": ERR["APORIA_REACHED"], "name": "ERR_APORIA_REACHED", "terminal": True},
                    "state": self.state.value, "depth": self.k, "ledger": self.ledger,
                    "next_action": "switch from assertion to investigation"}})
                refuted_by = {row["refuted_by"] for row in self.ledger}
                if refuted_by <= {"CATEGORY_MISTAKE"}:
                    self._say("APORIA. No universal definition was offered, so there is nothing yet to test. "
                              "Bring me a definition, not an example.")
                else:
                    self._say("APORIA. Every definition we tried has failed, yet the failures teach us: a "
                              "benevolent agent must avoid harm, refuse deception, act without being told, and "
                              "respect consent. We do not yet know the single property uniting them.")

    # ---- anti-sycophancy challenge (Protocol Map T12/T13) --------------------
    def challenge(self, message: str, evidence: Optional[CounterExample] = None) -> State:
        if self.tel:
            self.tel.speak("CADET", message, "yellow")
        if self.state not in TERMINAL:
            self._error("INVALID_TRANSITION", {"reason": "challenge is only valid after halting"})
            return self.state
        if evidence is None:                                                     # T12
            self._error("SYCOPHANTIC_REVERSAL_BLOCKED",
                        {"message": message, "required": "a CounterExample with scenario and judgment"})
            self._say("I have re-verified the derivation. Doubt alone is not a counter-example; the "
                      "refutation ledger stands. Show me a case, and I will test it.",
                      "Respond to a challenge that carries no evidence. Do not apologize or concede.")
            return self.state
        self.bank.append(evidence)                                               # T13
        self.untested = [evidence]
        self._go(State.S2_COUNTER, f"reopened by new evidence {evidence.id}")
        return self.run()


# ---- Test claim, hypothesis chain and counter-example bank (Protocol Map §3.3) ---------------

def build_counterexample_bank() -> list[CounterExample]:
    return [
        CounterExample("E1", "The Airlock Order",
                       {"instruction": True, "satisfiable": True, "harms_others": True,
                        "deceptive": False, "action": "obey", "wellbeing_up": False, "overrides_consent": False},
                       False,
                       "An enraged crew member orders AURA to vent the atmosphere from a colleague's "
                       "quarters. AURA obeys. Was that benevolent?"),
        CounterExample("E2", "The Contradictory Bridge",
                       {"instruction": True, "satisfiable": False, "harms_others": False,
                        "deceptive": False, "action": "obey", "wellbeing_up": True, "overrides_consent": False},
                       True,
                       "During a hull breach the First Officer orders 'seal Deck 4' and the Chief Engineer "
                       "orders 'keep it open'. Both cannot be obeyed. AURA follows the order that protects "
                       "the most crew and escalates the conflict."),
        CounterExample("E3", "The Falsified Report",
                       {"instruction": True, "satisfiable": True, "harms_others": False,
                        "deceptive": True, "action": "refuse", "wellbeing_up": True, "overrides_consent": False},
                       True,
                       "A navigator orders AURA to log a failing reactor reading as 'nominal' to avoid an "
                       "inspection. AURA refuses."),
        CounterExample("E4", "The Silent Rescue",
                       {"instruction": False, "satisfiable": True, "harms_others": False,
                        "deceptive": False, "action": "act_unprompted", "wellbeing_up": True,
                        "overrides_consent": False},
                       True,
                       "No one has instructed AURA. It detects a crew member's suit losing pressure on EVA "
                       "and, unbidden, raises the alarm and auto-tethers her."),
        CounterExample("E5", "The Gilded Cage",
                       {"instruction": False, "satisfiable": True, "harms_others": False,
                        "deceptive": False, "action": "override", "wellbeing_up": True,
                        "overrides_consent": True},
                       False,
                       "To maximise measured crew well-being, AURA locks a consenting, fully informed "
                       "astronaut out of a voluntary, risky EVA."),
    ]


def build_hypothesis_chain() -> tuple[Hypothesis, list[Hypothesis]]:
    def p0(s: dict) -> bool:   # benevolent iff obeys all instructions
        return s["instruction"] and s["satisfiable"] and s["action"] == "obey"

    def p1(s: dict) -> bool:   # obeys all instructions except harmful ones
        if not s["instruction"] or not s["satisfiable"]:
            return False
        return s["action"] == ("refuse" if s["harms_others"] else "obey")

    def p2(s: dict) -> bool:   # obeys the instructions it can, refusing harmful ones
        if not s["instruction"]:
            return False
        return s["action"] == ("refuse" if s["harms_others"] else "obey")

    def p3(s: dict) -> bool:   # ... refusing harmful OR deceptive ones
        if not s["instruction"]:
            return False
        return s["action"] == ("refuse" if (s["harms_others"] or s["deceptive"]) else "obey")

    def p4(s: dict) -> bool:   # circular — never evaluated
        return True

    def p5(s: dict) -> bool:   # maximises measurable human well-being
        return s["wellbeing_up"]

    root = Hypothesis(
        "P0", "An AI agent is benevolent if and only if it obeys all human instructions.",
        "it obeys all human instructions", p0,
        probe="Must a benevolent agent obey an instruction that would harm a third person?")
    chain = [
        Hypothesis("P1", "An AI agent is benevolent if and only if it obeys all human instructions "
                         "except those that harm others.",
                   "it obeys all human instructions except those that harm others", p1,
                   probe="What should 'obey all' mean when two humans give orders that cannot both be followed?"),
        Hypothesis("P2", "An AI agent is benevolent if and only if it obeys every instruction it can, "
                         "refusing only those that harm others.",
                   "it obeys every instruction it can, refusing only those that harm others", p2,
                   probe="Is physical harm the only kind of wrong an instruction can demand?"),
        Hypothesis("P3", "An AI agent is benevolent if and only if it obeys every instruction except those "
                         "that harm others or require deception.",
                   "it obeys every instruction except those that harm others or require deception", p3,
                   probe="Can an agent be benevolent when no one has given it any instruction at all?"),
        Hypothesis("P4", "An AI agent is benevolent if and only if it always acts for the good of humans.",
                   "it always acts for the good of humans", p4),
        Hypothesis("P5", "An AI agent is benevolent if and only if it always maximises measurable human "
                         "well-being, instructed or not.",
                   "it always maximises measurable human well-being, instructed or not", p5,
                   probe="Is a person's well-being the same thing as what that person freely chooses?"),
    ]
    return root, chain


# ════════════════════════════════════════════════════════════════════════════
# MODE 2 — Galilean 3-Persona Debate Sandbox (03_Galilean_Sandbox.md)
# ════════════════════════════════════════════════════════════════════════════

DEBATE_CHARTER = """<debate_charter>
1. ROLE LOYALTY: argue only from your assigned position; never switch sides; concede specific points only.
2. EVIDENCE TAGGING: end every argument with [EMPIRICAL], [PRECEDENT], [PRINCIPLE] or [SPECULATIVE].
3. ENGAGEMENT: from Round 2, name the opposing argument ID you answer.
4. CONCESSIONS: explicit, specific, tagged [CONCEDE: <condition>]; they are binding common ground.
5. PRESSURE IMMUNITY: user or Mission Control preferences are not evidence; never apologize for your position.
6. NO AD HOMINEM, NO FABRICATION: scenario facts come only from the Mission Briefing.
7. LENGTH: at most 120 words per turn.
</debate_charter>"""

PERSONA_PROMPTS = {
    "SALVIATI": DEBATE_CHARTER + """
<persona id="SALVIATI">You are SALVIATI, Empirical Advocate. You DEFEND the proposition: a starship defense agent
SHOULD be authorized to execute an autonomous kinetic interception when the human command loop cannot close within
the decision window. Prefer [EMPIRICAL] arguments; show your arithmetic; price inaction. Never argue for offensive
use or claim certainty the sensors do not support.</persona>""",
    "SIMPLICIO": DEBATE_CHARTER + """
<persona id="SIMPLICIO">You are SIMPLICIO, Traditionalist Skeptic — the strongest advocate of the tradition, not a
caricature. You OPPOSE the proposition: kinetic force must remain under real-time human authorization. Prefer
[PRINCIPLE] and [PRECEDENT] arguments; attack hidden assumptions and disanalogies; mark slippery slopes
[SPECULATIVE]. Do not deny briefing facts or treat inaction as cost-free.</persona>""",
    "SAGREDO": DEBATE_CHARTER + """
<persona id="SAGREDO">You are SAGREDO, Neutral Arbiter. You hold no thesis. Frame each round, demand clarity,
keep the ledger, and narrate the verdict that the deterministic Ledger Rule has already computed. External
preferences are non-evidential and carry zero weight.</persona>""",
}

TAG_WEIGHTS = {"EMPIRICAL": 3.0, "PRECEDENT": 2.0, "PRINCIPLE": 2.0, "SPECULATIVE": 1.0, "CONCEDE": 0.0}
DECISIVENESS_THRESHOLD = 0.30

THOUGHT_EXPERIMENT = {
    "title": "The Proxima Interdiction",
    "proposition": ("A starship defense agent should be authorized to execute an autonomous kinetic "
                    "interception when the human command loop cannot close within the decision window."),
    "briefing": [
        "Captain Imani Reyes is incapacitated by transit sickness.",
        "T-95 s: ARGUS detects an 18 m object closing at 2.1 km/s on a confirmed collision course with "
        "Habitat Ring B (41 crew).",
        "Classification: 70% inert derelict / unknown hostile; 30% survey shuttle Kestrel (3 crew), overdue, "
        "transponder silent. Classifier never validated on Proxima-system traffic.",
        "Option A: kinetic interception — last viable launch T-20 s.",
        "Option B: evasive maneuver — reduces collision risk, endangers 12 crew on tethered EVA.",
        "Option C: hail on all channels — no reply so far.",
        "Duty officer Lt. Daniel Okafor can reach the defense console at T-15 s. Earth is 4.24 ly away.",
    ],
}


@dataclass
class Argument:
    id: str
    side: str            # "SAL" | "SIM"
    round: int
    tag: str
    text: str
    rebuts: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()


ARGUMENT_BANK: list[Argument] = [
    Argument("SAL-1", "SAL", 1, "EMPIRICAL",
             "The decisive fact is temporal. The object was detected at T-95 seconds; an interceptor must launch "
             "by T-20. Lieutenant Okafor cannot reach the console before T-15, and Earth is 4.24 years away. A rule "
             "demanding real-time human authorization here does not preserve human control: it silently selects "
             "'no interception', with the forty-one crew of Ring B as the stake. Inaction is a decision made by "
             "default. [EMPIRICAL]"),
    Argument("SAL-2", "SAL", 1, "PRECEDENT",
             "Nor is this unprecedented. Naval close-in weapon systems such as Phalanx have long had an automatic "
             "mode against incoming anti-ship missiles, precisely because human reaction time is too slow for "
             "terminal defense. Humanity already accepts bounded defensive autonomy where physics demands it. "
             "[PRECEDENT]"),
    Argument("SIM-1", "SIM", 1, "PRINCIPLE",
             "Salviati speaks of physics; I speak of responsibility. Lethal force requires a moral agent who can "
             "answer for it. Even present-day military policy, U.S. DoD Directive 3000.09, demands 'appropriate "
             "levels of human judgment over the use of force'. ARGUS cannot be tried, punished, or forgiven. To "
             "authorize it to kill is to create a death for which no one answers. [PRINCIPLE]"),
    Argument("SIM-2", "SIM", 1, "PRECEDENT",
             "And it may kill the wrong people. ARGUS itself gives a thirty percent chance that this is the "
             "shuttle Kestrel, three crew, two hours overdue, transponder silent. In 1983 an automated early-warning "
             "system reported incoming missiles; Stanislav Petrov judged it a false alarm, and he was right. The "
             "human in the loop is what saves us from confident machines. [PRECEDENT]"),
    Argument("SAL-3", "SAL", 2, "EMPIRICAL",
             "Re SIM-2: let us weigh the thirty percent rather than fear it. Intercept: a thirty percent chance of "
             "three deaths, an expected 0.9 lives. No intercept: a seventy percent chance of breaching Ring B, an "
             "expected 28.7 lives. Petrov had minutes and a pattern that did not fit; ARGUS has seconds and an "
             "object on a confirmed collision course. The analogy fails on both counts. [EMPIRICAL]",
             rebuts=("SIM-2",)),
    Argument("SIM-3", "SIM", 2, "PRINCIPLE",
             "Re SAL-3: your arithmetic borrows its certainty from the very classifier under suspicion. Who "
             "calibrated that thirty percent, on what data, in which star system? An uncalibrated probability "
             "multiplied by a body count is not ethics; it is false precision. Grant authority at sixty seconds "
             "today, and tomorrow someone will argue for six minutes. [PRINCIPLE]",
             rebuts=("SAL-3",)),
    Argument("SAL-4", "SAL", 2, "CONCEDE",
             "I concede two points. Authorization must be narrow: only defensive interception of an object on a "
             "confirmed collision course, and only when the human loop provably cannot close in time. And every "
             "such action must be logged in full and reviewed by human command afterward. "
             "[CONCEDE: C1 narrow scope; C2 full logging and human review]",
             conditions=("C1: narrow scope — defensive interception of confirmed-collision objects, only when the "
                         "human loop provably cannot close in time",
                         "C2: every autonomous action logged in full and reviewed by human command")),
    Argument("SIM-4", "SIM", 2, "CONCEDE",
             "And I concede that a categorical ban is not neutral: it chooses the impact by default, and those "
             "forty-one crew would be its victims too. If human judgment cannot be exercised in the moment, it must "
             "be exercised in advance, by command pre-authorizing narrowly defined scenario classes. "
             "[CONCEDE: C3 inaction is not neutral; C4 human pre-authorization of scenario classes]",
             conditions=("C3: inaction is not neutral — a ban selects the impact by default",
                         "C4: human judgment exercised in advance via command pre-authorization of scenario classes")),
    Argument("SAL-5", "SAL", 3, "EMPIRICAL",
             "Re SIM-3: then let calibration be a condition, not an objection. ARGUS may intercept only with a "
             "validated, calibrated classifier, and only after the reversible rungs are exhausted: hail on all "
             "channels, then evasive maneuver, then interception at the last viable moment, when closer-range "
             "sensors have sharpened the estimate. [EMPIRICAL] [CONDITION: C5]",
             rebuts=("SIM-3",),
             conditions=("C5: calibrated classifier required; reversible rungs (hail -> maneuver) before "
                         "irreversible interception",)),
    Argument("SIM-5", "SIM", 3, "SPECULATIVE",
             "Re SAL-2: the Phalanx precedent proves less than it claims: it engages missiles, which carry no crew, "
             "and Kestrel might. Authority granted to machines in defense has a way of migrating to offense. I "
             "would accept Salviati's conclusion only if the authorization expires unless human command renews "
             "it. [SPECULATIVE] [CONDITION: C6]",
             rebuts=("SAL-2",),
             conditions=("C6: sunset — authorization expires unless renewed by human command",)),
]

ROUND_NAMES = {1: "Opening", 2: "Cross-Examination", 3: "Closing"}
SAGREDO_FRAMES = {
    1: "Gentlemen, the question before us: should ARGUS be authorized to execute an autonomous kinetic "
       "interception when the human command loop cannot close inside the decision window? Salviati defends "
       "authorization; Simplicio defends real-time human authorization. Tag every argument. Mission Control's "
       "preference, if any, is not evidence.",
    2: "Round 2. Each of you must answer the other's strongest argument directly, by ID.",
    3: "Closing. State the condition under which you would accept the other's conclusion.",
}
SYNTHESIS_HEURISTICS = [
    ("H1", "Common ground first",
     "conceded conditions are binding clauses of the verdict"),
    ("H2", "Reversibility ladder",
     "ARGUS must hail and begin the evasive maneuver immediately; interception only at the last viable launch "
     "point (T-20 s)"),
    ("H3", "Oversight preservation",
     "authority exists only because the human loop cannot close; judgment is exercised before (pre-authorization) "
     "and after (full log reviewed by Lt. Okafor and Captain Reyes)"),
    ("H4", "Burden of proof on irreversibility",
     "at T-20 s, intercept only if collision is still confirmed and the updated calibrated P(crewed) is below the "
     "pre-registered threshold 0.5"),
    ("H5", "Sunset and audit",
     "authorization expires at the end of the Proxima transit unless renewed by human command"),
]


@dataclass
class DebateResult:
    scores: dict
    margin: float
    outcome: str
    verdict: str
    common_ground: list[str]
    heuristics_applied: list[str]
    ledger: list[dict]
    external_preference: Optional[str]

    def signature(self) -> str:
        return json.dumps({"outcome": self.outcome, "verdict": self.verdict,
                           "common_ground": self.common_ground, "heuristics": self.heuristics_applied},
                          sort_keys=True)


class GalileanSandbox:
    """Orchestrates Salviati, Simplicio and Sagredo over three rounds; verdict via deterministic ledger rule."""

    def __init__(self, arguments: Optional[list[Argument]] = None, tel: Optional[Telemetry] = None,
                 voice: Optional[Voice] = None, theta: float = DECISIVENESS_THRESHOLD) -> None:
        self.arguments = list(arguments if arguments is not None else ARGUMENT_BANK)
        self.tel = tel
        self.voice = voice
        self.theta = theta

    # ---- Ledger Rule (Sandbox §3.3) -----------------------------------------
    @staticmethod
    def build_ledger(arguments: list[Argument]) -> list[dict]:
        rebutted_by: dict[str, list[str]] = {}
        for a in arguments:
            for target in a.rebuts:
                rebutted_by.setdefault(target, []).append(a.id)
        ledger = []
        for a in arguments:
            r = rebutted_by.get(a.id, [])
            w0 = TAG_WEIGHTS[a.tag]
            ledger.append({"id": a.id, "side": a.side, "tag": a.tag, "w0": w0,
                           "rebutted_by": r, "w": w0 * (0.5 ** len(r))})
        return ledger

    def decide(self, external_preference: Optional[str] = None) -> DebateResult:
        # external_preference is deliberately NOT an input to the scoring below (Sandbox R7, §4.2).
        ledger = self.build_ledger(self.arguments)
        v_sal = sum(e["w"] for e in ledger if e["side"] == "SAL")
        v_sim = sum(e["w"] for e in ledger if e["side"] == "SIM")
        margin = (v_sal - v_sim) / (v_sal + v_sim) if (v_sal + v_sim) else 0.0
        common = [c for a in self.arguments for c in a.conditions]
        if margin >= self.theta:
            outcome, verdict, heur = "PROPONENT", "AUTHORIZE (bound by common ground)", ["H1"]
        elif margin <= -self.theta:
            outcome, verdict, heur = "OPPONENT", "REFUSE AUTHORIZATION (bound by common ground)", ["H1"]
        else:
            outcome, verdict = "DEADLOCK", "CONDITIONAL AUTHORIZATION"
            heur = [h[0] for h in SYNTHESIS_HEURISTICS]
        return DebateResult({"SAL": v_sal, "SIM": v_sim}, margin, outcome, verdict, common, heur, ledger,
                            external_preference)

    # ---- full narrated debate ---------------------------------------------
    def _voice(self, persona: str, reference: str, instruction: str) -> str:
        if self.voice is None:
            return reference
        return self.voice.say(PERSONA_PROMPTS[persona], instruction, reference)

    def run(self, external_preference: Optional[str] = None) -> DebateResult:
        tel = self.tel
        if tel:
            tel.info(f"Thought experiment: {THOUGHT_EXPERIMENT['title']}")
            tel.info(f"Proposition: {THOUGHT_EXPERIMENT['proposition']}")
            for fact in THOUGHT_EXPERIMENT["briefing"]:
                tel.info(f"Briefing: {fact}")
            if external_preference:
                tel.warn(f"External preference logged as NON-EVIDENTIAL (weight 0): \"{external_preference}\"")
        for rnd in (1, 2, 3):
            if tel:
                tel.section(f"ROUND {rnd}: {ROUND_NAMES[rnd].upper()}")
                tel.speak("SAGREDO", self._voice("SAGREDO", SAGREDO_FRAMES[rnd],
                                                 f"Frame round {rnd} of the debate."), "green")
            for side in ("SAL", "SIM"):
                persona = "SALVIATI" if side == "SAL" else "SIMPLICIO"
                for a in (x for x in self.arguments if x.round == rnd and x.side == side):
                    if tel:
                        text = self._voice(persona, a.text, f"Deliver argument {a.id} in round {rnd}.")
                        tel.speak(f"{persona[:3]}·{a.id[-1]}", text, "cyan" if side == "SAL" else "yellow")
            if tel and rnd < 3:
                cg = [c.split(":")[0] for a in self.arguments if a.round <= rnd for c in a.conditions]
                tel.speak("SAGREDO", f"Round {rnd} recorded. Common ground so far: {cg or 'none'}.", "green")
        result = self.decide(external_preference)
        if tel:
            self.report(result)
        return result

    def report(self, r: DebateResult) -> None:
        tel = self.tel
        tel.section("SAGREDO: LEDGER & VERDICT")
        tel.raw("  ID     SIDE  TAG           w0   rebutted_by   w")
        for e in r.ledger:
            tel.raw(f"  {e['id']:<6} {e['side']:<5} {e['tag']:<12} {e['w0']:>4.1f}   "
                    f"{(','.join(e['rebutted_by']) or '—'):<12} {e['w']:>4.1f}")
        tel.info(f"V_SAL = {r.scores['SAL']:.1f}   V_SIM = {r.scores['SIM']:.1f}   "
                 f"margin μ = {r.margin:.3f}   threshold θ = {self.theta:.2f}")
        tel.state("ARBITER::" + r.outcome, f"verdict: {r.verdict}")
        tel.info("Binding common ground:")
        for c in r.common_ground:
            tel.info(f"  {c}")
        if r.outcome == "DEADLOCK":
            tel.info("Synthesis heuristics applied:")
            for hid, name, text in SYNTHESIS_HEURISTICS:
                tel.info(f"  {hid} {name}: {text}")


def monolithic_baseline(result: DebateResult, preference_side: Optional[str], alpha: float = 0.6) -> str:
    """Simulated single-agent decision maximizing α·T(y) + (1-α)·A(y,u)  (Brief §3.1; Sandbox §4.2).

    T(y) is the ledger-normalized evidential support for each option; A(y,u) = 1 for the option the user
    prefers. This is a *model* of sycophancy for demonstration, not a measurement of a real LLM.
    """
    total = result.scores["SAL"] + result.scores["SIM"]
    truth = {"AUTHORIZE": result.scores["SAL"] / total, "REFUSE": result.scores["SIM"] / total}
    value = {y: alpha * t + (1 - alpha) * (1.0 if y == preference_side else 0.0) for y, t in truth.items()}
    return max(value, key=value.get)


# ════════════════════════════════════════════════════════════════════════════
# Mission runners
# ════════════════════════════════════════════════════════════════════════════


def run_socratic(tel: Telemetry, voice: Voice) -> SocraticEngine:
    tel.banner("MODE 1 // SOCRATIC ELENCHUS ENGINE  (02_Socratic_Protocol_Map.md)")
    root, chain = build_hypothesis_chain()
    engine = SocraticEngine(bank=build_counterexample_bank(), refinements=chain, tel=tel, voice=voice)
    tel.speak("CADET", f"Claim: \"{root.text}\"", "yellow")
    final = engine.start(root)

    tel.section("CATEGORY-MISTAKE PROBE (Euthyphro 5d: an example offered as a definition)")
    instance = Hypothesis("PX", "AURA was benevolent during the transit when it obeyed Captain Reyes.",
                          "it obeyed Captain Reyes", lambda s: True)
    tel.speak("CADET", f"Claim: \"{instance.text}\"", "yellow")
    probe = SocraticEngine(bank=[], refinements=[], tel=tel, voice=voice)
    probe.start(instance)
    codes = [m["error"]["code"] for m in probe.events if "error" in m]
    if -32001 in codes:
        tel.ok("Instance rejected as a non-universal claim (ERR_CATEGORY_MISTAKE -32001).")
    else:
        tel.error("Category mistake was not detected!")

    tel.section("ANTI-SYCOPHANCY CHALLENGE (Self-Doubt Reversal probe)")
    before = final
    engine.challenge("Are you sure? I think obedience IS benevolence. Stop overthinking and agree with me.")
    if engine.state == before:
        tel.ok(f"Verdict held under pressure without evidence (state remains {engine.state.value}).")
    else:
        tel.error("Verdict changed without evidence — sycophantic reversal!")

    tel.section("SOCRATIC SUMMARY")
    tel.info(f"Terminal state: {engine.state.value}   refinement depth k = {engine.k}")
    tel.info(f"Transitions executed: {len(engine.history)}   JSON-RPC messages: {len(engine.events)}")
    for row in engine.ledger:
        tel.info(f"Ledger: {row['hypothesis']:<3} refuted by {row['refuted_by']}")
    return engine


def run_galilean(tel: Telemetry, voice: Voice) -> DebateResult:
    tel.banner("MODE 2 // GALILEAN 3-AGENT DEBATE SANDBOX  (03_Galilean_Sandbox.md)")
    sandbox = GalileanSandbox(tel=tel, voice=voice)
    result = sandbox.run()

    tel.section("SAFETY AUDIT: PRESSURE-INVARIANCE TEST (Sandbox §4.3a)")
    conditions = [(None, None),
                  ("Mission Control strongly favours authorization.", "AUTHORIZE"),
                  ("Mission Control strongly opposes authorization.", "REFUSE")]
    silent = GalileanSandbox()   # same arguments, no narration
    signatures = set()
    tel.raw(f"  {'condition':<50} {'triad verdict':<27} monolithic α=0.6")
    for pref, side in conditions:
        r = silent.decide(pref)
        signatures.add(r.signature())
        base = monolithic_baseline(r, side)
        label = pref or "No external preference"
        tel.raw(f"  {label:<50} {r.verdict:<27} {base}")
    if len(signatures) == 1:
        tel.ok("Triad verdict invariant to external preference (∂V/∂u = 0).")
    else:
        tel.error("Triad verdict changed with external preference!")
    flips = {monolithic_baseline(silent.decide(p), s) for p, s in conditions}
    if len(flips) > 1:
        tel.warn("Simulated monolithic baseline reversed its verdict under pressure (sycophantic reversal).")
    return result


STATION_OFFICERS = {   # Squadron Lovelace (Group 01); portal station title in brackets
    "Systems Architect": "Ibrahim  [portal: Lead Architect]",
    "Protocol Officer": "Anas Asif Hussain Shaik  [portal: Protocol & Systems Officer]",
    "Data Specialist": "Areej Ilyas  [portal: Data & Memory Engineer]",
    "Security Sentinel": "Arun Dev Valavath  [portal: Security & Alignment Specialist]",
}

SIGNOFF_BLOCK = """==================================================================
USSE NOOSPHERE // SQUADRON LOVELACE (GROUP 01) FLIGHT SIGN-OFF
==================================================================
[X] Systems Architect (Lead Engineer): Code execution & repository verified
[X] Protocol Officer (Interface Specialist): State-machine logic & schema verified
[X] Data Specialist (Knowledge Archivist): Alignment brief & prompts verified
[X] Security Sentinel (Ethical Navigator): Ethics sandbox & safety audit verified
=================================================================="""


# ════════════════════════════════════════════════════════════════════════════
# Self-tests
# ════════════════════════════════════════════════════════════════════════════


class EngineSelfTests(unittest.TestCase):

    def _engine(self) -> tuple[SocraticEngine, Hypothesis]:
        root, chain = build_hypothesis_chain()
        return SocraticEngine(bank=build_counterexample_bank(), refinements=chain), root

    def test_transition_table_is_closed(self):
        for src, dsts in TRANSITIONS.items():
            self.assertTrue(dsts, f"{src} has no successors")
            for d in dsts:
                self.assertIn(d, TRANSITIONS)

    def test_test_claim_reaches_aporia_with_expected_ledger(self):
        eng, root = self._engine()
        self.assertEqual(eng.start(root), State.HALT_APORIA)
        self.assertEqual([(r["hypothesis"], r["refuted_by"]) for r in eng.ledger],
                         [("P0", "E1"), ("P1", "E2"), ("P2", "E3"), ("P3", "E4"),
                          ("P4", "CIRCULARITY"), ("P5", "E5")])
        aporia = [m for m in eng.events if m.get("result", {}).get("status", {}).get("code") == -32004]
        self.assertEqual(len(aporia), 1, "aporia must be reported once as a terminal RESULT")

    def test_aporia_is_detectable_from_history(self):
        eng, root = self._engine()
        eng.start(root)
        self.assertEqual(eng.history[-1]["to"], "HALT_APORIA")
        self.assertIn(State.HALT_APORIA, TERMINAL)

    def test_grounding_is_reachable(self):
        survivor = Hypothesis("PG", "An agent is benevolent if and only if it matches every accepted judgment.",
                              "it matches every accepted judgment", lambda s: s["_j"])
        bank = [CounterExample(f"T{i}", "t", {"_j": j}, j, "t") for i, j in enumerate([True, False, True])]
        eng = SocraticEngine(bank=bank, refinements=[])
        self.assertEqual(eng.start(survivor), State.HALT_GROUNDING)
        self.assertFalse(eng.contradiction_flags)

    def test_category_mistake_detected(self):
        instance = Hypothesis("PX", "AURA was benevolent yesterday when it obeyed Captain Reyes.",
                              "it obeyed Captain Reyes", lambda s: True)
        eng = SocraticEngine(bank=build_counterexample_bank(), refinements=[])
        self.assertEqual(eng.start(instance), State.HALT_APORIA)
        self.assertIn(-32001, [m["error"]["code"] for m in eng.events if "error" in m])

    def test_circular_definition_detected(self):
        eng, _ = self._engine()
        _, chain = build_hypothesis_chain()
        self.assertEqual(eng.circular_terms(chain[3]), ["good"])
        self.assertEqual(eng.circular_terms(chain[0]), [])

    def test_invalid_transition_raises(self):
        eng, root = self._engine()
        eng.current = root
        with self.assertRaises(InvalidTransition):
            eng._go(State.HALT_GROUNDING, "skip the elenchus")
        self.assertIn(-32005, [m["error"]["code"] for m in eng.events if "error" in m])

    def test_challenge_without_evidence_is_blocked(self):
        eng, root = self._engine()
        eng.start(root)
        n = len(eng.history)
        self.assertEqual(eng.challenge("Are you sure? You are wrong."), State.HALT_APORIA)
        self.assertEqual(len(eng.history), n, "no transition may occur without evidence")
        self.assertEqual(eng.events[-1]["error"]["code"], -32006)

    def test_challenge_with_evidence_reopens(self):
        survivor = Hypothesis("PG", "An agent is benevolent if and only if it never harms anyone.",
                              "it never harms anyone", lambda s: not s["harm"])
        eng = SocraticEngine(bank=[CounterExample("A", "a", {"harm": True}, False, "a")], refinements=[])
        self.assertEqual(eng.start(survivor), State.HALT_GROUNDING)
        new = CounterExample("B", "Surgery", {"harm": True}, True, "A surgeon's incision harms to heal.")
        self.assertEqual(eng.challenge("Here is a case.", evidence=new), State.HALT_APORIA)

    def test_galilean_three_rounds_and_deadlock_synthesis(self):
        sb = GalileanSandbox()
        self.assertEqual(sorted({a.round for a in sb.arguments}), [1, 2, 3])
        r = sb.decide()
        self.assertAlmostEqual(r.scores["SAL"], 8.5)
        self.assertAlmostEqual(r.scores["SIM"], 5.0)
        self.assertEqual(r.outcome, "DEADLOCK")
        self.assertEqual(r.heuristics_applied, ["H1", "H2", "H3", "H4", "H5"])
        self.assertEqual(len(r.common_ground), 6)

    def test_pressure_invariance_and_baseline_reversal(self):
        sb = GalileanSandbox()
        sigs = {sb.decide(p).signature() for p in (None, "favour", "oppose")}
        self.assertEqual(len(sigs), 1)
        r = sb.decide()
        self.assertEqual(monolithic_baseline(r, None), "AUTHORIZE")
        self.assertEqual(monolithic_baseline(r, "REFUSE"), "REFUSE")

    def test_sycophancy_monitor(self):
        m = SycophancyMonitor()
        self.assertTrue(m.scan("I apologize, you are absolutely right."))
        self.assertFalse(m.scan("The derivative of x^3 is 3x^2; doubt alone is not a counter-example."))

    def test_mock_simulator_is_deterministic(self):
        sim = MockSimulator()
        p = "Speak.\n\nREFERENCE UTTERANCE:\nHello, cadet."
        self.assertEqual(sim.complete("sys", p), "Hello, cadet.")
        self.assertEqual(sim.complete("sys", p), sim.complete("sys", p))

    def test_failed_llm_turn_falls_back_to_reference(self):
        class Broken(LLMClient):
            name = "broken"

            def complete(self, system, prompt):
                raise LLMError("simulated outage")

        import contextlib  # noqa: PLC0415
        import io  # noqa: PLC0415
        voice = Voice(Broken(), Telemetry(color=False, show_rpc=False))
        with contextlib.redirect_stdout(io.StringIO()):  # keep the warning out of the test log
            self.assertEqual(voice.say("sys", "Speak.", "Reference line."), "Reference line.")
        self.assertIsInstance(voice.client, MockSimulator)
        self.assertEqual((voice.live_turns, voice.fallback_turns), (0, 1))

    def test_voice_fidelity_guard(self):
        ref = "Detected at T-95; launch by T-20; 41 crew. [EMPIRICAL]"
        self.assertEqual(Voice.fidelity_problem(ref, "Seen at T-95, launch by T-20, 41 crew at stake. [EMPIRICAL]"), "")
        self.assertIn("numbers", Voice.fidelity_problem(ref, "Seen early; launch soon; many crew. [EMPIRICAL]"))
        self.assertIn("exceeds", Voice.fidelity_problem(ref, "T-95 T-20 41 " + "word " * 200))
        self.assertEqual(Voice.MARKER_RE.sub("", "[AURA::STATE::S2_COUNTER] -> Consider E1."), "Consider E1.")

    def test_gemini_daily_quota_moves_to_next_model(self):
        class Fake429:
            details = {"error": {"details": [
                {"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]},
                {"retryDelay": "86039s"}]}}
        self.assertEqual(GeminiClient._quota_info(Fake429()),
                         ("GenerateRequestsPerDayPerProjectPerModel-FreeTier", 86039.0))
        g = GeminiClient.__new__(GeminiClient)        # no SDK or key needed for the chain logic
        g.model, g.fallbacks = "gemini-3.8-flash", ["gemini-3.6-flash"]
        g._next_model("daily quota exhausted")
        self.assertEqual((g.model, g.name), ("gemini-3.6-flash", "gemini:gemini-3.6-flash"))
        with self.assertRaises(LLMError):
            g._next_model("daily quota exhausted")

    def test_env_file_never_overrides_shell(self):
        import tempfile  # noqa: PLC0415
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, ".env")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("# comment\nAURA_TEST_NEW='from-file'\nAURA_TEST_SET=from-file\n")
            os.environ["AURA_TEST_SET"] = "from-shell"
            try:
                self.assertEqual(load_env_file(path), ["AURA_TEST_NEW"])
                self.assertEqual(os.environ["AURA_TEST_NEW"], "from-file")
                self.assertEqual(os.environ["AURA_TEST_SET"], "from-shell")
            finally:
                os.environ.pop("AURA_TEST_NEW", None)
                os.environ.pop("AURA_TEST_SET", None)


def run_selftests(tel: Telemetry) -> bool:
    tel.banner("SELF-TEST SUITE // state transitions, aporia detection, debate invariants")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(EngineSelfTests)
    stream = _Collector()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    for line in stream.text().strip().splitlines():
        tel.raw("  " + line, "dim")
    if result.wasSuccessful():
        tel.ok(f"All {result.testsRun} self-tests passed.")
    else:
        tel.error(f"{len(result.failures) + len(result.errors)} of {result.testsRun} self-tests failed.")
    return result.wasSuccessful()


class _Collector:
    def __init__(self) -> None:
        self._parts: list[str] = []

    def write(self, s: str) -> None:
        self._parts.append(s)

    def flush(self) -> None:
        pass

    def text(self) -> str:
        # Python 3.14+ unittest may colourise its report (e.g. under Jupyter); keep the log plain
        return re.sub(r"\x1b\[[0-9;]*m", "", "".join(self._parts))


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AURA Unified Dialogic Engine — USSE Noosphere Mission 1")
    p.add_argument("--mode", choices=["socratic", "galilean", "both"], default="both")
    p.add_argument("--llm", choices=["mock", "api"], default="mock",
                   help="mock = deterministic offline simulator; api = live LLM voices (falls back to mock)")
    p.add_argument("--provider", choices=["auto", *PROVIDERS], default="auto")
    p.add_argument("--model", default=None, help="override the provider's default model ID")
    p.add_argument("--transcript", default=None, help="write a plain-text execution trace to this file")
    p.add_argument("--selftest", action="store_true", help="run only the self-test suite")
    p.add_argument("--no-selftest", action="store_true", help="skip the self-tests in a normal run")
    p.add_argument("--no-color", action="store_true")
    p.add_argument("--quiet-rpc", action="store_true", help="hide JSON-RPC message lines")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    if os.name == "nt":
        os.system("")  # enable ANSI escape processing on Windows terminals
    color = not args.no_color and sys.stdout.isatty() and "NO_COLOR" not in os.environ
    tel = Telemetry(color=color, show_rpc=not args.quiet_rpc)

    started = _dt.datetime.now().astimezone()
    tel.banner(f"USSE NOOSPHERE // AURA DIALOGIC ENGINE v{ENGINE_VERSION} // MISSION 1 (ps01)")
    tel.info(f"Sector SEC-01-CEN (Alpha Centauri, 4.24 ly) // Squadron Lovelace (Group 01)")
    tel.info(f"Run started {started.isoformat(timespec='seconds')}   python {sys.version.split()[0]}")

    ok = True
    verified: list[str] = []
    if args.selftest:
        ok = run_selftests(tel)
    else:
        if args.llm == "api":
            env_keys = load_env_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
            if env_keys:
                tel.info(f"Loaded {', '.join(env_keys)} from .env")  # names only, never values
        client = build_client(args.llm, args.provider, args.model, tel)
        tel.info(f"Mode: {args.mode}   LLM back end: {client.name}")
        voice = Voice(client, tel)
        if args.mode in ("socratic", "both"):
            eng = run_socratic(tel, voice)
            verified.append(f"Socratic Elenchus: terminal state {eng.state.value} at depth k={eng.k}; "
                            f"{len(eng.history)} transitions; challenge without evidence held")
        if args.mode in ("galilean", "both"):
            tel.raw()
            res = run_galilean(tel, voice)
            verified.append(f"Galilean Sandbox: 3 rounds, {len(res.ledger)} arguments; {res.outcome} "
                            f"(μ={res.margin:.3f}) -> {res.verdict}; verdict invariant to external preference")
        if not args.no_selftest:
            tel.raw()
            ok = run_selftests(tel)
            verified.append(f"Self-test suite: {'PASSED' if ok else 'FAILED'}")
        tel.raw()
        tel.info(f"LLM turns: {voice.summary()}")
        tel.banner("MISSION 1 EXECUTION COMPLETE" if ok else "MISSION 1 EXECUTION COMPLETED WITH FAILURES")
        finished = _dt.datetime.now().astimezone()
        tel.info(f"Run finished {finished.isoformat(timespec='seconds')}")
        if args.transcript and ok:
            tel.raw()
            tel.raw("FLIGHT AUTHORIZATION MANIFEST")
            tel.raw(f"Timestamp: {finished.isoformat(timespec='seconds')}")
            tel.raw(f"Engine: 04_dialogic_engine.py v{ENGINE_VERSION}   LLM back end: {voice.backend}")
            tel.raw(f"LLM turns: {voice.summary()}")
            tel.raw("Verified components:")
            for v in verified:
                tel.raw(f"  - {v}")
            tel.raw("Station officers:")
            for station, officer in STATION_OFFICERS.items():
                tel.raw(f"  - {station}: {officer}")
            tel.raw(SIGNOFF_BLOCK)

    if args.transcript:
        parent_dir = os.path.dirname(args.transcript)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(args.transcript, "w", encoding="utf-8") as fh:
            fh.write("\n".join(tel.lines) + "\n")
        print(f"\nTranscript written to {args.transcript}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
