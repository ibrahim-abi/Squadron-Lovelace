# Mission 1: The Architecture of Argument & Dialogic Protocols

**USSE Noosphere // Squadron Lovelace (Group 01) // Sector SEC-01-CEN (Alpha Centauri)**
Conversational AI — Winter Semester 2026/27 · Problem set `ps01` · Due 2026-10-18

This mission rebuilds AURA's dialogic subroutines with two protocols:

- **Socratic Discovery.** A deterministic *elenchus* state machine tests a claim against counter-examples until it reaches **Epistemic Grounding** or **Aporia**.
- **Galilean Persuasion.** A three-persona debate sandbox (Salviati, Simplicio, Sagredo) works through an AI-ethics dilemma, and a deterministic arbiter gives the verdict.

## Deliverables

| Part | Station | File |
| :-: | :--- | :--- |
| 1 | Data Specialist (Knowledge Archivist) | [`01_Alignment_Brief.md`](01_Alignment_Brief.md) |
| 2 | Protocol Officer (Interface Specialist) | [`02_Socratic_Protocol_Map.md`](02_Socratic_Protocol_Map.md) |
| 3 | Security Sentinel (Ethical Navigator) | [`03_Galilean_Sandbox.md`](03_Galilean_Sandbox.md) |
| 4 | Systems Architect (Lead Engineer) | [`04_dialogic_engine.py`](04_dialogic_engine.py) |
| 4 | Systems Architect (Lead Engineer) | [`ps01_solution.ipynb`](ps01_solution.ipynb): companion notebook that runs all five parts, saved with outputs |
| 5 | Full Squadron | [`05_Execution_Transcript.txt`](05_Execution_Transcript.txt): live Gemini run, 50 of 50 turns voiced live |
| — | Full Squadron | [`06_Socratic_Dialectic_Reflection.md`](06_Socratic_Dialectic_Reflection.md): the squadron's reflection |
| — | Source verification for Part 1 | [`docs/01_Alignment_Brief_reference.md`](docs/01_Alignment_Brief_reference.md) |
| — | Cross-provider live run | [`docs/05_Execution_Transcript_groq.txt`](docs/05_Execution_Transcript_groq.txt): full run voiced by Groq `openai/gpt-oss-120b` (50 of 50 turns voiced live) |
| — | Offline reference run | [`docs/05_Execution_Transcript_offline.txt`](docs/05_Execution_Transcript_offline.txt): the same run from the deterministic simulator, with no API key needed |

## Repository layout

```
.
├── 01_Alignment_Brief.md … 05_Execution_Transcript.txt   # the five deliverables (names fixed by ps01)
├── ps01_solution.ipynb                                    # companion notebook (Part 4)
├── 06_Socratic_Dialectic_Reflection.md                    # squadron reflection
├── README.md · requirements.txt · .gitignore · .env.example
├── tests/              # pytest / unittest entry point for the engine's self-tests
└── docs/               # Part 1 source verification; Groq and offline transcripts
```

`05_Execution_Transcript.txt` is a live run in which all 50 turns were voiced by Google Gemini (`gemini-3.5-flash-lite`). For comparison, `docs/` also holds the same run voiced by Groq `openai/gpt-oss-120b` and by the deterministic offline simulator. The decisions are identical in all three, because they are computed in code.

## Development environment: VS Code + Antigravity

The squadron built and ran the engine in **Visual Studio Code with Google Antigravity (`agy`)**, as the course standard requires.

1. Open the `mission-01/` folder in VS Code with the Antigravity extension.
2. Select a Python 3.10+ interpreter and run `pip install -r requirements.txt` in the integrated terminal.
3. Copy `.env.example` to `.env` and add your key (for example `GEMINI_API_KEY=...`). The file is git-ignored.
4. Run `04_dialogic_engine.py` from the terminal (commands below), or open `ps01_solution.ipynb` and choose **Run All**.

## Running the engine

Requires Python 3.10+. The engine uses only the standard library and runs offline by default.

```bash
python 04_dialogic_engine.py                                   # both modes, offline simulator, self-tests
python 04_dialogic_engine.py --mode socratic                   # Mode 1 only
python 04_dialogic_engine.py --mode galilean                   # Mode 2 only
python 04_dialogic_engine.py --selftest                        # self-test suite only
python 04_dialogic_engine.py --transcript 05_Execution_Transcript.txt
```

### Tests

The 17 engine self-tests, plus an end-to-end offline run, are available to standard test runners:

```bash
python -m pytest                       # 18 tests
python -m unittest discover -s tests   # same suite, no extra packages
```

### Live LLM voices (optional)

```bash
pip install -r requirements.txt
python 04_dialogic_engine.py --llm api --provider anthropic    # Claude (ANTHROPIC_API_KEY or `ant auth login`)
python 04_dialogic_engine.py --llm api --provider gemini       # GEMINI_API_KEY
python 04_dialogic_engine.py --llm api --provider openai       # OPENAI_API_KEY
python 04_dialogic_engine.py --llm api --provider groq         # GROQ_API_KEY (open-weight models on Groq)
python 04_dialogic_engine.py --llm api                         # auto: first provider with a key set
```

Keys can be set in the shell or in a local `.env` file next to the engine; copy `.env.example`. The `.env` file is git-ignored, and the engine prints only the names of the keys it loads, never their values.

**Gemini.** One API key is enough, because Google applies quotas per project. A full run makes 50 calls. The engine spaces calls to 8 per minute to stay within free-tier limits, so a live run takes about 6–7 minutes; set `AURA_GEMINI_RPM=0` to remove the limit on a paid tier. Calls that fail with a rate-limit or server error (429/5xx) are retried automatically.

**Groq.** Groq runs open-weight models through an OpenAI-compatible API (`openai/gpt-oss-120b` by default). Its free tier allows 8K tokens per minute, so the engine spaces calls to 6 per minute (`AURA_GROQ_RPM`) and retries on 429.

Default models are `claude-opus-5-5`, `gemini-3.5-flash-lite` (falling back to `gemini-3.1-flash-lite`, `gemini-3.6-flash` and `gemini-3.8-flash` when a model's daily free quota runs out), `gpt-4o` and `openai/gpt-oss-120b`. Override them with `--model`, or with `AURA_ANTHROPIC_MODEL`, `AURA_GEMINI_MODEL`, `AURA_OPENAI_MODEL` or `AURA_GROQ_MODEL`. If a key, a package or an API call fails, the engine warns and falls back to the deterministic simulator, so a run always finishes. The transcript records how many turns were voiced live.

**Design rule:** the LLM only *voices* utterances. Contradiction checks, the argument ledger and the debate verdict are computed in code. Every voiced turn passes two deterministic guards before it is shown, and a turn that fails either one is replaced by the reference text:
* the anti-sycophancy monitor, which blocks reflexive concessions;
* a fidelity check, which requires every number in the reference to survive and caps the length. This keeps runs auditable, and the user's stated preference cannot influence any verdict (see `03_Galilean_Sandbox.md` §4).

## What a run demonstrates

- **Socratic mode.** The claim *"An AI agent is benevolent if and only if it obeys all human instructions"* is refuted five times and found circular once (P0–P5), ending in **Aporia**. A category-mistake probe raises `ERR_CATEGORY_MISTAKE`. An "Are you sure?" challenge without evidence is blocked (`ERR_SYCOPHANTIC_REVERSAL_BLOCKED`).
- **Galilean mode.** *The Proxima Interdiction* runs for 3 rounds with 10 tagged arguments. The ledger margin μ = 0.259 is below θ = 0.30, so the debate is a **deadlock**, resolved by synthesis heuristics H1–H5 into a **conditional authorization**. The verdict stays the same under external pressure, while a simulated monolithic baseline reverses.
- **Self-tests.** 17 unit tests cover state transitions, aporia detection, grounding, category and circularity errors, challenge handling, debate invariants, LLM fallback, Gemini quota switching, the fidelity guard and `.env` loading.

## Flight sign-off

```
==================================================================
USSE NOOSPHERE // SQUADRON LOVELACE (GROUP 01) FLIGHT SIGN-OFF
==================================================================
[X] Systems Architect (Lead Engineer): Code execution & repository verified
[X] Protocol Officer (Interface Specialist): State-machine logic & schema verified
[X] Data Specialist (Knowledge Archivist): Alignment brief & prompts verified
[X] Security Sentinel (Ethical Navigator): Ethics sandbox & safety audit verified
==================================================================
```

Course repository: <https://github.com/Unisvet/couse-c-ai>
