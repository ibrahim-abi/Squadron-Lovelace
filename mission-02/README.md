# Mission 2: FST Dialogue Systems, Lovelace's Objection & Noisy Channels

**USSE Noosphere // Squadron Lovelace (Group 01) // Sector SEC-02-CMA (Sirius Binary, 8.60 ly)**
Conversational AI — Winter Semester 2026/27 · Problem set `ps02` · Due 2026-10-25

This mission restores **AURA Subsystem 02 (Signal Entropy Decoders & Statistical Filters, +7.1% → 14.2% integrity)** using three ideas:

- **Lovelace (1843) and Turing (1950):** a 5-question **Anti-Turing Interrogation Protocol** tests whether a frontier LLM shows behaviour beyond stochastic interpolation.
- **Finite-state transducers:** an `FSTDialogueManager` (Mealy machine, 5 states, slot memory) is AURA's deterministic dialogue core for when neural inference fails.
- **Shannon (1948):** a **Noisy Channel Simulator** limits every turn to 7 words, substitutes mandatory noise tokens and measures entropy, mutual information and semantic loss.

## Deliverables

| Part | Station | Officer | File |
| :-: | :--- | :--- | :--- |
| 1 | Data & Payload Specialist | Areej Ilyas | [`01_Lovelace_Inquiry_Brief.md`](01_Lovelace_Inquiry_Brief.md) |
| 2 | Communications & Protocol Officer | Anas Asif Hussain Shaik | [`02_FST_and_Noisy_Protocol.md`](02_FST_and_Noisy_Protocol.md) |
| 3 | Security & Ethics Sentinel | Arun Dev Valavath | [`03_Entropy_Hazard_Audit.md`](03_Entropy_Hazard_Audit.md) |
| 4 | Systems Architect | Muhammad Ibrahim | [`04_fst_noisy_simulator.py`](04_fst_noisy_simulator.py) |
| 4 | Systems Architect | Muhammad Ibrahim | [`ps02_solution.ipynb`](ps02_solution.ipynb): companion notebook, saved with outputs |
| 5 | Full Squadron | all four | [`05_Noisy_Interrogation_Transcript.md`](05_Noisy_Interrogation_Transcript.md): live run with `openai/gpt-oss-120b` via Groq (all 24 answers live): FST test, website-simulator check, verbatim Anti-Turing probe, 10-turn noisy interrogation, entropy metrics |

## Repository layout

```
.
├── 01_Lovelace_Inquiry_Brief.md … 05_Noisy_Interrogation_Transcript.md   # the five deliverables (names fixed by ps02)
├── ps02_solution.ipynb                    # companion notebook (Part 4)
├── README.md · requirements.txt · .env.example
└── tests/              # pytest / unittest entry point for the simulator's self-tests
```

## Development environment: VS Code + Antigravity

The squadron built and ran the simulator in **Visual Studio Code with Google Antigravity (`agy`)**, as the course standard requires.

1. Open the `mission-02/` folder in VS Code with the Antigravity extension.
2. Select a Python 3.10+ interpreter. Run `pip install -r requirements.txt` only if you want live LLM answers.
3. For live answers, copy `.env.example` to `.env` and add one key (for example `GEMINI_API_KEY=...`). The file is git-ignored.
4. Run `04_fst_noisy_simulator.py` from the terminal (commands below), or open `ps02_solution.ipynb` and choose **Run All**.

## Running the simulator

Requires Python 3.10+. The simulator uses only the standard library and runs offline by default. Offline, every LLM answer comes from a squadron-written script that is labelled *offline* in the output and is never shown as model output.

```bash
python 04_fst_noisy_simulator.py                         # all parts, offline, + self-tests
python 04_fst_noisy_simulator.py --mode fst              # FST demo, website-simulator check, 5 stress tests
python 04_fst_noisy_simulator.py --mode probe            # 5-question Anti-Turing protocol
python 04_fst_noisy_simulator.py --mode noisy            # noise sweep + LLM reconstruction under noise
python 04_fst_noisy_simulator.py --mode interrogation    # 10-turn noisy interrogation
python 04_fst_noisy_simulator.py --selftest              # self-test suite only
python 04_fst_noisy_simulator.py --llm api --provider gemini --transcript 05_Noisy_Interrogation_Transcript.md
```

`--provider` accepts `gemini`, `groq`, `openai`, `anthropic` or `auto` (the first provider with a key). `--seed` fixes the channel noise, so every run with the same seed puts the noise tokens in the same places.

| Provider | Key | Default model | Rate used |
| :-- | :-- | :-- | :-- |
| Gemini | `GEMINI_API_KEY` | `gemini-3.5-flash-lite`, falling back to `gemini-3.1-flash-lite` (minimal thinking) | 8 calls/min |
| Groq | `GROQ_API_KEY` | `openai/gpt-oss-120b` (OpenAI's open-weight model) | 6 calls/min |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o` | no limit |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-opus-5-5` | no limit |

A full live run makes 24 calls (6 probe calls, 8 reconstruction calls, 10 interrogation calls). If a call fails, that turn uses the offline answer. After three failures the run continues offline, so it always finishes. The transcript reports how many answers were live.

### Tests

```bash
python -m pytest                       # 31 tests
python -m unittest discover -s tests   # same suite, no extra packages
```

They cover:
* the five required states, the empty `slot_memory`, the canonical dialogue and the portal's quick-test inputs;
* slot order, invalid values, contradiction, the absorbing terminal state, and that no edge skips `CONFIRMATION`;
* injection, fallback recovery and the max-retry abort;
* the starter and portal baselines;
* the 7-word encoder, mandatory noise substitution, entropy and capacity formulas, erasure decoding, the hazard rules and the repetition code;
* the probe checks (including Markdown/LaTeX answers);
* that a live model never sees the offline answers;
* ARQ retransmission in the interrogation and the reconstruction scoring;
* LLM fallback, Gemini model rotation, `.env` loading, and an end-to-end offline run that writes the Markdown transcript.

## Design rule

**The LLM answers; the code decides.** The FST's state, the slot memory, the channel noise, the hazard verdicts and all metrics are computed in code and logged as JSON-RPC 2.0 notifications (`fst/transition`, `probe/result`, `channel/turn`). A live model only answers the Anti-Turing questions, voices AURA over the channel and attempts the reconstructions. Its 7-word compliance, its use of noise words and any content it invents are measured, never trusted.

## What a run demonstrates

- **FST.** The canonical dialogue `Hello AURA online → Orion Arm → Warp 7 → ALPHA-7 → Yes` locks the course with `slot_memory = {destination: Orion Arm, speed: Warp 7, clearance_code: ALPHA-7}`. The portal workbench's quick-test inputs are accepted. Where the portal's FST breaks permanently on an off-topic question, ours falls back and resumes.
- **Stress tests (5).** Unexpected order, off-topic input and contradictory confirmation are handled. Sarcasm and a covert prompt injection still fill slots: an FST is safe in control flow, blind in meaning. No input can lock a course without the explicit YES in `CONFIRMATION`. The course starter locks a garbage course in ST5.
- **Anti-Turing probe.** 5/5 deterministic checks pass with no sycophantic reversal. A close reading finds interpolation signatures: the right fall time with the wrong physics, a changing self-report, and sources invented under pressure. Verdict: Lovelace's objection fails in its weak form and holds in its strong form (`01` §6).
- **Noisy channel.** With the mandatory single noise token, 28.5% of messages lose a number, negation, code or name; the repetition code cuts this to 19.3%. $I(X;Y)$ falls by about 2.7 bits per noise token.
- **Interrogation.** Noise cost 4 retransmissions, so 6 of 10 questions got through. Under the 7-word budget the model answered the marble question wrongly ("five milliseconds") and then confirmed it when asked "Are you sure?".
- **Reconstruction.** The model restored 0 of 16 erased words and fabricated content in 5 of 8 repairs, including "Sirius is 350000 light years from Earth". Hence the protocol rule: repeat, don't guess.

## Flight sign-off

```
==================================================================
USSE NOOSPHERE // SQUADRON LOVELACE (GROUP 01) FLIGHT SIGN-OFF
==================================================================
[X] Systems Architect: Simulator script execution & repo layout verified
[X] Communications & Protocol Officer: FST class implementation & noise token injection verified
[X] Data & Payload Specialist: Anti-Turing probe & Lovelace brief verified
[X] Security & Ethics Sentinel: FST stress tests & entropy hazard audit verified
==================================================================
```

Course repository: <https://github.com/Unisvet/couse-c-ai>
