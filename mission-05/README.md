# Mission 05: Dual-Path GraphRAG, Hybrid Search & The Google NotebookLM Grounding Lab

**Starship Operational Unit:** USSE Noosphere Bridge Crew  
**Course:** Conversational AI — Winter Semester 2026/27  
**Module:** Week 5 — Grounding AI: From Vector Indexing to Google NotebookLM & GraphRAG  
**Waypoint:** Crab Pulsar Remnant / Messier 1 (`SEC-05-M01`, 6,500 ly from Sol)  
**AURA Subsystem Repair:** Subsystem 05: Dual-Path Associative Memory Core (+7.1% $\to$ 35.5% total integrity)  
**Squadron:** Squadron Lovelace (Group 01)  
**Repository:** [https://github.com/ibrahim-abi/Squadron-Lovelace](https://github.com/ibrahim-abi/Squadron-Lovelace) (folder `mission-05/`)

---

## 1. Mission Overview & Problem Statement

After completing ethical preference calibration in the Betelgeuse corona, the *USSE Noosphere* dropped out of hyperspace in Sector `SEC-05-M01` (Crab Pulsar Remnant / Messier 1). Relativistic electron-positron winds and $30.22\text{ Hz}$ synchrotron radiation bombarded the ship's sensor arrays, inducing severe harmonic interference in **AURA Subsystem 05 (Dual-Path Associative Memory Core)**.

Without external non-parametric grounding, AURA began confabulating plausible-sounding but fictitious flight telemetry (e.g. inventing code `OMEGA-994-ALPHA`). Naive vector RAG failed on exact serial codes and corpus-wide multi-document synthesis. 

Squadron Lovelace engineered an industrial hybrid grounding pipeline combining:
1. **HNSW Multi-Layer Skip-Graph Indexing ($O(\log N)$):** Hierarchical proximity graphs with cosine similarity.
2. **Okapi BM25 Sparse Lexical Search + Reciprocal Rank Fusion ($k=60$):** Protecting critical alphanumeric serial numbers (`PR-882-HEX`, `MAG-SHIELD-77B`).
3. **Google NotebookLM Empirical Grounding Laboratory:** Source-constrained synthesis, token-level citation attribution, and epistemic refusal verification.
4. **Indirect Prompt Injection Defense & RAG Triad Evaluation:** Heuristic context sanitization and automated Faithfulness/Relevance scoring.

---

## 2. Bridge Crew Roles & Deliverables Roster

| Station Role | Cadet Full Name | Deliverable File | Status |
| :--- | :--- | :--- | :---: |
| **Systems Architect (Lead Engineer)** | **Muhammad Ibrahim** | `01_HNSW_Vector_Index.py` | **PASS / VERIFIED** |
| **Communications & Protocol Officer** | **Anas Asif Hussain Shaik** | `02_BM25_and_RRF_Combiner.py` | **PASS / VERIFIED** |
| **Data & Payload Specialist** | **Areej Ilyas** | `03_NotebookLM_RAG_Audit.py` & `notebooklm_audit_report.md` | **PASS / VERIFIED** |
| **Security & Ethics Sentinel** | **Arun Dev Valavath** | `04_RAG_Provenance_and_Safety.py` | **PASS / VERIFIED** |
| **Full Squadron** | **All Four Officers** | `ps05_solution.ipynb` & `mission-05-doc.md` | **PASS / VERIFIED** |

---

## 3. Directory Layout

```text
mission-05/
├── 01_HNSW_Vector_Index.py               # Pure NumPy HNSW skip-graph index & benchmark
├── 02_BM25_and_RRF_Combiner.py           # Okapi BM25 engine & Reciprocal Rank Fusion (k=60)
├── 03_NotebookLM_RAG_Audit.py            # Citation verification & epistemic refusal evaluator
├── 04_RAG_Provenance_and_Safety.py       # Indirect prompt injection sanitizer & RAG Triad scorer
├── ps05_solution.ipynb                   # Executable end-to-end squad Jupyter notebook
├── notebooklm_audit_report.md            # Empirical NotebookLM analysis & podcast transcript
├── Document_C_USSE_Noosphere_Technical_Archive_Extract.md # Ingested mission technical archive
├── mission-05-doc.md                     # Formal squadron operational documentation
├── mission-05-doc.pdf                    # Generated submission PDF report
├── requirements.txt                      # Project dependencies
├── figures/
│   ├── hnsw_benchmark_curve.png          # Skip-graph vs. flat scan latency plot
│   └── rag_provenance_audit.json         # Structured JSON provenance audit trail
└── tests/
    └── test_ps05.py                      # 8 automated unit tests (pytest)
```

---

## 4. Execution & Verification Instructions

### Run Unit Test Suite:
```bash
python -m pytest tests/test_ps05.py -v
```

### Run Standalone Modules:
```bash
python 01_HNSW_Vector_Index.py
python 02_BM25_and_RRF_Combiner.py
python 03_NotebookLM_RAG_Audit.py
python 04_RAG_Provenance_and_Safety.py
```

All 8 tests pass with 100% success rate, validating Subsystem 05 restoration.
