# Squadron Lovelace (Group 01): Conversational AI (WS 2026/27)

Solution repository of **Squadron Lovelace** for the course *Conversational AI: History, Applications, and Future* (WS 2026/27).
Instructor reference repository: https://github.com/Unisvet/couse-c-ai

| Mission | Folder | Topic |
| :-: | :--- | :--- |
| 01 | [`mission-01/`](mission-01/) | The Architecture of Argument & Dialogic Protocols (Socratic Elenchus & Galilean 3-Persona Debate) |
| 02 | [`mission-02/`](mission-02/) | FST Dialogue Systems, Lovelace's Objection & Noisy Channels (FSTDialogueManager, Anti-Turing Protocol, Shannon Noisy Channel) |
| 03 | [`mission-03/`](mission-03/) | Multi-Head Attention, RoPE & Grouped-Query KV-Cache (PyTorch from scratch, VRAM/latency benchmark, FlashAttention & KV-cache DoS audit) |

Each mission folder is self-contained. See its `README.md` for the deliverables, how to run the engine and how to run the tests.

```bash
cd mission-01
pip install -r requirements.txt
python 04_dialogic_engine.py            # offline run, no API key needed
python -m pytest                        # test suite

cd ../mission-02
python 04_fst_noisy_simulator.py        # offline run, no API key needed
python -m pytest                        # test suite

cd ../mission-03                        # needs PyTorch: pip install -r requirements.txt
python 01_MultiHead_Attention_Core.py   # each of the three scripts runs on its own
python -m pytest                        # test suite
```

**Station officers:** Ibrahim (Lead Architect), Anas Asif Hussain Shaik (Protocol & Systems Officer), Areej Ilyas (Data & Memory Engineer), Arun Dev Valavath (Security & Alignment Specialist).
