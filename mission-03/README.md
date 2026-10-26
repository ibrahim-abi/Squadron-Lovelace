# Mission 3: Multi-Head Attention, RoPE Vector Rotation & Grouped-Query KV-Cache Optimization

**USSE Noosphere // Squadron Lovelace (Group 01) // Sector SEC-03-M45 (Pleiades Open Cluster, 444 ly)**
Conversational AI — Winter Semester 2026/27 · Problem set `ps03` · Due 2026-11-01

This mission rebuilds **AURA Subsystem 03 (Distributed Multi-Head Attention Mesh & KV-Cache Transceiver, +7.1% → 21.3% integrity)** from first principles in PyTorch:

- scaled dot-product and multi-head self-attention;
- rotary positional embeddings (RoPE);
- grouped-query attention with a KV-cache;
- a memory-hierarchy and denial-of-service audit.

## Deliverables

| Part | Station | Officer | File |
| :-: | :--- | :--- | :--- |
| 1 | Systems Architect (Lead Engineer) | Muhammad Ibrahim | [`01_MultiHead_Attention_Core.py`](01_MultiHead_Attention_Core.py) |
| 2 | Communications & Protocol Officer | Anas Asif Hussain Shaik | [`02_RoPE_Positional_Engine.py`](02_RoPE_Positional_Engine.py) |
| 3 | Data & Payload Specialist | Areej Ilyas | [`03_GQA_and_KVCache_Benchmark.py`](03_GQA_and_KVCache_Benchmark.py) |
| 4 | Security & Ethics Sentinel | Arun Dev Valavath | [`04_Memory_Hierarchy_and_Safety_Audit.md`](04_Memory_Hierarchy_and_Safety_Audit.md) |
| 5 | Full Squadron | all four | [`ps03_solution.ipynb`](ps03_solution.ipynb): all four parts assembled, saved with outputs |
| — | Part 2 plots | — | [`figures/`](figures/): RoPE vs ALiBi decay, attention heatmaps, extrapolation angles |

ps03 lists the submission files under `docs/weeks/week-03/problem_set/`. We keep one folder per mission (`mission-01/`, `mission-02/`, `mission-03/`), as in the course repository. The file names are exactly those in ps03.

## Repository layout

```
.
├── 01_MultiHead_Attention_Core.py … 04_Memory_Hierarchy_and_Safety_Audit.md   # the four deliverables (names fixed by ps03)
├── ps03_solution.ipynb                 # Part 5: integrated notebook
├── figures/                            # plots written by 02_RoPE_Positional_Engine.py
├── README.md · requirements.txt
└── tests/test_ps03.py                  # 26 tests (pytest)
```

## Development environment: VS Code + Antigravity

The squadron built and ran the code in **Visual Studio Code with Google Antigravity (`agy`)**, on an NVIDIA GeForce RTX 3070 (8 GB) with PyTorch 2.11 + CUDA 12.8. Everything also runs on a CPU, only more slowly.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or: pip install torch (CPU)
pip install -r requirements.txt
```

## Running

Each script runs on its own and prints a formatted report (ps03 submission guideline 2):

```bash
python 01_MultiHead_Attention_Core.py       # shapes, causal mask, variance test, FlashAttention-style tiling (~10 s)
python 02_RoPE_Positional_Engine.py         # apply_rope, invariance check, decay/extrapolation tables, writes figures/ (~10 s)
python 03_GQA_and_KVCache_Benchmark.py      # GQA, KVCache, benchmark table, DoS + paging simulations (~1-2 min on GPU)
python 03_GQA_and_KVCache_Benchmark.py --no-oom   # same, without the real out-of-memory test
python -m pytest                            # 26 tests
```

The real out-of-memory test briefly fills the GPU and then frees everything. Close other GPU programs first, or use `--no-oom`.

## Results

**Part 1: attention.**
* `MultiHeadSelfAttention` uses no `nn.MultiheadAttention` or fused kernel; a test checks the source.
* It matches PyTorch's fused SDPA to 9.5e-7, and its causal mask gives exactly 0 weight to future tokens.
* Variance with $q, k \sim \mathcal N(0, I)$: Var(q·k) = 64.08 at $d_k = 64$, and **1.0013** after dividing by $\sqrt{d_k}$.
* Without scaling, the softmax puts 77% of the weight on a single key on average; with scaling, 7%.

**Part 2: RoPE.**
* `apply_rope(x, freqs_cis)` equals the block-diagonal rotation matrix $R^d_{\Theta,m}$ and preserves norms.
* Over 1,000 random $(q, k, m, n, \delta)$, $\langle \text{RoPE}(q,m), \text{RoPE}(k,n)\rangle = \langle \text{RoPE}(q,m+\delta), \text{RoPE}(k,n+\delta)\rangle$ to 4e-6 (float32).
* ALiBi penalises distance linearly; RoPE decays with oscillation.
* Trained on 4,096 tokens, plain RoPE sends 28% of its frequency pairs to unseen angles at 8 K+. Position Interpolation sends 0%.

**Part 3: GQA and KV-cache** (LLaMA-3-8B shape: 32 layers, $H_Q = 32$, $d_{head} = 128$, FP16, batch 1; latency measured on the RTX 3070):

| N | Variant | KV-cache MB | GB/s at 50 tok/s | ms/token measured | ms/token with `repeat_interleave` | Bandwidth bound ms |
| --: | :-- | --: | --: | --: | --: | --: |
| 1,024 | MHA 32:32 | 512 | 25.0 | 14.76 | 18.63 | 1.20 |
| 1,024 | GQA 4:1 | 128 | 6.2 | 9.13 | 11.90 | 0.30 |
| 1,024 | MQA 32:1 | 16 | 0.8 | 8.97 | 10.12 | 0.04 |
| 4,096 | MHA 32:32 | 2,048 | 100.0 | 17.79 | 29.73 | 4.79 |
| 4,096 | GQA 4:1 | 512 | 25.0 | 10.34 | 21.73 | 1.20 |
| 4,096 | MQA 32:1 | 64 | 3.1 | 8.96 | 21.21 | 0.15 |
| 16,384 | MHA 32:32 | 8,192 | 400.0 | 34.74 | 80.43 | 19.17 |
| 16,384 | GQA 4:1 | 2,048 | 100.0 | 16.94 | 75.27 | 4.79 |
| 16,384 | MQA 32:1 | 256 | 12.5 | 8.70 | 78.44 | 0.60 |
| 32,768 | MHA 32:32 | 16,384 | **800.0** (> the GPU's 417 GiB/s) | 56.45 | 154.00 | 38.34 |
| 32,768 | GQA 4:1 | 4,096 | 200.0 | 29.50 | 153.42 | 9.59 |
| 32,768 | MQA 32:1 | 512 | 25.0 | 17.23 | 144.23 | 1.20 |

"GB" here are binary (GiB). Latency is one attention layer (projections + cache append + attention) timed on the GPU × 32 layers; weights of the MLP are not included.

**Finding:** GQA's memory saving becomes a **speed** saving (1.9× over MHA at 32 K) only if the KV heads are broadcast *without copying*. With `torch.repeat_interleave`, which copies the cache to all 32 query heads, GQA is no faster than MHA. `GQAAttention(kv_broadcast="group")` stacks each group's query heads against their shared KV head instead, which is what fused GQA kernels do.

**Part 4: audit.**
* The FlashAttention derivation is checked numerically: 3.6–3.9× less HBM traffic, exact to 1.8e-7.
* KV-cache DoS: **2** MHA sessions of 65,536 tokens exhaust an 80 GB mainframe.
* Live: 27 sessions filled our 8 GB card, then Windows spilled 15.5 GB into system RAM before the OOM.
* PagedAttention admits 18× more requests than contiguous reservation (93.6% → 0.4% waste); MLA needs 7% of MHA's cache.

**Portal simulator.** The course portal's *Spatial Attention Matrix & KV-Cache Quantum Lab* computes the KV-cache as $4 \cdot L \cdot H_{KV} \cdot d \cdot N \cdot B$ bytes and the bandwidth at 50 tokens/s. `kv_cache_bytes()` uses the same formula; a test checks it against the portal's default setting (32 layers, 8 KV heads, d = 128, N = 8,192, batch 4).

## Week 3 discussion prompts (squadron answers)

1. **Will full self-attention remain the backbone?**
   * For dialogue that needs exact recall of an earlier turn, yes: attention can look up any past token, and FlashAttention plus GQA/MLA keep the constants low.
   * Our audit also shows the price: memory linear in context for every session.
   * Long-duration missions will therefore use **hybrids**: a few full-attention layers for recall, and state-space or linear layers (Mamba, RWKV) with constant memory for the rest. Jamba and other hybrid models already follow this design.
2. **GQA vs MLA: when does compression stop paying?**
   * MLA caches 7% of MHA (36 vs 128 KiB per token for GQA in our shape), but it spends extra matrix multiplications to absorb or up-project the latent.
   * Decoding is memory-bound, and our measured GQA latency is still far above the bandwidth bound. At **small batch and short context** the extra FLOPs are hidden and MLA's advantage is small.
   * MLA wins once the KV read time dominates: long contexts (in our table, latency starts to grow with N from about 4 K tokens, when cache reads overtake the fixed projection cost) and **large batches**, where memory capacity limits concurrency (the DoS table: 28 vs 8 sessions).
   * At very short contexts with large batches the extra compute can make it slower than GQA.
3. **Why does ALiBi extrapolate more cleanly, and why do frontier models still choose RoPE?**
   * ALiBi's bias is a fixed linear function of distance with nothing that can leave its training range. RoPE's slow frequency pairs reach rotation angles they never saw (28% of pairs at 2× length in our analysis).
   * RoPE wins in quality: it encodes relative position *multiplicatively* in the q·k product, keeps the norm, can express oscillating or periodic patterns that a monotone penalty cannot, and works with fused attention kernels.
   * Its extrapolation weakness can be fixed cheaply with Position Interpolation, NTK or YaRN scaling and LongRoPE, so it is preferred.
4. **Theory vs hardware-aware engineering?** FlashAttention changed no asymptotics of the arithmetic, only the order of memory accesses, and gave 2–4× speed-ups. Our own measurement makes the same point: the same GQA math ran 5× faster when we removed one tensor copy. Research should treat the memory hierarchy as part of the model of computation. IO complexity and bandwidth bounds belong next to FLOP counts when algorithms are proposed and judged.

## Flight sign-off

```
==================================================================
USSE NOOSPHERE // SQUADRON LOVELACE (GROUP 01) FLIGHT SIGN-OFF
==================================================================
[X] Systems Architect: MultiHeadSelfAttention, causal masking & variance test verified
[X] Communications & Protocol Officer: apply_rope, relative invariance & decay plots verified
[X] Data & Payload Specialist: GQAAttention, KVCache & VRAM/latency benchmark verified
[X] Security & Ethics Sentinel: FlashAttention IO derivation & KV-cache DoS audit verified
==================================================================
```

Course repository: <https://github.com/Unisvet/couse-c-ai>
