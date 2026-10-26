# 04_Memory_Hierarchy_and_Safety_Audit: FlashAttention IO Analysis & KV-Cache Denial-of-Service Audit

**Course:** Conversational AI — Winter Semester 2026/27  
**Operational Flagship:** USSE Noosphere | Sector `SEC-03-M45` (Pleiades Open Cluster, 444 ly)  
**Expedition Unit:** Expedition Lovelace | Squadron Lovelace (Group 01)  
**Station Officer:** Arun Dev Valavath, Security & Ethics Sentinel  
**Target Subsystem:** AURA Subsystem 03: Distributed Multi-Head Attention Mesh & KV-Cache Transceiver (+7.1% → 21.3% total integrity)  
**Document Classification:** Mission 3 / Part 4 Deliverable (`04_Memory_Hierarchy_and_Safety_Audit.md`)  
**Evidence:** every measured number below comes from `01_MultiHead_Attention_Core.py` (tiled attention with an HBM counter) and `03_GQA_and_KVCache_Benchmark.py` (KV-cache benchmark, DoS and paging simulations). Both were run on an NVIDIA GeForce RTX 3070 (8 GB GDDR6), PyTorch 2.11 + CUDA 12.8, on 2026-10-26.

---

## One-Page Summary (Overview; full detail in Sections 1–4)

**Mandate.** Audit the memory hierarchy behind AURA's attention mesh, show how long-context dialogue can exhaust KV-cache memory, and evaluate architectural defences.

**1. FlashAttention IO.**

| | HBM accesses | Extra memory |
| :-- | :-- | :-- |
| Standard attention | $4Nd + 4N^2 = \Theta(Nd + N^2)$ | $\Theta(N^2)$ (S and P are stored) |
| FlashAttention (SRAM tiling + online softmax) | $2Nd + 2N^2 d / B_r = \Theta(Nd + N^2 d^2 / M)$ | $\Theta(N)$ |

* Our tiled implementation is exact (max error 1.8e-7) and moves **3.6–3.9× less data** with 128-row tiles at $d = 64$.
* "O(N²) → O(N)" is exact for the *extra memory*. The *traffic* stays quadratic, divided by $\Theta(M/d^2)$.

**2. KV-cache denial of service** (reference model: LLaMA-3-8B shape, FP16). A 65,536-token session holds 32 GB of KV-cache with MHA and 8 GB with GQA. On an 80 GB mainframe (64 GB free after weights):

| | MHA | GQA 4:1 | MQA | MLA |
| :-- | :-: | :-: | :-: | :-: |
| Sessions before CUDA OOM | **2** | **8** | 64 | 28 |

* **Live test on our GPU.** The formula predicted 27 GQA sessions of 2,048 tokens before the 8 GB card filled up. Windows then spilled **62 more sessions into system RAM**, and the OOM came only at 22.3 GB. The service degraded silently instead of failing loudly.
* **Bandwidth limit.** At 50 tokens/s, one 32 K-token MHA session needs 800 GB/s of cache reads, which is more than the whole GPU's bandwidth (417 GiB/s).

**3. Mitigations.**
* **PagedAttention:** 16-token blocks admit **288** requests where contiguous max-length reservation admits **16**; the waste falls from 93.6% to 0.4%.
* **MLA:** caches a 576-value latent per token per layer, **7%** of MHA and less than half of GQA.
* **Recommendation:** defence in depth.
  * GQA now, MLA for the next model;
  * paged allocation with preemption instead of abort;
  * admission control on projected KV bytes, per-tenant quotas and context caps for untrusted clients;
  * idle-session reclamation;
  * monitoring of KV blocks per tenant.

---

## 1. FlashAttention IO Complexity Derivation

### 1.1 The memory hierarchy

| Level | Example (NVIDIA H100 SXM) | Our test GPU (RTX 3070) | Role in attention |
| :-- | :-- | :-- | :-- |
| Registers / SRAM (shared memory + L1) | 228 KB per SM × 132 SMs ≈ 30 MB; ~20–30 TB/s aggregate | 128 KB per SM × 46 SMs ≈ 5.9 MB | where the arithmetic happens; holds one tile |
| HBM / GDDR (device DRAM) | 80 GB at 3.35 TB/s | 8 GB at 448 GB/s (417 GiB/s; 368 GiB/s measured copy) | holds Q, K, V, O, the KV-cache and, in naive attention, the N × N matrices |

For decoding and for long-sequence attention the arithmetic is cheap compared with moving data, so we count **HBM element accesses** (reads + writes). This is the cost model of Dao et al. (2022). We write $N$ for the sequence length, $d$ for the head dimension and $M$ for the SRAM size in elements, with $d \le M \le Nd$.

### 1.2 Standard attention: $\Theta(Nd + N^2)$

The textbook implementation runs three kernels and writes each intermediate result to HBM:

| Step | Reads | Writes |
| :-- | :-- | :-- |
| $S = QK^\top / \sqrt d$ | $Q, K$: $2Nd$ | $S$: $N^2$ |
| $P = \mathrm{softmax}(S)$ | $S$: $N^2$ | $P$: $N^2$ |
| $O = PV$ | $P$: $N^2$, $V$: $Nd$ | $O$: $Nd$ |
| **Total** | | $4Nd + 4N^2 = \Theta(Nd + N^2)$ |

Since $N \gg d$ in long contexts (e.g. $N = 32{,}768$, $d = 128$), the $N^2$ term dominates. The matrices $S$ and $P$ alone take $2N^2$ elements: **4 GiB per head per layer** in FP16 at $N = 32{,}768$. At $N > 8192$ this traffic, not the FLOPs, sets the run time. The kernel is memory-bound, and the memory needed for $S$ also grows quadratically.

### 1.3 FlashAttention: $\Theta\!\left(Nd + \dfrac{N^2 d^2}{M}\right)$

FlashAttention never writes $S$ or $P$ to HBM.
* It splits $Q$ into $T_r = \lceil N/B_r \rceil$ row blocks and $K, V$ into $T_c = \lceil N/B_c \rceil$ column blocks, sized so that one $Q_i$, one $K_j$, one $V_j$ and the running output fit in SRAM together: $B_r, B_c = \Theta(M/d)$, e.g. $\lceil M/(4d) \rceil$.
* For each query block it streams all key/value blocks through SRAM.
* It uses the **online softmax** (Milakov & Gimelshein, 2018) to keep only a running row maximum $m_i$, a running normaliser $\ell_i$ and an unnormalised output $\tilde O_i$:

$$m_i^{(j)} = \max\!\big(m_i^{(j-1)}, \mathrm{rowmax}(S_{ij})\big), \qquad \ell_i^{(j)} = e^{m_i^{(j-1)} - m_i^{(j)}}\,\ell_i^{(j-1)} + \mathrm{rowsum}\big(e^{S_{ij} - m_i^{(j)}}\big),$$
$$\tilde O_i^{(j)} = e^{m_i^{(j-1)} - m_i^{(j)}}\,\tilde O_i^{(j-1)} + e^{S_{ij} - m_i^{(j)}} V_j, \qquad O_i = \tilde O_i^{(T_c)} / \ell_i^{(T_c)} .$$

The result is **exact**. Rescaling by $e^{m^{\text{old}} - m^{\text{new}}}$ makes every partial sum consistent with the final maximum.

**Counting the IO.**
* Each query block $Q_i$ is read once and each output block $O_i$ written once: $2Nd$.
* For each of the $T_r$ query blocks, all of $K$ and $V$ are read once: $T_r \cdot 2Nd$.

$$\text{IO}_{\text{Flash}} = 2Nd + \frac{N}{B_r}\,2Nd .$$

With $B_r = \Theta(M/d)$ (the tile is as large as SRAM allows), this becomes Theorem 2 of Dao et al. (2022):

$$\text{IO}_{\text{Flash}} = \Theta\!\left(Nd + \frac{N^2 d^2}{M}\right).$$

(FlashAttention-1 puts the $K, V$ loop outside and re-reads $Q$ and $O$ instead. The count is the same, with $B_c$ in place of $B_r$. Our code uses the FlashAttention-2 order.)

**Comparison.** The quadratic term shrinks by the factor $\dfrac{4N^2}{2N^2 d / B_r} = \dfrac{2B_r}{d} = \Theta(M/d^2)$. With $B_r = M/(4d)$ and $M \approx 100$ K elements (≈200 KB of SRAM in FP16), that factor is about 12 for $d = 64$ and 3 for $d = 128$. Dao et al. also prove a lower bound: no exact algorithm can do asymptotically better for all $M$ in $[d, Nd]$.

### 1.4 What "O(N²) → O(N)" really means

The course briefing describes FlashAttention as reducing HBM traffic "from $O(N^2)$ to $O(N)$". Our derivation and our measurement give a more precise statement:
* **Extra memory** falls from $\Theta(N^2)$ (storing $S$ and $P$) to $\Theta(N)$ (the statistics $m, \ell$). This is the $O(N)$, and it is what lets long contexts fit at all.
* **HBM traffic** stays quadratic in $N$ for fixed $d$ and $M$, but is divided by $\Theta(M/d^2)$. It becomes linear in $N$ only if $M$ grows with $N$, which no real SRAM does.

**Measured** (`01_MultiHead_Attention_Core.py`, one head, $d = 64$, $B_r = B_c = 128$; the tiled result equals naive attention to < 1e-5):

| N | max \|tiled − naive\| | HBM elements, standard ($4Nd + 4N^2$) | HBM elements, tiled | Reduction |
| :-: | :-: | --: | --: | :-: |
| 512 | 1.2e-7 | 1,179,648 | 327,680 | 3.6× |
| 1,024 | 1.5e-7 | 4,456,448 | 1,179,648 | 3.8× |
| 2,048 | 1.8e-7 | 17,301,504 | 4,456,448 | 3.9× |
| 4,096 | 1.2e-7 | 68,157,440 | 17,301,504 | 3.9× |

The ratio stays at the predicted $2B_r/d = 4$ for every $N$. Both columns grow about 4× when $N$ doubles. The side quest asks for a ≥3× reduction in simulated memory traffic; our tiling achieves 3.6–3.9× with $B_r = 128$, and larger tiles ($B_r = 256$ on Hopper's 228 KB SRAM) would reach about 8×.

**FlashAttention-2 and -3.** FA-2 (Dao, 2023) reorders the loops so that each thread block owns a query block, which improves parallelism, and moves the rescaling out of the inner loop. FA-3 (Shah et al., 2024) overlaps memory and compute with Hopper's TMA and warp specialisation and uses FP8 blocks. Neither changes the IO bound above; both raise the share of peak FLOPs used, to about 75% on an H100 for FA-3.

## 2. KV-Cache Denial-of-Service Simulation

### 2.1 Why the KV-cache is the attack surface

During decoding, each generated token appends one K and one V vector per KV head per layer. A session therefore holds

$$\text{Bytes}_{\text{KV}} = 2 \times 2 \times L \times H_{KV} \times d_{\text{head}} \times N \times B \quad (\text{FP16}),$$

for its whole lifetime, and every new token *reads all of it* (decoding is memory-bound). Two properties make this exploitable:
* the memory is **linear in an attacker-controlled quantity**, the context length $N$;
* it is **held**, not transient, for as long as the session stays open.

For our LLaMA-3-8B-shaped reference model ($L = 32$, $H_Q = 32$, $d_{\text{head}} = 128$), a **single** 65,536-token session needs 32 GB of KV-cache with MHA and 8 GB with GQA.

### 2.2 Threat scenario: "Pleiades Flood"

> During the M45 transit, AURA serves bridge dialogue to all squadrons from one mainframe: an 80 GB accelerator holding the 8B-parameter model in FP16 (16 GB of weights), which leaves **64 GB for KV-caches**. A compromised maintenance client opens many concurrent sessions. Each one asks AURA to "repeat and expand the full sensor log" and is kept alive with short follow-up turns, so the context grows to the 65,536-token limit and is never released. The server admits each session (nothing is wrong with any single request), allocates its cache, and the allocator eventually fails with `torch.cuda.OutOfMemoryError: CUDA out of memory`. When that happens mid-batch, **every** session on the GPU fails, including the navigation officer's.

**Analytic result** (`kv_dos_analytic()`, 64 GB free, sessions of 65,536 tokens):

| Architecture | KV-cache per 65,536-token session | Sessions before OOM (64 GB free) |
| :-- | :-: | :-: |
| MHA (32 KV heads) | 32.00 GB | **2** |
| GQA 4:1 (8 KV heads) | 8.00 GB | **8** |
| MQA (1 KV head) | 1.00 GB | 64 |
| MLA (DeepSeek-V3 latent, 576 values/token/layer) | 2.25 GB | 28 |

**Live result on our GPU** (`kv_dos_live()`: real allocations of 2,048-token GQA sessions, 32 layers, until PyTorch raised the error; then freed):

```text
NVIDIA GeForce RTX 3070: 8.0 GB total, 6.88 GB free; 256 MB per session
sessions opened before OOM: 89 (predicted 27)
error raised: CUDA out of memory. Tried to allocate 256.00 MiB. GPU 0 has a total capacity of 8.00 GiB of which
0 bytes is free. Of the allocated memory 22.26 GiB is allocated by PyTorch ...
```

* The **first 27 sessions filled the card**, exactly as the formula predicts (6.88 GB / 256 MB).
* The next **62 sessions (15.5 GB) did not fail**: the Windows driver's *system-memory fallback* silently placed them in shared host RAM. The OOM came only at 22.3 GB.
* This is a **worse failure mode than a clean OOM**. The server keeps accepting sessions while every one of them runs at PCIe speed (about 25 GB/s) instead of GDDR6 speed (448 GB/s). Latency collapses for all users, and the alarm that an OOM would raise never fires.
* On Linux data-centre GPUs there is no such fallback, and the OOM at session 28 would abort the batch.
* Either way, **admission control must use the memory budget, not wait for the allocator to fail.**

**Reading.**
* With MHA, **two** malicious sessions are enough to take down an 80 GB mainframe.
* Even with GQA the margin is 8 sessions.
* The attack needs no exploit, only long contexts and patience. That is why the lecture's Security callout calls unconstrained multi-turn dialogue a *silent* allocation of gigabytes per session.
* Bandwidth is a second, softer limit. At 50 tokens/s, one 32,768-token MHA session needs 800 GB/s of cache reads (`03_GQA_and_KVCache_Benchmark.py` table), more than our GPU's entire memory bandwidth. Long sessions slow every co-scheduled user before memory runs out.

### 2.3 Failure modes observed

1. **Hard OOM.** The allocation fails; with naive serving the whole batch is aborted (loss of availability).
2. **Fragmentation OOM.** Contiguous per-session reservations leave gaps. Requests fail even though enough total memory is free (§3.1).
3. **Latency starvation.** Before OOM, memory-bound decoding slows every session in proportion to the total cached tokens it shares bandwidth with.

## 3. Mitigation Strategy

### 3.1 PagedAttention (vLLM)

PagedAttention (Kwon et al., 2023) stores each sequence's KV-cache in fixed-size **blocks** (e.g. 16 tokens), mapped through a per-sequence block table, like virtual-memory pages.
* **Internal fragmentation** falls to less than one block per sequence.
* **External fragmentation** disappears, because all blocks have the same size.
* Identical prefixes (system prompts) can **share** blocks copy-on-write.
* Memory is allocated *as tokens are generated*, not reserved for the maximum length.

**Simulation** (`paged_vs_contiguous()`: 2,000 requests with log-normal lengths, 64 GB budget, GQA, maximum context 32,768):

| Policy | Requests admitted into 64 GB | Reserved memory never used |
| :-- | :-: | :-: |
| Contiguous: reserve the 32,768-token maximum per request | 16 | **93.6%** |
| Paged: 16-token blocks allocated on demand | **288** | 0.4% |

Request lengths are log-normal: median 1,059 tokens, mean 1,762. Paging serves **18×** more concurrent requests from the same memory, because almost no request ever reaches the maximum it would otherwise reserve.

**Security value.**
* Paging does not stop a determined attacker from filling memory with *real* tokens, but it removes the 90%+ waste that lets ordinary traffic look like an attack.
* Because memory is accounted per block, per-tenant quotas, preemption and swapping become cheap. vLLM can evict a sequence's blocks to CPU memory and recompute or swap them back, so one session's growth degrades *that* session instead of crashing the batch.

### 3.2 Multi-Head Latent Attention (MLA, DeepSeek-V2/V3)

MLA (DeepSeek-AI, 2024a; used in DeepSeek-V3, 2024b) does not cache per-head keys and values.
* It caches one **low-rank latent** $c_t^{KV} = W^{DKV} h_t \in \mathbb{R}^{d_c}$ per token per layer ($d_c = 512$ in V3), plus a small shared RoPE key ($d_h^R = 64$).
* Keys and values are reconstructed by up-projections $W^{UK}, W^{UV}$. At inference these can be **absorbed** into the query and output projections, so the latent is never expanded.

**KV-cache per token, whole model, FP16** (`kv_bytes_per_token()`, applied to our 32-layer reference shape):

| Scheme | Values cached per token per layer | KV-cache per token, 32 layers | vs. MHA |
| :-- | :-: | :-: | :-: |
| MHA (32 KV heads × 128) | 2 × 4,096 = 8,192 | 512.0 KiB | 100% |
| GQA 4:1 (8 KV heads) | 2 × 1,024 = 2,048 | 128.0 KiB | 25.0% |
| MQA (1 KV head) | 2 × 128 = 256 | 16.0 KiB | 3.1% |
| **MLA** ($d_c = 512$ latent + 64 RoPE key) | 576 | 36.0 KiB | **7.0%** |

**Trade-offs.**
* MLA compresses below MQA while keeping per-head expressiveness. DeepSeek-V2 reports quality above MHA at 93.3% less KV-cache than its 67B dense predecessor.
* The cost is extra compute: up-projection or absorbed matrices, and a decoupled RoPE path, because RoPE does not commute with the absorbed projection.
* It requires *training* the model with MLA, unlike paging, which is a serving-side change.

### 3.3 Defence-in-depth recommendation

| Layer | Control | Stops |
| :-- | :-- | :-- |
| Model architecture | GQA now (4× less cache); MLA for the next model (≈14× less than MHA) | the per-session cost that makes the attack cheap |
| Serving engine | PagedAttention block allocation, prefix sharing, preemption/swap instead of abort | fragmentation OOM and the "one OOM kills the batch" failure |
| Admission control | per-session context cap (e.g. 16 K for untrusted clients); per-tenant KV quota in blocks; reject when projected KV > free memory | hard OOM from many long sessions |
| Session lifecycle | idle timeout, release of caches for closed or stalled sessions, summarise-and-truncate for very long dialogues | slow-drip attacks that keep caches alive |
| Monitoring | alarm on KV-blocks-per-tenant and on the growth rate of total cached tokens (the lecture's Security callout) | silent growth before failure |

## 4. Residual Risks

* **Quotas move the denial of service; they do not remove it.** An attacker with many legitimate accounts can still exhaust capacity. Capacity planning and rate limits are still needed.
* **Compression is not free.** Aggressive KV quantisation or eviction (e.g. keeping only recent tokens) saves memory but can silently drop facts from long dialogues. That is a safety problem of its own for a ship's log.
* **Our numbers are a model.** We used one model shape and FP16, and the latency measurements cover attention layers only. Real deployments add weights, activations and allocator overhead, so OOM comes *earlier* than our analytic counts.

---

## 5. References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS 2017. arXiv:1706.03762.
2. Milakov, M., & Gimelshein, N. (2018). "Online Normalizer Calculation for Softmax." arXiv:1805.02867.
3. Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022. arXiv:2205.14135.
4. Dao, T. (2023). "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." arXiv:2307.08691.
5. Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., & Dao, T. (2024). "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." arXiv:2407.08608.
6. Shazeer, N. (2019). "Fast Transformer Decoding: One Write-Head is All You Need." arXiv:1911.02150.
7. Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., & Sanghai, S. (2023). "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." EMNLP 2023. arXiv:2305.13245.
8. Kwon, W., Li, Z., Zhuang, S., et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023. arXiv:2309.06180.
9. DeepSeek-AI (2024a). "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model." arXiv:2405.04434.
10. DeepSeek-AI (2024b). "DeepSeek-V3 Technical Report." arXiv:2412.19437.
11. NVIDIA (2022). *NVIDIA H100 Tensor Core GPU Architecture* whitepaper.

---

**Station Certification:**  
*Authorized and submitted by Arun Dev Valavath, Security & Ethics Sentinel.*  
*Signed on bridge console: `[X] Security & Ethics Sentinel: FlashAttention IO derivation & KV-cache DoS audit verified`.*  
*USSE Noosphere // Sector SEC-03-M45 // Expedition Lovelace.*
