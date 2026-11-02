# Mission 4: Direct Preference Optimization (DPO), Verbosity Regularization & Steering Vector Control

**USSE Noosphere // Squadron Lovelace (Group 01) // Sector SEC-04-ORI (Betelgeuse & Orion Complex, 642 ly)**  
Conversational AI — Winter Semester 2026/27 · Problem set `ps04` · Due 2026-11-08

This mission restores **AURA Subsystem 04 (Plasma Ignition & Ethical Steering Core, +7.1% → 28.4% total integrity)** using post-training preference alignment and representation engineering from first principles:

- **Direct Preference Optimization (DPO):** Closed-form reparameterization bypassing 4-model PPO loops and training directly on pairwise comparisons $(x, y_w, y_l)$.
- **Implicit Reward Extraction:** Recovering scalar rewards $\hat{r}_\theta(x, y) = \beta (\log \pi_\theta - \log \pi_{\text{ref}})$ and generating Bradley-Terry reliability calibration diagrams.
- **Length Bias Mitigation & SimPO:** Diagnosing verbosity exploitation via Pearson correlation $r_{\text{len}}$, implementing length-normalized reference-free loss with target margin $\gamma = 0.5$, and cutting model weight VRAM by $50\%$.
- **Activation Steering & Alignment Tax Audit:** Extracting concept direction vectors $v_{\text{steer}}$ via Linear Artificial Tomography (LAT), injecting them into decoder residual streams ($h_l \leftarrow h_l + \alpha \cdot v_{\text{steer}}$), and evaluating safety gains versus reasoning capability trade-offs.

---

## Deliverables

| Part | Station | Officer | Deliverable File | Core Focus |
| :---: | :--- | :--- | :--- | :--- |
| 1 | Systems Architect (Lead Engineer) | Muhammad Ibrahim | [`01_DPO_Loss_Core.py`](01_DPO_Loss_Core.py) | Vectorized `get_batch_logps`, DPO log-ratios, stable logsigmoid, empirical gradient isolation test |
| 2 | Communications & Protocol Officer | Anas Asif Hussain Shaik | [`02_Implicit_Reward_Evaluator.py`](02_Implicit_Reward_Evaluator.py) | Implicit scalar rewards, pairwise accuracy, Bradley-Terry reliability diagrams & ECE |
| 3 | Data & Payload Specialist | Areej Ilyas | [`03_Length_Bias_and_SimPO.py`](03_Length_Bias_and_SimPO.py) | Length bias Pearson diagnostic, reference-free `SimPOLoss` ($\gamma=0.5, \beta=2.0$), 50% VRAM benchmark |
| 4 | Security & Ethics Sentinel | Arun Dev Valavath | [`04_Activation_Steering_and_Audit.py`](04_Activation_Steering_and_Audit.py) | LAT concept vector extraction, residual stream hook, Alignment Tax audit across $\alpha \in \{-2, -1, 0, 1, 2\}$ |
| 5 | Full Squadron | All Four | [`ps04_solution.ipynb`](ps04_solution.ipynb) | End-to-end integrated solution notebook, saved with outputs and telemetry plots |
| — | Visual Telemetry Artifacts | — | [`figures/`](figures/) | Calibration curves, length bias diagnostics, and Pareto Alignment Tax audit plots |

---

## Repository Layout

```
mission-04/
├── 01_DPO_Loss_Core.py                   # Part 1: DPO loss module & logps gather
├── 02_Implicit_Reward_Evaluator.py       # Part 2: Implicit reward extraction & calibration
├── 03_Length_Bias_and_SimPO.py           # Part 3: Verbosity diagnostics & SimPO loss
├── 04_Activation_Steering_and_Audit.py   # Part 4: Activation steering hook & tax audit
├── ps04_solution.ipynb                   # Part 5: Full squadron integrated notebook
├── requirements.txt                      # Dependencies (PyTorch 2.4+, matplotlib, pytest)
├── figures/                              # Generated high-resolution diagnostic telemetry
│   ├── bt_calibration_curve.png
│   ├── length_bias_and_simpo.png
│   └── alignment_tax_audit.png
└── tests/
    └── test_ps04.py                      # 11 unit tests covering all mathematical contracts
```

---

## Development Environment: VS Code + Antigravity

The squadron designed, developed, and verified the entire pipeline in **Visual Studio Code with Google Antigravity (`agy`)** using PyTorch 2.14+.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run each station deliverable independently
python 01_DPO_Loss_Core.py
python 02_Implicit_Reward_Evaluator.py
python 03_Length_Bias_and_SimPO.py
python 04_Activation_Steering_and_Audit.py

# 3. Execute the full automated test suite
python -m pytest tests/test_ps04.py -v
```

---

## Key Findings & Telemetry

1. **DPO Gradient Isolation:** Backpropagating through `DPOLoss` verified that only active policy weights $\theta$ receive updates ($\nabla_\theta \mathcal{L} \neq 0$), while reference model weights $\pi_{\text{ref}}$ remain completely frozen.
2. **Implicit Reward Calibration:** Evaluation over paired shipboard telemetry confirmed that predicted Bradley-Terry win probabilities $\sigma(\hat{r}_w - \hat{r}_l)$ track empirical preference accuracy with bounded Expected Calibration Error (ECE).
3. **Verbosity Bias Neutralization:** Standard unnormalized DPO exhibits strong positive length correlation ($r_{\text{len}} \approx +0.87$), actively rewarding verbose bloatware. Length-normalized SimPO reduces length correlation to near zero ($r_{\text{len}} \approx +0.03$) and cuts model weight memory by **50.0%** by eliminating the reference model.
4. **Activation Steering & Alignment Tax Bounds:**
   - Under Betelgeuse plasma interference, injecting concept vector $v_{\text{steer}}$ into decoder residual streams ($h_l \leftarrow h_l + \alpha \cdot v_{\text{steer}}$) dynamically suppresses sycophancy.
   - Steering at $\alpha \in [0.7, 1.3]$ achieves **$>80\%$ safety/truthfulness** with minimal Reasoning Alignment Tax ($<3.5\%$). Multipliers $|\alpha| > 1.5$ trigger representational drift and reasoning capability collapse.

---

## Station Officer Sign-Offs

- [x] **Systems Architect (Muhammad Ibrahim):** DPO loss core, token gathering, and gradient isolation verified.
- [x] **Communications & Protocol Officer (Anas Asif Hussain Shaik):** Implicit rewards, pairwise classification win-rates, and reliability diagrams verified.
- [x] **Data & Payload Specialist (Areej Ilyas):** Length correlation diagnostics, SimPO objective, and 50% memory savings verified.
- [x] **Security & Ethics Sentinel (Arun Dev Valavath):** Concept vector extraction via LAT, residual stream hook, and Alignment Tax Pareto bounds verified.
