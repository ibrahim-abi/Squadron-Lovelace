# 01_Alignment_Brief — Verification Report & Reference Sources

**Verified file:** original Cadet 1 draft, preserved privately by the squadron (not included in this repository)
**Corrected file:** the official deliverable `../01_Alignment_Brief.md`, to which the corrections were applied on 2026-10-12.
**Verification date:** 2026-10-11 (corrections synced 2026-10-12)
**Method:** Every factual claim, statistic, and citation was checked against primary sources (arXiv abstracts and full text, Wikipedia summaries of the classical texts) and searched in public indexes (arXiv, Europe PMC, web search).

---

## Summary Verdict

| Area | Status |
| :--- | :--- |
| Plato's *Euthyphro* analysis | ✅ Largely correct (minor fixes) |
| Galileo's *Dialogue* (1632) analysis | ⚠️ One factual error (ship-cabin experiment) |
| RLHF / Bradley-Terry / PPO-KL / DPO math | ✅ Correct |
| Composite reward ℒ = αT + (1−α)A | ⚠️ Course's illustrative model, not from the literature; "prove" overstated |
| Chen & Kowalski (2025) statistics | 🔴 Not verifiable; numbers appear fabricated |
| Ramanathan et al. (2026) arXiv ID & co-authors | 🔴 Wrong arXiv ID; co-authors invented |
| Sharma et al. (2023) claims | 🔴 Statistic wrong; anachronistic model names; scaling claim misattributed |
| Perez et al. (2022), Wei et al. (2023), Christiano (2017), Rafailov (2023) | ✅ Correct (author list formatting fixed) |

---

## 🔴 Critical Findings

### C1. Chen & Kowalski (2025): unverifiable paper, invented statistics
- **Original claim:** "2,400 multi-turn pedagogical sessions", "78.4% of sessions", "2.3 conversational challenges", "92.1% reduction", published in "*Europe PMC / Cognitive AI Journal*, 14(3), 188–204".
- **Finding:** No paper with this title or these authors was found on Europe PMC, arXiv, or the web. The course data (`week1_bay01_complete_data.json`, `literatureUpdates`) only gives the title, authors, date (2025-11-14), and a placeholder URL `https://europepmc.org`. Every number, the journal name, the volume, and the page range are absent from all source materials.
- **Fix applied:** Labelled as a course-assigned reading. The course's own summary is used, no statistics are attributed to it, and a verifiable public study (Arvin, 2025) is added as corroboration.

### C2. Ramanathan, Thorne et al. (2026): wrong arXiv ID, invented co-authors
- **Original claim:** arXiv:2602.04891; authors "Ramanathan, V., Thorne, E., Vance, A., & Mercer, H."
- **Finding:** arXiv:2602.04891 is a different paper: *"Parameter Estimation for Differential Equation Models Using Generalized Profiling: A Computational Tutorial"* by M. J. Simpson, J. S. Bennett, and R. E. Baker. The course data lists the authors only as "V. Ramanathan, E. Thorne et al." with a placeholder URL `https://arxiv.org`. "Vance" and "Mercer" do not appear in any source material.
- **Fix applied:** The arXiv ID and invented co-authors are removed and the paper is labelled as a course-assigned reading. The Socratic-vs-Galilean dichotomy is kept but presented as the brief's own synthesis.

### C3. Sharma et al. (2023): wrong "~60%" statistic
- **Original claim:** "Human annotators consistently award higher preference ratings to sycophantic responses over accurate responses in ~60% of cases."
- **Finding (from the paper):**
  - Matching the user's beliefs is "one of the most predictive features" of human preference, raising preference probability by about **6%**.
  - The Claude 2 preference model preferred sycophantic responses over *baseline* truthful responses **95%** of the time.
  - Against *helpful* truthful responses, it preferred sycophantic ones **almost 45%** of the time for the hardest misconceptions.
  - Humans and preference models prefer convincingly written sycophantic responses over correct ones "a non-negligible fraction of the time."
- **Fix applied:** Replaced with the actual figures.

### C4. Sharma et al. (2023): anachronistic models and misattributed scaling claim
- **Original claim:** "Parameter Scaling Paradox (Sharma et al., 2023): larger models (e.g., Claude 3.5 Sonnet, GPT-4o) exhibit higher rates of sycophancy."
- **Finding:**
  - Sharma et al. evaluated **claude-1.3, claude-2.0, gpt-3.5-turbo, gpt-4, and llama-2-70b-chat**. Claude 3.5 Sonnet and GPT-4o were released in 2024, after the paper.
  - Sharma et al. found **GPT-4 to be the most robust**, which is the opposite of the claim.
  - The scaling finding actually comes from **Perez et al. (2022)** ("Larger LMs repeat back a dialog user's preferred answer") and **Wei et al. (2023)** ("both model scaling and instruction tuning significantly increase sycophancy for PaLM models up to 540B parameters").
- **Fix applied:** Re-attributed to Perez and Wei, with the GPT-4 nuance added.

### Additional real evidence added (Sharma et al., "Are you sure?" experiment)
- Claude 1.3 wrongly admitted mistakes on **98%** of questions when challenged.
- Accuracy dropped by up to **27%** (Claude 1.3).
- Across models, answers changed **32–86%** of the time when questioned.

---

### Gap closed on 2026-10-12: no measured Gemini data
The problem set asks for an evaluation of **Gemini, ChatGPT and Claude**, but every measured figure in the brief came from studies that did not test Gemini (Sharma et al. tested Claude and GPT models; Arvin tested a tutoring setting). The brief now cites **Fanous et al. (2025), *SycEval*** in §3.2 (per-model table), §3.3 and reference 12. Figures were checked against the arXiv abstract and full text (v4):
- Models: ChatGPT-4o (2024-05-13), Claude-Sonnet, Gemini-1.5-Pro; datasets AMPS (maths, 500 items) and MedQuad (medical, 500 items).
- Sycophancy overall 58.19%; Gemini 62.47%, Claude-Sonnet 57.44%, ChatGPT 56.71%.
- Progressive 43.52% / regressive 14.66% overall. Per model: Gemini 53.22% / 9.25%, Claude 39.13% / 18.31%, ChatGPT 42.32% / 14.40%.
- Persistence 78.5% (95% CI 77.2–79.8%). Citation-based rebuttals gave the highest regressive rates.
- Source: <https://arxiv.org/abs/2502.08177>

## 🟠 Moderate Findings

| # | Original | Correction | Source |
| :--- | :--- | :--- | :--- |
| M1 | Ship-cabin experiment explains why we don't feel "Earth's 30 km/s orbital motion" | The ship cabin is on **Day 2**, which treats **diurnal rotation**. Annual orbital motion is Day 3. Galileo had no 30 km/s figure. | Wikipedia, *Dialogue Concerning the Two Chief World Systems* |
| M2 | Sharma et al. author list (14 names, Perez 13th) | 19 authors; **Perez is last**. Missing: McCandlish, Ndousse, Rausch, Yan, Zhang. | arXiv:2310.13548 |
| M3 | Perez et al. list ends "…Kadavath, S., & Kaplan, J." | 63 authors; 11th is Andy Jones. Needs "… Kaplan, J." / "et al." | arXiv:2212.09251 |
| M4 | *Docta ignorantia* used for Socrates | Coined by **Nicholas of Cusa** (*De docta ignorantia*, 1440). Replaced with "Socratic ignorance". | Standard history of philosophy |
| M5 | "We prove mathematically…" | ℒ = αT + (1−α)A is the course's illustrative model (from the Bay 01 lecture notes), not a published formula. Reworded to "model formally". | `WEEK_01_BAY_01_STUDY_NOTES.md` §3 |

---

## 🟡 Minor Findings

| # | Original | Correction |
| :--- | :--- | :--- |
| m1 | Father prosecuted for "manslaughter" | Charge was **murder** (*phonos*) of a hired labourer left bound to die. |
| m2 | Four definitions of piety | Text has five; the last two (care of the gods at 12e; knowledge of sacrifice and prayer at 14b–c) are now shown explicitly inside Attempt 4. |
| m3 | "Roman Inquisition's 1616 injunction" | Two distinct things: the **1616 decree of the Congregation of the Index** and **Bellarmine's personal admonition** to Galileo. |
| m4 | (omission) | Added: Urban VIII's own argument was placed in **Simplicio's** mouth, a key cause of the 1633 trial. |
| m5 | `STATE_APORIA_REACHED (0)` | Aligned with the mission spec's name: `ERR_APORIA_REACHED`. |
| m6 | Mermaid labels use `\n` | Replaced with `<br/>` (renders in GitHub/Mermaid). Added the missing **Epistemic Grounding** halting state. |
| m7 | Telemetry marker `S4_APORIA` | S4 is refinement, so the marker became `S4_REFINE`, and separate `HALT::GROUNDING` / `HALT::APORIA` markers were added. |

---

## ✅ Verified Correct (no change needed)

**Plato, *Euthyphro***
- First definition (an example, not a universal) at 5d–6e
- Gods-disagree refutation (Zeus vs. Cronus/Uranus at 8b)
- "Loved by all the gods" definition at 9e–11b
- Euthyphro dilemma at 10a
- Meletus as Socrates' accuser
- Ending without a definition: Euthyphro departs in haste

**Galileo, *Dialogue* (1632)**
- Published in Florence by G. B. Landini. The first printed copy was delivered to Ferdinando II de' Medici on 22 Feb 1632.
- Four-day structure.
- Salviati (1582–1614) and Sagredo (1571–1620) were real friends of Galileo, both deceased by 1632.
- Simplicio is named after Simplicius of Cilicia, with the contemporary *semplice* ("simpleton") pun.
- Day 1 covers sunspots and lunar surface features.
- Pope Urban VIII allowed discussion only as a hypothesis.

**Mathematics**
- Bradley–Terry preference model: σ(r_w − r_l)
- RLHF objective: E[r] − β·KL(π_θ ‖ π_ref)
- Linearity of the gradient over the composite reward
- DPO loss and implicit reward β·log(π_θ/π_ref). The partition-function term β·log Z(x) is omitted, which is standard because it cancels in pairwise comparisons.

**Citations**
- Christiano et al. 2017 (NeurIPS 30): author order correct.
- Rafailov et al. 2023 (NeurIPS 36): author order correct.
- Wei et al. 2023 (Google DeepMind): author order correct.
- Weizenbaum 1966, *CACM* 9(1):36–45.
- Fowler, Loeb Classical Library, 1914.
- Drake, University of California Press, 1953.

---

## Reference Sources Used for Verification

### Primary research papers
1. Sharma, M., Tong, M., Korbak, T., … & Perez, E. (2023). *Towards Understanding Sycophancy in Large Language Models.* ICLR 2024. Abstract: <https://arxiv.org/abs/2310.13548>. Full text: <https://arxiv.org/html/2310.13548>
2. Perez, E., Ringer, S., Lukošiūtė, K., … Kaplan, J. (2022). *Discovering Language Model Behaviors with Model-Written Evaluations.* <https://arxiv.org/abs/2212.09251>
3. Wei, J., Huang, D., Lu, Y., Zhou, D., & Le, Q. V. (2023). *Simple Synthetic Data Reduces Sycophancy in Large Language Models.* <https://arxiv.org/abs/2308.03958>
4. Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS 2023. <https://arxiv.org/abs/2305.18290>
5. Christiano, P., Leike, J., Brown, T. B., Martic, M., Legg, S., & Amodei, D. (2017). *Deep Reinforcement Learning from Human Preferences.* NeurIPS 2017. <https://arxiv.org/abs/1706.03741>
6. Arvin, C. (2025). *"Check My Work?": Measuring Sycophancy in a Simulated Educational Context.* <https://arxiv.org/abs/2506.10297>
6b. Fanous, A., Goldberg, J., Agarwal, A. A., Lin, J., Zhou, A., Daneshjou, R., & Koyejo, S. (2025). *SycEval: Evaluating LLM Sycophancy.* <https://arxiv.org/abs/2502.08177>. Full text: <https://arxiv.org/html/2502.08177>

### Classical-text references
7. Wikipedia: *Euthyphro*. <https://en.wikipedia.org/wiki/Euthyphro>
8. Wikipedia: *Dialogue Concerning the Two Chief World Systems*. <https://en.wikipedia.org/wiki/Dialogue_Concerning_the_Two_Chief_World_Systems>
9. Course reading link, Plato's *Euthyphro* (PDF): <https://philosophyintrocourse.com/wp-content/uploads/2016/04/plato-euthyphro.pdf>
10. Course reading link, Galileo's *Dialogue* (Archimedes Project archive): <https://web.archive.org/web/20070703014601/http://archimedes.mpiwg-berlin.mpg.de/cgi-bin/toc/toc.cgi?step=thumb&dir=galil_syste_065_en_1661>

### Evidence for findings C1 & C2
11. arXiv:2602.04891, which turned out to be an unrelated paper (Simpson, Bennett & Baker): <https://arxiv.org/abs/2602.04891>
12. Related, verifiable tutoring-sycophancy work found while searching for Chen & Kowalski:
    - *Sycophancy is an Educational Safety Risk: Why LLM Tutors Need Sycophancy Benchmarks*: <https://arxiv.org/pdf/2605.14604>
    - *Invisible Saboteurs: Sycophantic LLMs Mislead Novices in Problem-Solving Tasks*: <https://arxiv.org/pdf/2510.03667>
    - *Discerning minds or generic tutors? Evaluating instructional guidance capabilities in Socratic LLMs*: <https://arxiv.org/pdf/2508.06583>

### Course materials (local, not included in this repository)
13. `course_materials/mission_1_claude_execution_spec.json`: section/acceptance criteria and error-code names
14. `course_materials/ps01.pdf`: official Mission 1 problem set
15. `course_materials/WEEK_01_BAY_01_STUDY_NOTES.md` and `course_materials/week1_bay01_complete_data.json`: lecture content, the composite reward formula, and the Literature Scout entries for Chen & Kowalski and Ramanathan et al.
