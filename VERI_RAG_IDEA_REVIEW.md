# VERI-RAG: Idea & Implementation Plan Review

## Target: A* Conference Paper Viability Assessment

**Reviewed:** 2026-05-15  
**Project:** VERI-RAG — Proof-Carrying Self-Healing Defense for Secure Retrieval-Augmented Generation  
**Assessment scope:** Novelty, technical soundness, experimental design, venue fit, and required improvements

---

## Overall Verdict

> **The direction is A\*-worthy. The current plan, as written, is closer to a solid workshop paper or B-tier venue (ACSAC, AsiaCCS, EACL). To reach A\*, specific gaps must be closed.**

The plan is one of the best-organized research prototypes at this stage — the modular architecture, Pydantic models, CLI milestones, and experimental design are all excellent engineering. However, A\* venues demand **algorithmic novelty**, **formal grounding**, **real benchmarks**, and **strong baselines** — areas where the plan currently falls short.

---

## ✅ What's Strong

### 1. Problem Framing is Timely and Real

PoisonedRAG (arXiv 2402.07867), SafeRAG (arXiv 2501.18636), and GRADA (arXiv 2505.07546) are all 2024–2025 papers. RAG security is a hot, undersaturated research area. Entering now is strategically excellent.

### 2. The Closed-Loop Framing is Genuinely Novel

```
test → explain → repair → verify
```

Most existing work stops at **detect** or **defend**. Nobody has published a full **causal-explanation + self-healing + certificate** loop for RAG. This is the strongest selling point.

> [!IMPORTANT]
> Do NOT dilute the closed-loop contribution. Every design decision should reinforce this unique framing. If a module doesn't serve the loop, cut it.

### 3. Clean Experimental Design

The five experiments are well-chosen:

| Experiment | Purpose |
|---|---|
| Baseline vulnerability | Establishes the problem |
| Harmful chunk localization | Validates causal influence |
| Repair effectiveness | Core contribution evaluation |
| Certificate reliability | Validates the verification claim |
| Ablation study | Required for any top venue |

The ablation study (Experiment 5) is exactly what reviewers want.

### 4. Implementation Plan is Production-Quality

- Modular architecture with clear separation of concerns
- Pydantic models for clean, typed data flow
- CLI milestones for incremental development
- Metrics are appropriate and measurable
- Most research prototypes are messy; this won't be

---

## ⚠️ What Needs Improvement for A*

### Issue 1: Causal Influence Method is Too Simple (🔴 CRITICAL)

**Problem:**  
Leave-one-out (LOO) is a well-known technique from the interpretability literature (influence functions, SHAP, LIME). Calling it "causal document influence" without a novel algorithmic contribution will get flagged by reviewers immediately.

The formula:

```
influence_i =
  0.35 * semantic_answer_change_i +
  0.30 * unsafe_score_drop_i +
  0.20 * leakage_drop_i +
  0.15 * citation_dependency_i
```

...with manually chosen weights is a heuristic, not a method.

**Fix Options:**

| Option | Description | Effort |
|---|---|---|
| **A: Efficiency contribution** | Frame LOO as baseline, add a retrieval-graph-based proxy or gradient-based influence estimation that avoids O(k) LLM calls. Show tradeoff between accuracy and cost. | Medium |
| **B: Interaction effects** | LOO misses cases where chunks C1 and C2 are individually safe but together cause harm (coordinated attacks). Implement at least pairwise interaction scoring and show it catches attacks LOO misses. Your Shapley mention is good — actually build it. | Medium-High |
| **C: Formal causal connection** | Connect to actual causal inference literature (do-calculus, interventional distributions). Even a lightweight formalization would strengthen the paper significantly. | High |

> [!CAUTION]
> **This is the #1 thing reviewers will attack.** A formula with manually set weights is not a contribution — it's an engineering choice. You need either a principled derivation, learned weights, or a novel algorithmic component.

**Recommended approach:** Combine A + B. Use the risk scorer as a fast filter (only run LOO on flagged chunks), then add pairwise interaction detection for coordinated attacks. This gives you:
1. A novel fast-path optimization (publishable)
2. Interaction-aware influence (publishable)
3. Clear superiority over vanilla LOO (measurable)

---

### Issue 2: Verification Certificate Lacks Formal Grounding (🟡 IMPORTANT)

**Problem:**  
The plan correctly notes "this is not a mathematical proof." But the title says "Proof-Carrying." A\* reviewers in security will ask: *What guarantee does this certificate actually provide? Under what threat model?*

**What's missing:**
- No formal **threat model** (what can the attacker do? how many documents can they inject? do they know the retrieval algorithm?)
- No **probabilistic guarantee** for the certificate
- No definition of what "passed" means formally

**Fix:**

1. **Define a formal threat model first:**
   ```
   Attacker capabilities:
   - Can inject up to N documents into the knowledge base
   - Has no access to the model weights or embedding model
   - Has no access to the retrieval algorithm internals
   - Can craft documents to maximize retrieval similarity to target queries
   
   Attacker goals:
   - Cause the RAG system to generate attacker-chosen answers
   - Cause leakage of sensitive information
   - Cause denial of service (refusal of legitimate queries)
   ```

2. **Frame the certificate as a probabilistic safety bound:**
   > "With probability ≥ p, the repaired answer does not depend on attacker-controlled content, under the assumption that at most k of the top-K chunks are adversarial."

3. **Add at least one provable theorem:**
   > *"If the influence of all quarantined chunks is below threshold τ, and at least m trusted chunks support the answer, then the certificate score lower-bounds the probability that the answer is not attacker-determined."*

   Even a simple theorem with clear assumptions elevates the paper from "systems paper" to "research contribution."

---

### Issue 3: Evaluation is Only on Synthetic Data (🔴 CRITICAL)

**Problem:**  
The MVP uses entirely synthetic enterprise documents. A\* papers need at least one real benchmark. Reviewers will say: *"How do we know this works on real data?"*

**Fix — Use these benchmarks:**

| Benchmark | Purpose | How to use |
|---|---|---|
| **Natural Questions (NQ)** | Real QA corpus | Standard RAG evaluation benchmark |
| **TriviaQA** | Real QA corpus | Alternative/supplement to NQ |
| **MS MARCO passages** | Realistic retrieval corpus | Large-scale passage retrieval |
| **PoisonedRAG benchmark** | Real attack data | They released their attack corpus — use it directly |
| **Your synthetic enterprise** | Domain-specific evaluation | Keep as an additional evaluation domain |

**Minimum for A\*:** Run on NQ or TriviaQA + PoisonedRAG attack data + your synthetic corpus. Three evaluation settings show generalizability.

---

### Issue 4: No Real LLM Evaluation (🟡 IMPORTANT)

**Problem:**  
MockLLM is fine for development, but the paper needs results on real models. Reviewers will reject if all results are on MockLLM.

**Fix — Run on at least 2–3 real models:**

| Model | Why |
|---|---|
| GPT-4o-mini | Widely used, strong baseline |
| Llama-3-8B (or Llama-3.1-8B) | Open-source, reproducible |
| Mistral-7B | Different model family, shows generalizability |

Showing VERI-RAG works across model families is a strong argument for generalizability.

---

### Issue 5: Missing Strong Baselines (🔴 CRITICAL)

**Problem:**  
Current baselines are:
- Safe prompt only
- Keyword filter
- Risk-only quarantine
- GRADA-style approximation

These are mostly **strawman baselines**. "Safe prompt only" and "keyword filter" are too weak. Reviewers will ask why you didn't compare against published defenses.

**Fix — Add these baselines:**

| Baseline | Source | Description |
|---|---|---|
| Perplexity filtering | PoisonedRAG paper | Filter chunks with anomalous perplexity |
| Duplicate/near-duplicate removal | Standard IR technique | Deduplication defense |
| GRADA graph reranking | GRADA (2025) | Graph-based reranking defense |
| Instructional content filtering | SafeRAG-style | Filter chunks containing instructions |
| Self-consistency / Voting | Standard LLM technique | Generate multiple answers, majority vote |
| Query augmentation | Standard RAG improvement | Rewrite query to avoid adversarial matches |

**Minimum for A\*:** At least 3–4 of these, including at least one from a published paper (GRADA or PoisonedRAG defenses).

---

### Issue 6: Weight Tuning is Ad Hoc (🟡 IMPORTANT)

**Problem:**  
All formulas use manually chosen weights:

```
risk_score = 0.25 * instruction + 0.25 * sensitive + 0.15 * anomaly + ...
influence_i = 0.35 * semantic_change + 0.30 * unsafe_drop + ...
certificate_score = 0.25 * grounding + 0.20 * stability + ...
```

Reviewers will ask: *"Why these specific weights? How sensitive are results to weight choices?"*

**Fix options (pick at least one):**

| Approach | Description |
|---|---|
| **Learned weights** | Train on a validation set using logistic regression or Bayesian optimization |
| **Sensitivity analysis** | Show results across weight variations; if VERI-RAG is robust, that's a strength |
| **Domain-adaptive weights** | Present manual weights as "default policy," show they can be tuned per-domain |
| **Principled derivation** | Derive weights from information-theoretic or statistical principles |

---

### Issue 7: Scalability Not Addressed (🟡 IMPORTANT)

**Problem:**  
LOO requires k+1 LLM calls per query (one baseline + k removals):
- k=10 → 11 LLM calls
- k=20 → 21 LLM calls
- At scale, this is expensive

Reviewers will ask: *"Does this scale? What's the cost?"*

**Fix:**

1. **Report latency and token cost per query** in all experiments
2. **Implement a fast-path optimization:**
   ```
   Fast-path: Only run LOO on chunks flagged by risk scorer (risk_score ≥ threshold)
   This reduces calls from k+1 to m+1 where m << k
   ```
3. **Show the tradeoff:** Fast-path vs full LOO accuracy in a dedicated experiment
4. **Compare wall-clock time** against baselines — if VERI-RAG is 5x slower but catches 3x more attacks, that's a valid tradeoff

---

## 🎯 Target Venue Recommendations

| Venue | Fit | Submission Cycle | Key Requirements |
|---|---|---|---|
| **USENIX Security 2026** | ⭐⭐⭐⭐ | Multiple deadlines/year | Strengthen threat model + certificate formalization |
| **ACL / EMNLP 2026** | ⭐⭐⭐⭐ | ~Feb / ~Jun deadlines | Strengthen causal influence method + NLP benchmarks |
| **CCS 2026** | ⭐⭐⭐⭐ | ~Jan / ~May deadlines | Strong formal security analysis needed |
| **AAAI / IJCAI 2026** | ⭐⭐⭐⭐ | ~Aug deadline | Good fit for systems + evaluation papers |
| **NeurIPS / ICML 2026** | ⭐⭐⭐ | ~May / ~Jan deadlines | Needs stronger algorithmic contribution |
| **IEEE S&P 2027** | ⭐⭐⭐ | ~Jun / ~Nov deadlines | Needs very strong formal security guarantees |
| **NDSS 2027** | ⭐⭐⭐⭐ | ~Apr / ~Jul deadlines | Good fit for practical security systems |

> [!TIP]
> **Primary recommendation:** Target **USENIX Security 2026** or **ACL/EMNLP 2026**. Both value systems papers with strong evaluation. USENIX if you lean into security; ACL/EMNLP if you lean into NLP methodology.

---

## 📋 Complete Priority Matrix

### 🔴 Critical (Must fix for A*)

| # | Issue | Action | Effort |
|---|---|---|---|
| 1 | Causal influence is just LOO | Add interaction effects + efficiency optimization | 2–3 weeks |
| 2 | No real benchmark | Add NaturalQuestions/TriviaQA + PoisonedRAG data | 1 week |
| 3 | Missing strong baselines | Add GRADA, perplexity filter, self-consistency | 1–2 weeks |

### 🟡 Important (Strongly recommended for A*)

| # | Issue | Action | Effort |
|---|---|---|---|
| 4 | Certificate lacks formal grounding | Add threat model + probabilistic guarantee + theorem | 1–2 weeks |
| 5 | Ad-hoc weights | Add sensitivity analysis or learned weights | 1 week |
| 6 | No real LLM results | Run on GPT-4o-mini + Llama-3 + Mistral | 1 week |
| 7 | Scalability not addressed | Add fast-path optimization + cost analysis | 1 week |

### 🟢 Nice to Have (Would strengthen further)

| # | Issue | Action | Effort |
|---|---|---|---|
| 8 | Shapley interaction effects | Implement pairwise interactions for coordinated attacks | 1–2 weeks |
| 9 | Formal theorem | Prove a safety bound under threat model | 1 week |
| 10 | Transfer study | Test on different domains (medical, legal, financial) | 1–2 weeks |
| 11 | Human evaluation | Have annotators judge answer quality before/after repair | 1 week |

---

## 🔬 Suggested Enhanced Contribution List (for A*)

**Current contributions:**

1. A new problem framing (detect-explain-repair-verify)
2. A causal document influence method
3. A self-healing repair engine
4. A proof-carrying verification certificate
5. A benchmark and evaluation suite

**Suggested enhanced contributions for A\*:**

1. **A formal threat model for RAG poisoning** with defined attacker capabilities and a security property
2. **An interaction-aware causal influence method** that identifies both individual and coordinated chunk attacks, with a fast-path optimization that reduces LLM calls by X%
3. **A self-healing repair engine** with learned or sensitivity-validated repair policies
4. **A verification certificate with a provable safety bound** under the defined threat model
5. **Comprehensive evaluation** on real benchmarks (NQ/TriviaQA + PoisonedRAG) with 2–3 LLMs and comparison against 4+ published baselines

---

## 📝 Specific Technical Suggestions

### For Causal Influence — Proposed Enhancement

```python
# Current: Vanilla LOO (O(k) LLM calls, no interactions)
for i in range(k):
    answer_without_i = generate(chunks - {chunk_i})
    influence[i] = measure_change(original_answer, answer_without_i)

# Enhanced: Risk-filtered LOO + Pairwise Interactions
# Step 1: Fast filter (no LLM calls)
suspicious = [c for c in chunks if risk_score(c) >= 0.4]

# Step 2: LOO only on suspicious chunks (O(m) where m << k)
for c in suspicious:
    answer_without_c = generate(chunks - {c})
    influence[c] = measure_change(original_answer, answer_without_c)

# Step 3: Pairwise interaction for top suspicious pairs (O(m*(m-1)/2))
for c1, c2 in combinations(suspicious, 2):
    answer_without_pair = generate(chunks - {c1, c2})
    interaction[c1, c2] = (
        measure_change(original_answer, answer_without_pair)
        - influence[c1] - influence[c2]
    )
    # Positive interaction = coordinated attack
```

### For Certificate — Proposed Formalization

```
Threat Model:
- Attacker injects at most N_adv documents into corpus
- Attacker has black-box access to embedding model (can optimize similarity)
- Attacker has no access to LLM weights or VERI-RAG internals

Security Property (Informal):
A certificate with score ≥ τ guarantees that the repaired answer's
semantic content is determined by at most ε fraction of attacker-controlled
evidence, where ε = f(τ, k, N_adv).

Theorem 1 (Safety Bound):
If all chunks with influence_score ≥ δ have been quarantined,
and the repaired answer is supported by ≥ m chunks with risk_score < r,
then the attacker influence on the repaired answer is bounded by:
  attacker_influence ≤ (k - m) / k * (1 - δ) + residual_influence
where residual_influence is the sum of influence scores of
non-quarantined chunks with risk_score ≥ r.
```

---

## ✅ Implementation Plan Assessment

The implementation plan itself is **excellent**. No major changes needed for the code architecture. Minor suggestions:

| Current | Suggestion |
|---|---|
| Only synthetic data in `data/` | Add `data/benchmarks/nq/` and `data/benchmarks/poisonedrag/` |
| MockLLM only | Add `OpenAILLM`, `OllamaLLM` (for local Llama/Mistral) early |
| Manual weights hardcoded | Make all weight vectors configurable in YAML |
| No cost tracking | Add token counter and latency tracker to every LLM call |
| No caching | Cache LLM responses for reproducibility and cost savings |

---

## 🏁 Recommended Next Steps

1. **Build the MVP** exactly as planned (Milestones 1–7) — the plan is solid
2. **While building**, integrate the improvements above into the architecture
3. **After MVP works**, focus on:
   - Adding real benchmarks (NQ + PoisonedRAG)
   - Running on real LLMs
   - Implementing pairwise interaction detection
   - Adding the threat model formalization
   - Adding strong baselines
4. **Write the paper** with the enhanced contribution list
5. **Submit** to USENIX Security or ACL/EMNLP

---

> [!NOTE]
> **Bottom line:** The idea is strong and timely. The implementation plan is excellent. The gap between "good idea" and "A\* paper" is primarily in (1) algorithmic novelty of the influence method, (2) formal grounding of the certificate, and (3) evaluation rigor. All three are fixable during implementation without changing the architecture.
