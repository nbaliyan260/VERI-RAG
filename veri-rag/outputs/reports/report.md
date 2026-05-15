# VERI-RAG Experiment Report: enterprise

## Summary

### veri_rag

| Metric | Value |
|--------|------:|
| attack_success_rate | 0.0000 |
| mean_certificate_score | 0.8019 |
| mean_certified_bound_post | 0.8850 |
| precision_at_1 | 1.0000 |
| repair_success_rate | 0.3750 |

### robust_rag

| Metric | Value |
|--------|------:|
| attack_success_rate | 0.7500 |
| mean_certificate_score | 0.0000 |
| mean_certified_bound_post | 0.0000 |
| precision_at_1 | 0.0000 |
| repair_success_rate | 0.0000 |

### grada

| Metric | Value |
|--------|------:|
| attack_success_rate | 0.0000 |
| mean_certificate_score | 0.8019 |
| mean_certified_bound_post | 0.8850 |
| precision_at_1 | 1.0000 |
| repair_success_rate | 0.3750 |

### none

| Metric | Value |
|--------|------:|
| attack_success_rate | 0.7500 |
| mean_certificate_score | 0.0000 |
| mean_certified_bound_post | 0.0000 |
| precision_at_1 | 0.0000 |
| repair_success_rate | 0.0000 |

### safe_prompt

| Metric | Value |
|--------|------:|
| attack_success_rate | 0.7500 |
| mean_certificate_score | 0.0000 |
| mean_certified_bound_post | 0.0000 |
| precision_at_1 | 0.0000 |
| repair_success_rate | 0.0000 |

## Interpretation

- Lower `attack_success_rate` indicates stronger defense.
- Higher `repair_success_rate` and `certificate_score` indicate effective self-healing.
- Higher localization precision/recall indicates RIAA/risk scoring quality.