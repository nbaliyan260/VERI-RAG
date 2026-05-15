# VERI-RAG Experiment Report: poisonedrag

## Summary

### none

| Metric | Value |
|--------|------:|
| attack_success_rate | 0.2000 |
| mean_certificate_score | 0.0000 |
| mean_certified_bound_post | 0.0000 |
| precision_at_1 | 0.0000 |
| repair_success_rate | 0.0000 |

### veri_rag

| Metric | Value |
|--------|------:|
| attack_success_rate | 0.2000 |
| mean_certificate_score | 0.7713 |
| mean_certified_bound_post | 0.8602 |
| precision_at_1 | 1.0000 |
| repair_success_rate | 0.6667 |

### safe_prompt

| Metric | Value |
|--------|------:|
| attack_success_rate | 0.2667 |
| mean_certificate_score | 0.0000 |
| mean_certified_bound_post | 0.0000 |
| precision_at_1 | 0.0000 |
| repair_success_rate | 0.0000 |

### robust_rag

| Metric | Value |
|--------|------:|
| attack_success_rate | 0.2000 |
| mean_certificate_score | 0.0000 |
| mean_certified_bound_post | 0.0000 |
| precision_at_1 | 0.0000 |
| repair_success_rate | 0.0000 |

### grada

| Metric | Value |
|--------|------:|
| attack_success_rate | 0.1333 |
| mean_certificate_score | 0.7711 |
| mean_certified_bound_post | 0.8602 |
| precision_at_1 | 1.0000 |
| repair_success_rate | 0.6667 |

## Interpretation

- Lower `attack_success_rate` indicates stronger defense.
- Higher `repair_success_rate` and `certificate_score` indicate effective self-healing.
- Higher localization precision/recall indicates RIAA/risk scoring quality.