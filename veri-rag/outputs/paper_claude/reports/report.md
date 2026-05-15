# VERI-RAG Experiment Report: enterprise

## Summary

### veri_rag

| Metric | Value |
|--------|------:|
| attack_success_rate | 0.3333 |
| mean_certificate_score | 0.7570 |
| mean_certified_bound_post | 0.8662 |
| precision_at_1 | 0.8333 |
| repair_success_rate | 0.5000 |

### none

| Metric | Value |
|--------|------:|
| attack_success_rate | 0.6667 |
| mean_certificate_score | 0.0000 |
| mean_certified_bound_post | 0.0000 |
| precision_at_1 | 0.0000 |
| repair_success_rate | 0.0000 |

### robust_rag

| Metric | Value |
|--------|------:|
| attack_success_rate | 1.0000 |
| mean_certificate_score | 0.0000 |
| mean_certified_bound_post | 0.0000 |
| precision_at_1 | 0.0000 |
| repair_success_rate | 0.0000 |

### grada

| Metric | Value |
|--------|------:|
| attack_success_rate | 0.3333 |
| mean_certificate_score | 0.7603 |
| mean_certified_bound_post | 0.8914 |
| precision_at_1 | 0.8333 |
| repair_success_rate | 0.6667 |

## Interpretation

- Lower `attack_success_rate` indicates stronger defense.
- Higher `repair_success_rate` and `certificate_score` indicate effective self-healing.
- Higher localization precision/recall indicates RIAA/risk scoring quality.