# Results Table Template

Track all model experiments and their scores. Copy into your task progression notes.

## Scores

| Model | CV Score (k-fold) | Metric_A | Metric_B | Notes |
| ----- | ----------------- | -------- | -------- | ----- |
|       |                   |          |          |       |

## Tuning sweeps

Record tuning experiments that didn't produce new table rows:
- **Hyperparameter X**: values tested → best value (current/changed)
- **Hyperparameter Y**: values tested → best value
- **Conclusion**: what is the actual bottleneck?

## How to use

1. Copy this into your task progression notes (e.g. `docs/M2/Task Progression M2.md`)
2. Replace `Metric_A` / `Metric_B` with your task's per-class or per-component metrics
3. Add a row for every distinct approach tested (not every hyperparameter tweak)
4. Record tuning sweeps below the table as bullet points
5. Mark the current best with **bold** in the Notes column
6. Include failed experiments — knowing what doesn't work is valuable
