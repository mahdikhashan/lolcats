Results: 10 (excluded: LoLCATs attention)

| # | Checkpoint | Positions 0–127 | Positions 128–511 | Positions 512–2047 | All positions (stage 1 loss) | PIQA | ARC-Easy |
|---|---|---|---|---|---|---|---|
| 1 | Second round, config 1 | 4.738 | 4.18 | 3.862 | 3.976 | 57.3 | 36.5 |
| 2 | Second round, config 2 (clipping) | 4.722 | 3.919 | 3.306 | 3.509 | 57.5 | 35.6 |
| 3 | fd128, LoLCATs recipe, bf16 | 4.375 | 3.668 | 3.054 | 3.251 | 57.6 | 39.0 |
| 4 | fd32, LoLCATs recipe, bf16 | 4.643 | 3.828 | 3.213 | 3.418 | – | – |
| 5 | fd32, paper recipe, bf16 | 5.309 | 6.976 | 8.677 | 8.146 | 55.8 | 34.1 |
| 6 | fd32, paper recipe, float32 | 4.731 | 4.74 | 5.018 | 4.948 | 57.7 | 35.7 |
| 7 | v2, C1: alpha per head | 4.548 | 4.002 | 3.699 | 3.809 | 57.5 | 34.6 |
| 8 | v2, R1b: per-head, hybrid | 2.661 | 2.892 | 3.127 | 3.054 | 55.9 | 29.9 |
| 9 | v2, window_rope, fd128, LoLCATs recipe, bf16 | 2.548 | 1.25 | 1.116 | 1.231 | 61.5 | 43.3 |
| 10 | v2, window_rope, hybrid, fd128, LoLCATs recipe, bf16 | 1.864 | 1.356 | 1.665 | 1.619 | 58.9 | 36.0 |

Rank correlation (Spearman) between the loss and the accuracy. Below 0: a lower loss comes with a higher accuracy. p: two-sided permutation p-value.

| Score | PIQA ρ | PIQA p | PIQA n | ARC-Easy ρ | ARC-Easy p | ARC-Easy n |
|---|---|---|---|---|---|---|
| Positions 0–127 | -0.63 | 0.077 | 9 | -0.30 | 0.44 | 9 |
| Positions 128–511 | -0.58 | 0.11 | 9 | -0.40 | 0.29 | 9 |
| Positions 512–2047 | -0.64 | 0.068 | 9 | -0.52 | 0.16 | 9 |
| All positions (stage 1 loss) | -0.58 | 0.11 | 9 | -0.40 | 0.29 | 9 |
