| Comparison | Task (metric) | n | A | B | A − B | Paired SE | z | Only A / only B | McNemar p | Holm p | 95% bootstrap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| window_rope vs teacher | piqa (acc) | 1,838 | 73.1 | 74.4 | -1.4 | 0.70 | -1.9 | 71 / 96 | 0.063 | 0.24 | [-2.7, +0.0] |
| window_rope vs teacher | arc_easy (acc) | 2,376 | 63.8 | 65.3 | -1.5 | 0.78 | -1.9 | 155 / 191 | 0.06 | 0.24 | [-3.0, +0.0] |
| window_rope vs teacher | arc_challenge (acc_norm) | 1,172 | 36.1 | 36.3 | -0.3 | 1.17 | -0.2 | 92 / 95 | 0.88 | 1 | [-2.6, +2.0] |
| window_rope vs teacher | hellaswag (acc_norm) | 10,042 | 60.7 | 63.7 | -3.0 | 0.31 | -9.7 | 336 / 637 | 3.1e-22 | 1.6e-21 | [-3.6, -2.4] |
| window_rope vs teacher | winogrande (acc) | 1,267 | 59.6 | 60.1 | -0.6 | 1.47 | -0.4 | 169 / 176 | 0.75 | 1 | [-3.3, +2.4] |

Accuracy in %. A difference is clear when the Holm p-value is below 0.05.
