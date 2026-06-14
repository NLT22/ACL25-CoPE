# CoPE Training Results on ITCPR

## Experimental Setup

The ITCPR CoPE experiments were trained on **CelebReID** and validated
cross-domain on **LAST**. The reported best checkpoints were selected using
LAST validation performance.

The main 100-epoch training configuration uses:

- Backbone: `openai/clip-vit-large-patch14`
- Batch size: `6`
- Warmup: `1` epoch
- Training source: `Celeb-reID`
- Validation source: `LAST`
- Best-checkpoint metric: `R@1`

## 100-Epoch Training Results

| Configuration | Best epoch | LAST R@1 | LAST R@5 | LAST R@10 |
|---|---:|---:|---:|---:|
| `alpha=1, beta=0` | 69 | **52.82** | **78.23** | 85.41 |
| Example config, likely `alpha=0.9, beta=0.1` | 88 | 51.93 | 77.13 | **85.64** |
| `alpha=0, beta=1` | 1 | 34.48 | 58.56 | 69.94 |
| `alpha=0.5, beta=0.5` | 0 | 28.29 | 51.71 | 63.43 |

The `alpha=0.5, beta=0.5` run only contains an epoch-0 best checkpoint.
Therefore, it should not be treated as a confirmed completed 100-epoch run.

## Main Finding

The `alpha=1, beta=0` configuration performs best overall:

- Highest `R@1`: **52.82**
- Highest `R@5`: **78.23**
- `R@10=85.41`, close to the example configuration's best value of `85.64`

Its performance decreases by epoch 99:

| Checkpoint | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| Best epoch 69 | 52.82 | 78.23 | 85.41 |
| Final epoch 99 | 48.29 | 74.81 | 83.31 |

The `alpha=0, beta=1` setting performs substantially worse. The same behavior
appears in both the 30-epoch and 100-epoch experiments: configurations relying
more heavily on `beta` produce lower retrieval performance.

## Alpha=1 Cross-Domain Clean Evaluation

The best `alpha=1, beta=0` checkpoint was also evaluated using the shared
robustness benchmark loader:

| Evaluation source | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| LAST | **51.93** | **77.46** | **84.53** |
| PRCC | **42.47** | **70.55** | **84.25** |

These values may differ slightly from training-time LAST validation because
they are produced by a separate benchmark evaluation pipeline.

## Older 30-Epoch Training Results

| Configuration | Best epoch | LAST R@1 | LAST R@5 | LAST R@10 |
|---|---:|---:|---:|---:|
| `alpha=1, beta=0` | 15 | **49.61** | 73.48 | 82.43 |
| Example config | 29 | 49.06 | **73.70** | **83.65** |
| `alpha=0.5, beta=0.5` | 13 | 38.90 | 63.43 | 73.70 |
| `alpha=0, beta=1` | 1 | 35.03 | 59.12 | 71.49 |

## Interpretation

The current experiments provide consistent evidence that:

1. Training longer improves the strongest CoPE configurations.
2. `alpha=1, beta=0` gives the best `R@1` and `R@5`.
3. Increasing reliance on `beta` reduces performance in the current code and
   dataset setup.
4. The final checkpoint is not necessarily the best checkpoint; model
   selection using validation recall remains necessary.

These results describe the behavior of the current implementation and local
ITCPR setup. They do not by themselves prove that the CoPE paper's loss design
is ineffective, because implementation details, dataset differences, and
evaluation protocol can affect the result.

## Result Files

- [`alpha=1, beta=0` best checkpoint metadata](checkpoints/itcpr_celeb_to_last_100epoch_alpha1_beta0/best_model/metadata.json)
- [`alpha=1, beta=0` final checkpoint metadata](checkpoints/itcpr_celeb_to_last_100epoch_alpha1_beta0/last_checkpoint/metadata.json)
- [Example-config best checkpoint metadata](checkpoints/itcpr_celeb_to_last_100epoch_as_example/best_model/metadata.json)
- [`alpha=0, beta=1` best checkpoint metadata](checkpoints/itcpr_celeb_to_last_100epoch_alpha0_beta1/best_model/metadata.json)
- [`alpha=0.5, beta=0.5` best checkpoint metadata](checkpoints/itcpr_celeb_to_last_100epoch_alpha05_beta05/best_model/metadata.json)
- [Alpha=1 LAST robustness evaluation](../Benchmark-Robustness-Text-Image-Compose-Retrieval/results/robustness_cope_alpha1_LAST/REPORT.md)
- [Alpha=1 PRCC robustness evaluation](../Benchmark-Robustness-Text-Image-Compose-Retrieval/results/robustness_cope_alpha1_PRCC/REPORT.md)
