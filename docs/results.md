# Results

## Training Summary

| Metric | Value |
|--------|-------|
| Total epochs | 60 (Phase 1: 30 + Phase 2: 30) |
| Best validation accuracy | 97.09% (epoch 59) |
| Final training accuracy | 98.90% (epoch 60) |
| Final validation accuracy | 96.70% (epoch 60) |
| Max train/val gap (Phase 1) | 3.0% (epoch 9) |
| Max train/val gap (Phase 2) | 2.3% (epoch 38) |
| Training time | 5.57 h (Kaggle P100 GPU) |
| Best checkpoint size | 117 MB |

## Test Performance

| Metric | Real only | Real+Syn |
|--------|-----------|----------|
| Accuracy | 1.0000 | 1.0000 |
| Macro F1 | 1.0000 | 1.0000 |
| Macro Precision | 1.0000 | 1.0000 |
| Macro Recall | 1.0000 | 1.0000 |

NOTE: The research paper reports 98% / 97% in some sections. To be reconciled before final publication.

## Ablation Studies

### Effect of Mixup

| Configuration | Train/Val Gap |
|---------------|---------------|
| Baseline (80/10/10, small val) | ~0% (memorised) |
| 70/20/10, 4-block unfreeze | 10-14% |
| Mixup enabled Phase 1 | 9-13% (soft-label artifact) |
| Final (this work) | less than 3% |

### Phase 2 Backbone LR

| Backbone LR | Train Acc | Val Acc | Gap |
|-------------|-----------|---------|-----|
| 1e-4 | 83-86% | 97-99% | ~13% |
| 1e-5 | 86-90% | 96-98% | ~8% |
| 1e-6 (selected) | 97-99% | 96-97% | less than 3% |

## Plots

- results/plots/training_curves.png
- results/plots/f1_comparison.png
- results/plots/confusion_matrix_real.png
- results/plots/confusion_matrix_combined.png
- results/plots/dataset_distribution.png
