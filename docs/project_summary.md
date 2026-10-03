# Project Summary

## Title

Vision Transformer Driven Mango Leaf Disease Detection with Targeted Spraying Using UAV

## Team

- Jamal Khan (22PWCSE2203)
- Naveed Ahmad (22PWCSE2165)
- Omer Khan (22PWCSE2130)

## Supervisor

Dr. Yasir Saleem Afridi

## Institution

Department of Computer Systems Engineering
University of Engineering and Technology (UET), Peshawar, Pakistan
2022 - 2026

## Problem

Manual inspection and blanket pesticide spraying in mango orchards are:
- Time-consuming
- Labour-intensive
- Costly
- Environmentally harmful
- Often ineffective in large orchards

## Solution

An integrated system combining:
1. Vision Transformer (DeiT-Small) classifier for 8-class mango leaf disease detection
2. CycleGAN-based domain adaptation to bridge ground-to-aerial imagery gap
3. Custom hexacopter UAV with relay-driven targeted spraying mechanism

## Key Contributions

1. CycleGAN domain adaptation without requiring real drone imagery
2. Custom LayerNorm classification head for stable ViT training
3. Experimental evidence that Mixup causes a fake train/val gap
4. Validated micro learning rate progressive unfreezing for ViT fine-tuning on small agricultural datasets

## Results

- Validation accuracy: 97.09%
- Test accuracy: 100% (thesis) / 98% (paper)
- Train/val gap: less than 3% across all 60 epochs
- Training time: 5.57 h on Kaggle P100 GPU
- Model size: 117 MB

## Documentation

- docs/hardware.md - hexacopter UAV specifications
- docs/dataset.md - MangoLeafBD and CycleGAN synthetic data
- docs/results.md - training curves and ablation studies
- docs/paper/ - thesis PDF and research paper PDF
- docs/hardware/ - drone photos
