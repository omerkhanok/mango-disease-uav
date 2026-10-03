# Dataset

## Source

MangoLeafBD — publicly available on Kaggle.
- 4,000 images
- 8 classes
- Balanced (500 images per class)

## Classes

1. Anthracnose
2. Bacterial Canker
3. Cutting Weevil
4. Dieback
5. Gall Midge
6. Healthy
7. Powdery Mildew
8. Sooty Mould

## Split

Stratified split on real images only:

| Split | Ratio | Approx. per class |
|-------|-------|-------------------|
| Train | 65% | ~325 |
| Validation | 20% | ~100 |
| Test | 15% | ~75 |

The larger validation split (20% instead of 10%) was chosen because a smaller validation set was easily memorized by the pretrained 
model.

## CycleGAN Synthetic Images

- 1,600 total synthetic drone-style images (200 per class)
- 100 per class -> training
- 100 per class -> validation
- 25 per class -> combined test set
- No synthetic images in the real-only test set

## After Augmentation

Offline augmentation (x3 copies per image): rotation, flip, color jitter, crop, blur.

| Split | Real | Synthetic | Augmented Total |
|-------|------|-----------|-----------------|
| Training | ~2,591 | 800 | ~10,173 |
| Validation | ~800 | 800 | ~4,800 |
| Test (real only) | ~600 | 0 | ~600 |
| Test (real + syn) | ~600 | 200 | ~800 |

## Leakage Audit

Two-stage verification before training:

1. MD5 hash — catches exact binary duplicates
   - 8 duplicates found between train and validation
   - 1 duplicate found between train and test
   - All 9 removed

2. ResNet-18 embeddings + cosine similarity (threshold >= 0.95)
   - Catches near-duplicates from augmentation or JPEG recompression
   - Removed images logged in results/logs/similar_pairs_report.txt
   - Top similar pairs visualized in results/logs/similar_pairs_visualization.png

## Directory Layout

data/
- samples/        # a few example images (small, tracked in git)
- raw_images/     # full dataset (NOT tracked, stored externally)
- raw_videos/     # drone flight videos (NOT tracked, stored externally)

Raw data is stored externally (Google Drive / Kaggle) to keep the repository lightweight.
