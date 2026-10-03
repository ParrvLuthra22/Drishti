# Phase 4 — VideoMAE Fight Detection Results

## Model
- Base: MCG-NJU/videomae-base
- Dataset: RWF-2000 (776 Fight + 800 NonFight train, 400 val)
- Training: 5 epochs, 62.5 minutes on Colab T4

## Results by epoch

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val F1 |
|-------|-----------|-----------|----------|---------|--------|
| 1 (head only) | 0.6323 | 63.9% | 0.5836 | 62.5% | 0.624 |
| 2 (full model) | 0.4972 | 74.9% | 0.3332 | 86.0% | 0.860 |
| 3 | 0.1978 | 92.3% | 0.3858 | 84.8% | 0.847 |
| 4 | 0.0489 | 98.3% | 0.3631 | 88.0% | 0.880 |
| 5 | 0.0101 | 99.9% | 0.3916 | 88.8% | **0.887** |

## Best model (epoch 5)
- Val accuracy: 88.75%
- Val F1 (macro): 0.8874
- NonFight precision/recall: 0.867 / 0.915
- Fight precision/recall: 0.910 / 0.860
- Checkpoint: Parrv/drishti-fight-detector on HuggingFace Hub

## Notes
- 24 clips skipped during extraction (Cyrillic filenames too long for Linux FS)
- Slight overfitting visible from epoch 3 (train acc 99.9% vs val 88.8%)
- Production deployment: load from HuggingFace Hub, run on 16-frame clips
