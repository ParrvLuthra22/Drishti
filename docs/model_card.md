# drishti-fight-detector

## Model description

VideoMAE-base fine-tuned for binary fight/violence
detection in surveillance camera footage.

- **Base model:** MCG-NJU/videomae-base
- **Task:** Video classification (fight detection)
- **Training data:** RWF-2000 dataset
- **Published:** Parrv/drishti-fight-detector

## Training details

| Parameter | Value |
|-----------|-------|
| Base model | VideoMAE-base (86M params) |
| Dataset | RWF-2000 (1576 train, 400 val) |
| Epochs | 5 (1 head-only + 4 full fine-tune) |
| Batch size | 4 |
| Head LR | 1e-3 |
| Backbone LR | 1e-5 |
| Scheduler | CosineAnnealingLR |
| Hardware | NVIDIA T4 (Google Colab) |
| Training time | 62.5 minutes |
| Input | 16 frames × 224×224, sampled at 1 frame/0.3s |

## Results

| Metric | Value |
|--------|-------|
| Val accuracy | 88.75% |
| Val F1 (macro) | 88.74% |
| NonFight precision/recall | 0.867 / 0.915 |
| Fight precision/recall | 0.910 / 0.860 |

Confusion matrix (NonFight, Fight):
[[183, 17], [28, 172]]

## Training strategy

Two-phase fine-tuning to prevent catastrophic
forgetting of pre-trained VideoMAE features:

1. **Phase A (epoch 1):** Freeze all backbone
   weights. Train only the 2-class classifier
   head. Establishes stable feature → class
   mapping before touching the backbone.

2. **Phase B (epochs 2-5):** Unfreeze all weights.
   Train with differential learning rates
   (backbone 100× lower than head) + cosine
   annealing. F1 jumps from 0.62 to 0.86 in
   the first full fine-tune epoch.

## Intended use

Security camera monitoring for fight/violence
detection. Best performance on overhead
surveillance footage with multiple people in
frame — matches the RWF-2000 training distribution.

## Limitations

- **Domain:** Trained on CCTV-style overhead footage.
  Close-up single-person webcam footage is
  out-of-distribution.
- **Temporal:** Requires 16 frames sampled at
  1 frame/0.3s (spanning ~5 seconds).
  Consecutive frames at full FPS give poor results.
- **Scale:** Base model only. Larger VideoMAE
  variants approach 95% F1 on RWF-2000.

## Citation / project

Part of Project Drishti — AI situational
awareness system.
GitHub: https://github.com/ParrvLuthra22/Drishti
