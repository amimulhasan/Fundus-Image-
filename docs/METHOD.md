# Method

## 1. Overview

We propose a **topology-aware multi-task Attention U-Net** that simultaneously performs:

1. Binary vessel segmentation  
2. Artery–vein (A/V) segmentation  
3. CIMT classification (Normal vs Thickened)  
4. CIMT regression (continuous thickness in mm)

The model is trained end-to-end with a combined loss that includes classical segmentation terms and explicit topology-preserving terms.

## 2. Architecture

- **Encoder–Decoder**: classic U-Net style with 5 resolution levels  
- **Attention Gates** (Oktay et al., 2018) on every skip connection  
- **Shared encoder** features feed four task-specific heads:  
  - Vessel head → 1-channel sigmoid  
  - A/V head → 3-class softmax  
  - Classification head → global average pool + linear  
  - Regression head → global average pool + linear  

Base channel count = 32 (configurable).

## 3. Topology-Aware Losses

### 3.1 Soft-clDice
A differentiable approximation of the clDice metric that encourages topological correctness (connectivity of the vessel skeleton).

### 3.2 Thin / Branch Weighted BCE
Pixels belonging to thin vessels and branch points receive higher weight (`thin_boost`, `branch_boost`) so that the network does not ignore fine structures.

Final vessel loss:

```
L_vessel = λ · Dice + (1-λ) · BCE + w_cl · soft-clDice + w_wb · weighted-BCE
```

## 4. Multi-Task Objective

```
L_total = w_seg·L_vessel + w_av·L_AV + w_cls·L_cls + w_reg·L_reg
```

All weights default to 1.0 and can be tuned in `config.yaml`.

## 5. Training Protocol

- Optimizer: AdamW (lr = 1e-4, weight decay = 1e-5)  
- Mixed precision (AMP)  
- Input size: 512 × 512  
- Batch size: 8  
- Epochs: 30  
- Patient-level split (no subject leakage)  
- Decision threshold for classification selected on the validation set only  

## 6. Evaluation

- Patient-level bootstrap (1000 resamples) for 95 % confidence intervals  
- Metrics: AUC, Accuracy, F1, MCC, Dice, clDice, sensitivity, specificity, MAE, RMSE, R², CCC, etc.  
- Qualitative inspection with error maps and zoomed ROIs chosen from the ground-truth skeleton density.
