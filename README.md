# Topology-Aware Attention U-Net  
### Multi-Task Retinal Vessel + Artery–Vein Segmentation, CIMT Classification & Regression

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Publication-ready** multi-task deep learning framework for fundus image analysis.

| Task | Output | Type |
|------|--------|------|
| **Task 1** | CIMT Classification | Binary (Normal / Thickened) |
| **Task 2** | CIMT Regression | Continuous thickness (mm) |
| **Task 3** | Vessel Segmentation | Binary mask + **topology-aware** losses (soft-clDice, thin/branch BCE) |
| **Task 4** | Artery–Vein Segmentation | Multi-class (Background / Artery / Vein) |

**Backbone:** Attention U-Net (Oktay et al.) with Attention Gates on skip connections.

---

## Key Results (Held-out Test Set)

| Metric                        | Value                          |
|-------------------------------|--------------------------------|
| Classification AUC            | **0.9486**                     |
| Classification Accuracy       | 0.885                          |
| Vessel Dice (whole image)     | ~0.93–0.94                     |
| Topology-aware improvements   | Soft-clDice + thin/branch BCE  |

> Full metrics with 95% patient-level bootstrap CIs are in `tables/` and `results/`.

---

## Repository Structure

```text
Fundus-CIMT-Attention-UNet/
├── config/
│   └── config.yaml              # All hyperparameters
├── data/
│   ├── raw/                     # Place original images, masks, CSV here
│   ├── processed/
│   └── splits/                  # Patient-level train/val/test lists
├── src/
│   ├── dataset.py
│   ├── models/
│   │   ├── attention_unet.py
│   │   └── losses.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── utils/
│       ├── seed.py
│       ├── metrics.py
│       └── journal_style.py     # 600 dpi, Times New Roman helpers
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_train.ipynb
│   ├── 03_evaluate_and_figures.ipynb   # All journal figures & tables
│   └── 04_ablation.ipynb
├── figures/                     # 600 dpi PNG + PDF + TIFF
├── tables/                      # CSV + LaTeX
├── checkpoints/
├── results/
├── scripts/
│   ├── export_journal_package.py
│   └── download_data.sh
├── docs/
│   ├── METHOD.md
│   ├── LIMITATIONS.md
│   and REPORTING_CHECKLIST.md
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

```bash
git clone https://github.com/amimulhasan/Fundus-Image-.git
cd Fundus-CIMT-Attention-UNet

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Data Preparation

1. Download the Fundus-CIMT dataset and artery-vein masks.
2. Place files according to paths in `config/config.yaml`:

```yaml
data:
  images_dir: data/raw/Fundus_CIMT_2903
  masks_dir:  data/raw/predicted_masks_5806
  csv_path:   data/raw/data_info.csv
  av_masks_dir: data/raw/Fundus_A_and_V
```

3. Patient-level split is created automatically (no patient overlap between train/val/test).

---

## Training

```bash
python src/train.py --config config/config.yaml
```

- Mixed precision (AMP) enabled by default
- Checkpoints saved to `checkpoints/`
- Training history → `results/training_history.csv`

---

## Evaluation & Publication Figures

```bash
python src/evaluate.py --checkpoint checkpoints/best_model.pth --config config/config.yaml
```

Or open the notebook:

```bash
jupyter notebook notebooks/03_evaluate_and_figures.ipynb
```

This generates **all** journal-quality figures (600 dpi, Times New Roman 12 pt bold) and tables:

- Fig. 1  Dataset overview  
- Fig. 2  Architecture  
- Fig. 3  Test performance (ROC / PR / scatter / Bland–Altman / calibration)  
- Fig. 4  Qualitative vessel + A/V  
- Fig. 5  Topology quantitative  
- Fig. 6  Training curves  
- … up to Fig. 16 (qualitative zoom)  
- Tables 1–23 (CSV + LaTeX)

---

## Export Journal Package

```bash
python scripts/export_journal_package.py
```

Creates `journal_package_extended.zip` containing every figure, table, caption file and result CSV.

---

## Configuration

All hyperparameters live in `config/config.yaml`. Important ones:

| Parameter       | Default | Description                     |
|-----------------|---------|---------------------------------|
| `img_size`      | 512     | Input resolution                |
| `batch_size`    | 8       |                                 |
| `epochs`        | 30      |                                 |
| `base_ch`       | 32      | Attention U-Net base channels   |
| `w_cl` / `w_wb` | 0.5     | Topology loss weights           |
| `thin_boost`    | 2.0     | Thin vessel weighting           |
| `branch_boost`  | 4.0     | Branch point weighting          |

---

## Citation

If you use this code or the trained models, please cite:

```bibtex
@article{yourpaper2026,
  title   = {Topology-Aware Attention U-Net for Multi-Task Retinal Vessel and CIMT Analysis},
  author  = {Hasan, Amimul and ...},
  journal = {...},
  year    = {2026}
}
```

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

---

## Contact

- GitHub: [amimulhasan](https://github.com/amimulhasan)
- Issues: Please open a GitHub issue for bugs or questions.
