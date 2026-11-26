#!/usr/bin/env python3
"""
Evaluation script – computes metrics and can be extended to generate all journal figures.
For full publication figures use notebooks/03_evaluate_and_figures.ipynb
"""

import argparse
import json
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FundusCIMTDataset, make_patient_splits
from models.attention_unet import AttentionUNet
from utils.seed import set_seed
from utils.metrics import binary_metrics, regression_metrics, dice_coefficient


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--checkpoint", required=True)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_df = make_patient_splits(
        cfg["data"]["csv_path"],
        val_patients=200,
        test_patients=100,
        seed=cfg["seed"],
    )
    test_ds = FundusCIMTDataset(
        test_df,
        cfg["data"]["images_dir"],
        cfg["data"]["masks_dir"],
        cfg["data"]["av_masks_dir"],
        img_size=cfg["img_size"],
        is_train=False,
    )
    loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

    model = AttentionUNet(
        in_channels=cfg["model"]["in_channels"],
        base_ch=cfg["model"]["base_ch"],
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_cls_true, all_cls_prob = [], []
    all_reg_true, all_reg_pred = [], []
    dice_scores = []

    for batch in tqdm(loader, desc="Evaluating"):
        images = batch["image"].to(device)
        outputs = model(images)

        prob = torch.sigmoid(outputs["cls"]).cpu().numpy()
        all_cls_prob.append(prob)
        all_cls_true.append(batch["cls"].numpy())

        all_reg_pred.append(outputs["reg"].cpu().numpy())
        all_reg_true.append(batch["reg"].numpy())

        pred_v = (outputs["vessel"].cpu().numpy() > 0.5)
        gt_v = batch["vessel"].numpy() > 0.5
        for i in range(pred_v.shape[0]):
            dice_scores.append(dice_coefficient(gt_v[i, 0], pred_v[i, 0]))

    cls_true = np.concatenate(all_cls_true)
    cls_prob = np.concatenate(all_cls_prob)
    reg_true = np.concatenate(all_reg_true)
    reg_pred = np.concatenate(all_reg_pred)

    cls_m = binary_metrics(cls_true, cls_prob)
    reg_m = regression_metrics(reg_true, reg_pred)

    summary = {
        "classification": cls_m,
        "regression": reg_m,
        "vessel_dice_mean": float(np.mean(dice_scores)),
        "vessel_dice_std": float(np.std(dice_scores)),
    }
    print(json.dumps(summary, indent=2))

    out_dir = Path(cfg["data"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary_test.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved → {out_dir / 'summary_test.json'}")


if __name__ == "__main__":
    main()
