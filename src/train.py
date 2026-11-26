#!/usr/bin/env python3
"""
Training script for Topology-Aware Attention U-Net (multi-task).
"""

import argparse
import os
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FundusCIMTDataset, make_patient_splits
from models.attention_unet import AttentionUNet
from models.losses import MultiTaskLoss
from utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # data
    train_df, val_df, _ = make_patient_splits(
        cfg["data"]["csv_path"],
        val_patients=200,
        test_patients=100,
        seed=cfg["seed"],
    )
    train_ds = FundusCIMTDataset(
        train_df,
        cfg["data"]["images_dir"],
        cfg["data"]["masks_dir"],
        cfg["data"]["av_masks_dir"],
        img_size=cfg["img_size"],
        is_train=True,
    )
    val_ds = FundusCIMTDataset(
        val_df,
        cfg["data"]["images_dir"],
        cfg["data"]["masks_dir"],
        cfg["data"]["av_masks_dir"],
        img_size=cfg["img_size"],
        is_train=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
    )

    # model & loss
    model = AttentionUNet(
        in_channels=cfg["model"]["in_channels"],
        base_ch=cfg["model"]["base_ch"],
        n_classes_vessel=cfg["model"]["n_classes_vessel"],
        n_classes_av=cfg["model"]["n_classes_av"],
    ).to(device)

    criterion = MultiTaskLoss(cfg["train"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["amp"])

    os.makedirs(cfg["data"]["checkpoint_dir"], exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['train']['epochs']}")
        for batch in pbar:
            images = batch["image"].to(device)
            targets = {
                "vessel": batch["vessel"].to(device),
                "av": batch["av"].to(device),
                "cls": batch["cls"].to(device),
                "reg": batch["reg"].to(device),
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg["train"]["amp"]):
                outputs = model(images)
                losses = criterion(outputs, targets)
            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
            running += losses["total"].item()
            pbar.set_postfix(loss=f"{losses['total'].item():.4f}")

        # simple validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                targets = {
                    "vessel": batch["vessel"].to(device),
                    "av": batch["av"].to(device),
                    "cls": batch["cls"].to(device),
                    "reg": batch["reg"].to(device),
                }
                outputs = model(images)
                losses = criterion(outputs, targets)
                val_loss += losses["total"].item()
        val_loss /= max(len(val_loader), 1)
        print(f"Epoch {epoch}: train_loss={running/len(train_loader):.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = Path(cfg["data"]["checkpoint_dir"]) / "best_model.pth"
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, ckpt)
            print(f"  → saved {ckpt}")

    print("Training finished.")


if __name__ == "__main__":
    main()
