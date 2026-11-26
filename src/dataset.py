"""
Dataset & patient-level split for Fundus-CIMT multi-task learning.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class FundusCIMTDataset(Dataset):
    def __init__(self, df, images_dir, masks_dir, av_masks_dir, img_size=512, is_train=False):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.av_masks_dir = Path(av_masks_dir)
        self.img_size = img_size
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def _load_image(self, name):
        path = self.images_dir / name
        img = Image.open(path).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        return TF.to_tensor(img)

    def _load_mask(self, name):
        # vessel mask
        stem = Path(name).stem
        candidates = list(self.masks_dir.glob(f"{stem}*"))
        if not candidates:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        else:
            m = Image.open(candidates[0]).convert("L")
            m = m.resize((self.img_size, self.img_size), Image.NEAREST)
            mask = (np.array(m) > 127).astype(np.float32)
        return torch.from_numpy(mask).unsqueeze(0)

    def _load_av(self, name):
        stem = Path(name).stem
        candidates = list(self.av_masks_dir.glob(f"{stem}*"))
        if not candidates:
            av = np.zeros((self.img_size, self.img_size), dtype=np.int64)
            has_av = False
        else:
            m = Image.open(candidates[0])
            m = m.resize((self.img_size, self.img_size), Image.NEAREST)
            av = np.array(m)
            # assume 0=bg, 1=artery, 2=vein (adjust if needed)
            if av.ndim == 3:
                av = av[..., 0]
            has_av = True
        return torch.from_numpy(av.astype(np.int64)), has_av

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = row["image_name"]

        image = self._load_image(name)
        vessel = self._load_mask(name)
        av, has_av = self._load_av(name)

        sample = {
            "image": image,
            "vessel": vessel,
            "av": av,
            "has_av": has_av,
            "cls": torch.tensor(row["label"], dtype=torch.float32),
            "reg": torch.tensor(row["thickness"], dtype=torch.float32),
            "name": name,
            "patient_id": name.split("_")[0],   # simple patient id extraction
        }
        return sample


def make_patient_splits(csv_path, val_patients=200, test_patients=100, seed=42):
    """
    Create patient-level train / val / test split with no overlap.
    Returns three dataframes.
    """
    df = pd.read_csv(csv_path)
    df["patient_id"] = df["image_name"].apply(lambda x: x.split("_")[0])

    patients = df["patient_id"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(patients)

    test_p = set(patients[:test_patients])
    val_p = set(patients[test_patients:test_patients + val_patients])
    train_p = set(patients[test_patients + val_patients:])

    train_df = df[df["patient_id"].isin(train_p)].reset_index(drop=True)
    val_df = df[df["patient_id"].isin(val_p)].reset_index(drop=True)
    test_df = df[df["patient_id"].isin(test_p)].reset_index(drop=True)

    print(f"patients train/val/test = {len(train_p)}/{len(val_p)}/{len(test_p)}")
    print(f"images  train/val/test = {len(train_df)}/{len(val_df)}/{len(test_df)}")
    return train_df, val_df, test_df
