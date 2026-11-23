import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from sklearn.model_selection import train_test_split

from normalization import brain_mask_from_adc, preprocess

CACHE = Path("/tmp/cache")
PATCH = (96, 96, 32)


def make_split(random_state=42):
    """Разбиение так же, как в preprocessing/split_dataset.py проекта (70/15/15, по пациентам, стратификация по наличию очага)."""
    cases = json.loads((CACHE / "cases.json").read_text(encoding="utf-8"))
    df = pd.DataFrame(cases)[["subject_id", "lesion_voxels", "lesion_ml"]]
    df["has_lesion"] = (df.lesion_voxels > 0).astype(int)
    train_val, test = train_test_split(df, test_size=0.15, random_state=random_state, stratify=df.has_lesion)
    train, val = train_test_split(train_val, test_size=0.15 / 0.85, random_state=random_state,
                                  stratify=train_val.has_lesion)
    out = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
    return out.sort_values("subject_id").reset_index(drop=True)


def load_case(sid):
    z = np.load(CACHE / f"{sid}.npz")
    brain_file = CACHE / "brain" / f"{sid}.npy"
    if brain_file.exists():
        brain = np.load(brain_file)
    else:
        brain = brain_mask_from_adc(z["image"][1])
    return z["image"], z["mask"], z["spacing"], brain


def pad_to(arr, size):
    pad = [(0, max(0, s - d)) for d, s in zip(arr.shape[-3:], size)]
    if not any(p[1] for p in pad):
        return arr
    return np.pad(arr, [(0, 0)] * (arr.ndim - 3) + pad)


class TrainSet(torch.utils.data.Dataset):
    """Патч 96×96×32 с центром в очаге или фоне в соотношении 1:1 (как RandCropByPosNegLabeld pos=1, neg=1), flip и rot90 с p = 0,5."""

    def __init__(self, sids, variant, seed):
        self.sids, self.variant, self.seed, self.epoch = list(sids), variant, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.sids)

    def __getitem__(self, i):
        image, mask, _, brain = load_case(self.sids[i])
        x = pad_to(preprocess(image, brain, self.variant), PATCH)
        y = pad_to(mask[None].astype(np.float32), PATCH)
        rng = np.random.RandomState(self.seed * 1_000_003 + self.epoch * 1009 + i)
        shape = np.array(x.shape[1:])
        fg = np.argwhere(y[0] > 0)
        if len(fg) and rng.rand() < 0.5:
            center = fg[rng.randint(len(fg))]
        else:
            center = np.array([rng.randint(s) for s in shape])
        half = np.array(PATCH) // 2
        start = np.clip(center - half, 0, shape - np.array(PATCH))
        sl = tuple(slice(s, s + p) for s, p in zip(start, PATCH))
        x, y = x[(slice(None),) + sl], y[(slice(None),) + sl]
        if rng.rand() < 0.5:
            x, y = x[:, ::-1], y[:, ::-1]
        if rng.rand() < 0.5:
            k = rng.randint(1, 4)
            x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k, axes=(1, 2))
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


def surface(mask):
    return mask & ~ndimage.binary_erosion(mask)


def hd95(pred, gt, spacing):
    """95-й перцентиль симметричных расстояний между поверхностями (мм). NaN, если одна из масок пуста."""
    if not pred.any() or not gt.any():
        return float("nan")
    sp, sg = surface(pred), surface(gt)
    d_to_gt = ndimage.distance_transform_edt(~sg, sampling=spacing)[sp]
    d_to_pred = ndimage.distance_transform_edt(~sp, sampling=spacing)[sg]
    return float(np.percentile(np.concatenate([d_to_gt, d_to_pred]), 95))


def metrics(pred, gt, spacing):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
    if tp + fp + fn == 0:
        dice = iou = 1.0
    else:
        dice, iou = 2 * tp / (2 * tp + fp + fn), tp / (tp + fp + fn)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else float("nan")
    vox_ml = float(np.prod(spacing)) / 1000.0
    return dict(dice=dice, iou=iou, precision=precision, recall=recall, hd95=hd95(pred, gt, spacing),
                gt_ml=(tp + fn) * vox_ml, pred_ml=(tp + fp) * vox_ml, tp=tp, fp=fp, fn=fn)
