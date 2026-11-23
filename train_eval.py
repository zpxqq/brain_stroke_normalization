"""Обучение 3D U-Net для одного варианта предобработки и одного seed, затем оценка на test.

Все настройки, кроме варианта предобработки, одинаковы: те же разбиение, архитектура (create_unet3d проекта),
потери Dice + BCE, AdamW (lr задаётся параметром --lr), батч 1, одинаковая последовательность патчей и аугментаций (seed),
выбор контрольной точки по среднему по пациентам Dice на val (порог 0,5, весь объём), порог 0,5 на test без постобработки.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.networks.nets import UNet

from common import PATCH, TrainSet, load_case, make_split, metrics
from normalization import preprocess


def create_unet3d(in_channels=3):
    return UNet(spatial_dims=3, in_channels=in_channels, out_channels=1, channels=(8, 16, 32, 64, 128),
                strides=(2, 2, 2, 2), num_res_units=2)


class SegmentationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss(sigmoid=True)
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, prediction, target):
        return self.dice(prediction, target) + self.bce(prediction, target)


@torch.inference_mode()
def evaluate(model, sids, variant, threshold=0.5):
    model.eval()
    rows = []
    for sid in sids:
        image, mask, spacing, brain = load_case(sid)
        x = torch.from_numpy(preprocess(image, brain, variant))[None]
        logits = sliding_window_inference(x, PATCH, 1, model)
        pred = (torch.sigmoid(logits)[0, 0].numpy() > threshold)
        rows.append(dict(subject_id=sid, **metrics(pred, mask > 0, spacing)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--val-every", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=0, help="только для отладки")
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bf16", type=int, default=1, help="bfloat16 autocast при обучении (AMX-процессор)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    run = a.out / f"v{a.variant}_s{a.seed}"
    run.mkdir(parents=True, exist_ok=True)

    split = make_split()
    train_ids = split[split.split == "train"].subject_id.tolist()
    val_ids = split[split.split == "val"].subject_id.tolist()
    test_ids = split[split.split == "test"].subject_id.tolist()
    if a.max_steps:
        train_ids, val_ids, test_ids = train_ids[:a.max_steps], val_ids[:3], test_ids[:3]

    torch.manual_seed(a.seed)
    model = create_unet3d()
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    criterion = SegmentationLoss()
    data = TrainSet(train_ids, a.variant, a.seed)

    log, best_dice = [], -1.0
    for epoch in range(1, a.epochs + 1):
        started = time.monotonic()
        data.set_epoch(epoch)
        order = np.random.RandomState(a.seed * 7919 + epoch).permutation(len(data))
        model.train()
        total = 0.0
        for i in order:
            x, y = data[int(i)]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bool(a.bf16)):
                logits = model(x[None])
            loss = criterion(logits.float(), y[None])
            loss.backward()
            optimizer.step()
            total += loss.item()
        row = dict(epoch=epoch, train_loss=total / len(order), seconds=time.monotonic() - started)
        if epoch % a.val_every == 0 or epoch == a.epochs:
            val = evaluate(model, val_ids, a.variant)
            row.update(val_dice=float(val.dice.mean()), val_iou=float(val.iou.mean()))
            if row["val_dice"] > best_dice:
                best_dice = row["val_dice"]
                torch.save(model.state_dict(), run / "best.pt")
                row["best"] = True
        log.append(row)
        pd.DataFrame(log).to_csv(run / "log.csv", index=False)
        print(row, flush=True)

    model.load_state_dict(torch.load(run / "best.pt", weights_only=True))
    test = evaluate(model, test_ids, a.variant)
    test.insert(0, "seed", a.seed)
    test.insert(0, "variant", a.variant)
    test.to_csv(run / "test_metrics.csv", index=False)
    val = evaluate(model, val_ids, a.variant)
    val.to_csv(run / "val_metrics.csv", index=False)
    print("TEST", test[["dice", "iou", "precision", "recall", "hd95"]].mean().to_dict(), flush=True)


if __name__ == "__main__":
    main()
