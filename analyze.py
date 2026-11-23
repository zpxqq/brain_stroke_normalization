"""Сводка результатов: таблица метрик, парные тесты Вилкоксона с поправкой Холма, рисунки 4-6."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

from common import PATCH, load_case, make_split
from normalization import adc_to_common_units, preprocess

NAMES = {1: "1. Baseline (Z по I≠0)", 2: "2. Z в маске мозга", 3: "3. Clipping + Z", 4: "4. Z (DWI, FLAIR) + шкала ADC"}
SHORT = {1: "Вариант 1", 2: "Вариант 2", 3: "Вариант 3", 4: "Вариант 4"}
COLORS = {1: "#d9a404", 2: "#2f9e5b", 3: "#1f77b4", 4: "#8e44ad"}
VOLUME_BINS = [0, 5, 20, np.inf]
VOLUME_LABELS = ["< 5 мл", "5–20 мл", "> 20 мл"]
METRICS = ["dice", "iou", "precision", "recall", "hd95"]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def bootstrap_ci(diff, n=10000, seed=0):
    rng = np.random.RandomState(seed)
    means = [diff[rng.randint(len(diff), size=len(diff))].mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


def fmt(mean, sd, digits=3):
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(exist_ok=True)

    frames = [pd.read_csv(f) for f in sorted(a.runs.glob("v*_s*/test_metrics.csv"))]
    data = pd.concat(frames, ignore_index=True)
    variants = sorted(data.variant.unique())
    seeds = sorted(data.seed.unique())
    data.to_csv(a.out / "test_metrics_all_runs.csv", index=False)

    per_patient = data.groupby(["variant", "subject_id"])[METRICS + ["gt_ml"]].mean(numeric_only=True).reset_index()
    per_patient.to_csv(a.out / "test_metrics_per_patient_seed_mean.csv", index=False)

    summary = {"seeds": [int(s) for s in seeds], "n_test": int(per_patient.subject_id.nunique()), "variants": {}}
    rows = []
    for v in variants:
        sub = per_patient[per_patient.variant == v]
        by_seed = data[data.variant == v].groupby("seed").dice.mean()
        empty_pred = int((data[(data.variant == v)].pred_ml == 0).sum())
        entry = {m: dict(mean=float(sub[m].mean()), sd=float(sub[m].std(ddof=1)), median=float(sub[m].median()),
                         q1=float(sub[m].quantile(.25)), q3=float(sub[m].quantile(.75))) for m in METRICS}
        entry["dice_by_seed"] = {int(s): float(x) for s, x in by_seed.items()}
        entry["dice_seed_sd"] = float(by_seed.std(ddof=1)) if len(by_seed) > 1 else None
        entry["empty_predictions"] = empty_pred
        summary["variants"][int(v)] = entry
        rows.append({"Вариант": NAMES[v],
                     "Dice": fmt(entry["dice"]["mean"], entry["dice"]["sd"]),
                     "IoU": fmt(entry["iou"]["mean"], entry["iou"]["sd"]),
                     "Precision": fmt(entry["precision"]["mean"], entry["precision"]["sd"]),
                     "Recall": fmt(entry["recall"]["mean"], entry["recall"]["sd"]),
                     "HD95, мм": fmt(entry["hd95"]["mean"], entry["hd95"]["sd"], 1),
                     "Dice по seed": " / ".join(f"{x:.3f}" for x in by_seed.values),
                     "SD Dice между seed": "" if entry["dice_seed_sd"] is None else f"{entry['dice_seed_sd']:.3f}"})
    table = pd.DataFrame(rows)
    table.to_csv(a.out / "table3.csv", index=False)

    tests = []
    base = per_patient[per_patient.variant == 1].set_index("subject_id")
    for metric in ("dice", "hd95"):
        raw = []
        for v in variants:
            if v == 1:
                continue
            cur = per_patient[per_patient.variant == v].set_index("subject_id")
            both = pd.concat([base[metric], cur[metric]], axis=1, keys=["b", "c"]).dropna()
            diff = (both.c - both.b).to_numpy()
            try:
                p = stats.wilcoxon(both.c, both.b).pvalue if np.any(diff != 0) else 1.0
            except ValueError:
                p = 1.0
            lo, hi = bootstrap_ci(diff)
            raw.append(dict(metric=metric, variant=int(v), n=len(diff), mean_diff=float(diff.mean()), ci_low=float(lo),
                            ci_high=float(hi), median_diff=float(np.median(diff)), p_raw=float(p),
                            wins=int((diff > 0).sum()), losses=int((diff < 0).sum())))
        adj = holm([r["p_raw"] for r in raw]) if raw else []
        for r, pa in zip(raw, adj):
            r["p_holm"] = float(pa)
            tests.append(r)
    pd.DataFrame(tests).to_csv(a.out / "wilcoxon_vs_baseline.csv", index=False)

    seed_dice = data.groupby(["variant", "seed"]).dice.mean().unstack(0)
    seed_hd = data.groupby(["variant", "seed"]).hd95.mean().unstack(0)
    seed_rows = []
    for v in variants:
        if v == 1:
            continue
        for name, table_ in (("dice", seed_dice), ("hd95", seed_hd)):
            d = (table_[v] - table_[1]).dropna()
            seed_rows.append(dict(metric=name, variant=int(v), n_seeds=len(d), mean_diff=float(d.mean()),
                                  sd_diff=float(d.std(ddof=1)) if len(d) > 1 else None,
                                  better_in=int((d > 0).sum() if name == "dice" else (d < 0).sum()),
                                  diffs=[round(float(x), 4) for x in d.values]))
    pd.DataFrame(seed_rows).to_csv(a.out / "seed_level_vs_baseline.csv", index=False)
    summary["seed_level"] = seed_rows
    seed_dice.to_csv(a.out / "dice_by_seed_and_variant.csv")
    summary["tests"] = tests

    pp = per_patient.copy()
    pp["volume_group"] = pd.cut(pp.gt_ml, VOLUME_BINS, labels=VOLUME_LABELS, right=False)
    sub_rows = []
    for g in VOLUME_LABELS:
        row = {"Объём очага": g, "N": int(pp[(pp.variant == variants[0]) & (pp.volume_group == g)].shape[0])}
        for v in variants:
            x = pp[(pp.variant == v) & (pp.volume_group == g)].dice
            row[SHORT[v]] = f"{x.mean():.3f}" if len(x) else "—"
        sub_rows.append(row)
    pd.DataFrame(sub_rows).to_csv(a.out / "dice_by_volume.csv", index=False)
    summary["dice_by_volume"] = sub_rows

    curves = {}
    for f in sorted(a.runs.glob("v*_s*/log.csv")):
        log = pd.read_csv(f)
        if "val_dice" in log:
            curves[f.parent.name] = log.dropna(subset=["val_dice"])[["epoch", "val_dice"]].values.tolist()
    summary["val_curves"] = curves
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(table.to_string(index=False))
    print(pd.DataFrame(tests).to_string(index=False))
    print(pd.DataFrame(seed_rows).to_string(index=False))
    print(pd.DataFrame(sub_rows).to_string(index=False))

    if a.figures:
        figures(a, per_patient, variants, seeds)


def figures(a, per_patient, variants, seeds):
    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=200)
    rng = np.random.RandomState(0)
    for k, v in enumerate(variants):
        y = per_patient[per_patient.variant == v].dice.to_numpy()
        ax.boxplot(y, positions=[k], widths=0.5, showfliers=False, medianprops=dict(color="black"),
                   boxprops=dict(color=COLORS[v]), whiskerprops=dict(color=COLORS[v]), capprops=dict(color=COLORS[v]))
        ax.scatter(k + rng.uniform(-0.15, 0.15, len(y)), y, s=14, color=COLORS[v], alpha=0.6, zorder=3)
        ax.scatter([k], [y.mean()], marker="D", color="white", edgecolor="black", zorder=4, s=36)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([NAMES[v].replace(" (", "\n(") for v in variants], fontsize=8)
    ax.set_ylabel("Dice по пациентам (среднее по seed)")
    ax.set_title("Распределение Dice на тестовой выборке (ромб — среднее)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(a.out / "fig4_dice_boxplot.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=200)
    for v in variants:
        sub = per_patient[per_patient.variant == v]
        ax.scatter(sub.gt_ml.clip(lower=0.05), sub.dice, s=16, alpha=0.5, color=COLORS[v], label=SHORT[v])
        edges = [0.05, 2, 5, 10, 20, 50, 500]
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            g = sub[(sub.gt_ml >= lo) & (sub.gt_ml < hi)]
            if len(g):
                xs.append(np.sqrt(lo * hi)); ys.append(g.dice.mean())
        ax.plot(xs, ys, color=COLORS[v], lw=1.6)
    ax.set_xscale("log")
    ax.set_xlabel("Объём эталонного очага, мл (логарифмическая шкала)")
    ax.set_ylabel("Dice")
    ax.set_title("Зависимость Dice от объёма очага (линии — средние по интервалам)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(a.out / "fig6_dice_vs_volume.png"); plt.close(fig)

    from train_eval import create_unet3d
    from monai.inferers import sliding_window_inference
    split = make_split()
    test = split[split.split == "test"]
    median_ml = test.lesion_ml.median()
    sid = test.iloc[(test.lesion_ml - median_ml).abs().argsort().iloc[0]].subject_id
    image, mask, spacing, brain = load_case(sid)
    z = int(np.argmax(mask.sum(axis=(0, 1))))
    seed = seeds[0]
    bx, by = np.where(brain.any(axis=2))
    x0, x1, y0, y1 = bx.min(), bx.max() + 1, by.min(), by.max() + 1
    crop = (slice(x0, x1), slice(y0, y1), z)
    fig, axes = plt.subplots(1, 1 + len(variants), figsize=(3 * (1 + len(variants)), 3.6), dpi=200)
    base_img = image[0][crop].T
    axes[0].imshow(base_img, cmap="gray", origin="lower")
    axes[0].contour(mask[crop].T, levels=[0.5], colors="red", linewidths=0.9)
    axes[0].set_title("DWI + эталон", fontsize=9)
    for ax, v in zip(axes[1:], variants):
        model = create_unet3d()
        model.load_state_dict(torch.load(a.runs / f"v{v}_s{seed}" / "best.pt", weights_only=True))
        model.eval()
        with torch.inference_mode():
            x = torch.from_numpy(preprocess(image, brain, v))[None]
            pred = torch.sigmoid(sliding_window_inference(x, PATCH, 1, model))[0, 0].numpy() > 0.5
        d = 2 * (pred & (mask > 0)).sum() / max(pred.sum() + (mask > 0).sum(), 1)
        ax.imshow(base_img, cmap="gray", origin="lower")
        ax.contour(mask[crop].T, levels=[0.5], colors="red", linewidths=0.9)
        ax.contour(pred[crop].T, levels=[0.5], colors="cyan", linewidths=0.9)
        ax.set_title(f"{SHORT[v]}, Dice = {d:.2f}", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{sid}, срез {z}: красный — эталон, голубой — предсказание (seed {seed})", fontsize=9)
    fig.tight_layout(); fig.savefig(a.out / "fig5_visual_comparison.png"); plt.close(fig)
    (a.out / "fig5_case.txt").write_text(f"{sid} slice {z} seed {seed}\n")


if __name__ == "__main__":
    main()
