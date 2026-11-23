"""Привести DWI, ADC, FLAIR и маску каждого случая ISLES 2022 к одной сетке (сетка DWI, ориентация RAS).

Повторяет геометрическую часть конвейера проекта (prepare_flair.py + Orientationd + ResampleToMatchd):
FLAIR передискретизируется на сетку DWI по аффинным матрицам заголовков (линейная интерполяция,
вне поля обзора 0), ADC и маска уже лежат на сетке DWI. Интенсивности не меняются: нормализация
выполняется позже и отличается для разных вариантов.
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform


def load(path):
    return nib.load(str(path))


def process(args):
    root, out_dir, sid = args
    out = out_dir / f"{sid}.npz"
    base = root / sid / "ses-0001"
    dwi = load(base / "dwi" / f"{sid}_ses-0001_dwi.nii.gz")
    adc = load(base / "dwi" / f"{sid}_ses-0001_adc.nii.gz")
    flair = load(base / "anat" / f"{sid}_ses-0001_FLAIR.nii.gz")
    msk = load(root / "derivatives" / sid / "ses-0001" / f"{sid}_ses-0001_msk.nii.gz")

    info = {"subject_id": sid, "dwi_shape": list(dwi.shape), "adc_same_grid": bool(
        adc.shape == dwi.shape and np.allclose(adc.affine, dwi.affine, atol=1e-3)),
        "mask_same_grid": bool(msk.shape == dwi.shape and np.allclose(msk.affine, dwi.affine, atol=1e-3)),
        "flair_shape": list(flair.shape)}

    dwi_a = np.asarray(dwi.dataobj, dtype=np.float32)
    adc_a = np.asarray(adc.dataobj, dtype=np.float32)
    msk_a = (np.asarray(msk.dataobj) > 0).astype(np.uint8)
    if not (info["adc_same_grid"] and info["mask_same_grid"]):
        raise RuntimeError(f"{sid}: ADC или маска не на сетке DWI")

    flair_a = np.asarray(flair.dataobj, dtype=np.float32)
    mat = np.linalg.inv(flair.affine) @ dwi.affine
    flair_on_dwi = affine_transform(flair_a, mat[:3, :3], offset=mat[:3, 3], output_shape=dwi.shape[:3],
                                    output=np.float32, order=1, mode="constant", cval=0.0, prefilter=False)
    del flair_a

    stack = nib.Nifti1Image(np.stack([dwi_a, adc_a, flair_on_dwi, msk_a.astype(np.float32)], axis=-1), dwi.affine)
    canon = nib.as_closest_canonical(stack)
    data = np.asarray(canon.dataobj, dtype=np.float32)
    zooms = np.asarray(canon.header.get_zooms()[:3], dtype=np.float32)
    image = np.moveaxis(data[..., :3], -1, 0).astype(np.float32)
    mask = (data[..., 3] > 0.5).astype(np.uint8)
    np.savez(out, image=image, mask=mask, spacing=zooms)
    info.update(shape=list(mask.shape), spacing=[float(z) for z in zooms], lesion_voxels=int(mask.sum()),
                lesion_ml=float(mask.sum() * zooms.prod() / 1000.0))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sids = sorted(p.name for p in a.root.glob("sub-strokecase*") if p.is_dir())
    jobs = [(a.root, a.out, s) for s in sids]
    infos = []
    with Pool(a.workers) as pool:
        for i, info in enumerate(pool.imap_unordered(process, jobs), 1):
            infos.append(info)
            if i % 10 == 0:
                print(f"{i}/{len(jobs)}", flush=True)
    infos.sort(key=lambda x: x["subject_id"])
    (a.out / "cases.json").write_text(json.dumps(infos, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
