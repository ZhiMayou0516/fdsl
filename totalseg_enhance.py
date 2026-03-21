#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mk_data_all_in_one_v5_totalseg_structure_aware.py

One-file runner for FDSL-style synthetic 3D segmentation dataset generation,
where ALL shapes come from TotalSegmentator (no BTCV, no PrimGeo shapes).

NEW in v5 (Organ-Structure-Aware composition)
---------------------------------------------
Adds relative spatial position priors + lightweight anatomical relation graph:

1) Each label_id gets an anchor distribution in normalized canvas coordinates,
   encouraging plausible relative locations (e.g., lungs in superior half, bladder inferior).
2) A relation graph defines constraints/preferences between labels:
   - contain: encourage B inside A (soft constraint)
   - avoid: discourage strong overlap between A and B (hard/soft)
   - contact: encourage boundary contact without heavy overlap
3) Placement becomes "sample candidates around anchor -> score by constraints -> choose best".

This turns the generator from pure copy-paste + occlusion threshold into
structure-aware composition with explicit, controllable anatomical priors.
"""

import csv
import json
import random
import argparse
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import nibabel as nib
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# ---- optional scipy ----
try:
    from scipy.ndimage import label as cc_label  # connected components (bank)
    from scipy import ndimage  # scaling + erosion + dilation
    _HAS_SCIPY = True
except Exception:
    cc_label = None
    ndimage = None
    _HAS_SCIPY = False


# ======================================================================================
# USER EDIT (paths)
# ======================================================================================
USER_PATHS = dict(
    TOTALSEG_ROOT=r"/mnt/data/tjq/data/totalseg",
    TOTALSEG_META_CSV=r"/mnt/data/tjq/data/totalseg/meta.csv",
    BANK_DIR=r"/mnt/data/tjq/data/totalseg/bank",
    OUT_DIR=r"/mnt/data/tjq/data/totalseg_syn_structure_aware",
)

# ======================================================================================
# SELECTED 32 CLASSES (Morphologically diverse)
# ======================================================================================
TARGET_32_CLASSES = [
    "spleen", "kidney_right", "kidney_left", "gallbladder", "liver",
    "stomach", "pancreas", "adrenal_gland_right", "adrenal_gland_left",
    "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
    "lung_middle_lobe_right", "lung_lower_lobe_right",
    "esophagus", "trachea", "thyroid_gland",
    "small_bowel", "duodenum", "colon", "urinary_bladder",
    "prostate",
    "sacrum", "vertebrae_L1", "vertebrae_C1",
    "rib_left_1", "rib_right_1",
    "scapula_left", "clavicle_left", "femur_left", "hip_left",
    "aorta"
]

# ======================================================================================
# Defaults
# ======================================================================================
DEFAULT_CFG = dict(
    # Synthetic split (ids are synthetic IDs, not real subjects)
    data_num=5000,
    val_rate=0.1,
    gen_test=False,

    # Volume settings
    space=96,
    num_per_image=20,
    no_rotate=True,
    noise_strength=0,
    occlusion_r=0.7,
    seed=1234,

    # Image style
    intensity=128,
    use_contour=True,
    contour_thickness=1,

    # Shape bank
    bank_margin=2,
    bank_min_voxels=300,
    bank_max_samples_per_case=2,
    bank_use_components=False,

    # Leakage control
    bank_split="train",
    bank_case_limit=10,

    # =========================
    # NEW: structure-aware placement
    # =========================
    structure_aware=True,              # enable new placement
    placement_candidates=40,           # candidates per organ instance (higher -> slower but better)
    anchor_jitter=0.12,                # std in normalized coords (0..1), around anchor mean
    hard_avoid_overlap=0.35,           # if avoid-pair overlap ratio > this => reject
    contact_dilate=1,                  # dilation radius to estimate contact
    contact_min_vox=20,                # minimum contacting voxels to count as "contact"
    contain_min_ratio=0.30,            # minimum ratio of child inside parent to count as "contained"
    relation_weight_contact=1.0,       # scoring weights
    relation_weight_contain=1.0,
    relation_weight_anchor=0.8,
    relation_weight_overlap=-1.0,      # penalize overlap generally (soft)
)

ROTATE_L_DEFAULT = ["stay", "side1", "side2", "side3", "side4", "upsidedown"]
NOISE_RANGE_DICT = {0: (0, 0), 1: (125, 200), 2: (300, 400)}


def _strip_niigz(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


# ======================================================================================
# meta.csv reading + split
# ======================================================================================
def read_totalseg_meta(meta_csv: Path) -> Dict[str, List[str]]:
    if not meta_csv.exists():
        raise FileNotFoundError(f"meta.csv not found: {meta_csv}")

    with open(meta_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        if reader.fieldnames is None:
            raise ValueError("meta.csv has no header / could not be parsed.")
        rows = list(reader)

    splits = {"train": [], "val": [], "test": []}
    for r in rows:
        sid = str(r.get("image_id", "")).strip()
        sp = str(r.get("split", "")).strip().lower()
        if not sid:
            continue
        if sp not in splits:
            sp = "train"
        splits[sp].append(sid)

    for k in splits:
        splits[k] = sorted(list(set(splits[k])))
    return splits


def select_bank_subjects(
    splits: Dict[str, List[str]],
    bank_split: str,
    bank_case_limit: int,
    seed: int,
) -> List[str]:
    bank_split = str(bank_split).lower()
    if bank_split == "train":
        pool = splits.get("train", [])
    elif bank_split == "val":
        pool = splits.get("val", [])
    elif bank_split == "trainval":
        pool = splits.get("train", []) + splits.get("val", [])
    elif bank_split == "all":
        pool = splits.get("train", []) + splits.get("val", []) + splits.get("test", [])
    else:
        raise ValueError(f"Unknown bank_split: {bank_split}")

    pool = sorted(list(dict.fromkeys(pool)))
    if not pool:
        raise RuntimeError(f"No subjects available for bank_split={bank_split}. Check meta.csv.")

    k = int(bank_case_limit)
    if k <= 0 or k >= len(pool):
        return pool

    rnd = random.Random(int(seed) + 99991)
    chosen = sorted(rnd.sample(pool, k))
    return chosen


# ======================================================================================
# Shape bank building
# ======================================================================================
def _bbox_crop(mask: np.ndarray, margin: int):
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None, None
    mn = coords.min(axis=0)
    mx = coords.max(axis=0) + 1
    mn = np.maximum(mn - margin, 0)
    mx = np.minimum(mx + margin, np.array(mask.shape))
    slc = tuple(slice(int(mn[i]), int(mx[i])) for i in range(3))
    crop = mask[slc]
    bbox = [int(mn[0]), int(mx[0]), int(mn[1]), int(mx[1]), int(mn[2]), int(mx[2])]
    return crop, bbox


def _list_structure_names(root: Path, subjects: List[str]) -> List[str]:
    names = set()
    for sid in subjects:
        seg_dir = root / sid / "segmentations"
        if not seg_dir.exists():
            continue
        for p in seg_dir.glob("*.nii.gz"):
            names.add(_strip_niigz(p.name))
        for p in seg_dir.glob("*.nii"):
            names.add(_strip_niigz(p.name))
    names = sorted(list(names))
    if not names:
        raise RuntimeError("No segmentation files found under <subject>/segmentations. Check TOTALSEG_ROOT.")
    return names


def build_totalseg_shape_bank(
    root_dir: Path,
    subjects: List[str],
    out_dir: Path,
    margin: int,
    min_voxels: int,
    max_samples_per_case: int,
    use_components: bool,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    struct_names = _list_structure_names(root_dir, subjects)
    labels_map = {"0": "background"}
    for i, name in enumerate(struct_names, start=1):
        labels_map[str(i)] = name

    index: Dict[int, List[str]] = {}
    for i in range(1, len(struct_names) + 1):
        (out_dir / f"label_{i:03d}").mkdir(parents=True, exist_ok=True)
        index[int(i)] = []

    for sid in tqdm(subjects, desc="[bank] scan subjects"):
        seg_dir = root_dir / sid / "segmentations"
        if not seg_dir.exists():
            print(f"[bank] WARN missing segmentations dir: {seg_dir}")
            continue

        for lab_id in range(1, len(struct_names) + 1):
            name = labels_map[str(lab_id)]
            fp_niigz = seg_dir / f"{name}.nii.gz"
            fp_nii = seg_dir / f"{name}.nii"
            fp = fp_niigz if fp_niigz.exists() else (fp_nii if fp_nii.exists() else None)
            if fp is None:
                continue

            arr = nib.load(str(fp)).get_fdata().astype(np.uint8)
            vox = int(arr.sum())
            if vox < int(min_voxels):
                continue

            crops = []
            if use_components and (cc_label is not None):
                cc, n = cc_label(arr > 0)
                sizes = []
                for k in range(1, n + 1):
                    sizes.append((int((cc == k).sum()), k))
                sizes.sort(reverse=True)
                for _, k in sizes[: int(max_samples_per_case)]:
                    comp = (cc == k).astype(np.uint8)
                    crop, bbox = _bbox_crop(comp, int(margin))
                    if crop is None:
                        continue
                    crops.append((crop, bbox))
            else:
                crop, bbox = _bbox_crop((arr > 0).astype(np.uint8), int(margin))
                if crop is not None:
                    crops.append((crop, bbox))

            for j, (crop, bbox) in enumerate(crops[: int(max_samples_per_case)]):
                out_path = out_dir / f"label_{lab_id:03d}" / f"{sid}_lab{lab_id:03d}_k{j}.npz"
                meta = {
                    "subject": sid,
                    "label_id": int(lab_id),
                    "label_name": str(labels_map[str(lab_id)]),
                    "bbox": bbox,
                    "voxels": int(crop.sum()),
                    "orig_shape": list(arr.shape),
                }
                np.savez_compressed(
                    str(out_path),
                    mask=crop.astype(np.uint8),
                    meta=np.array([meta], dtype=object),
                )
                index[int(lab_id)].append(str(out_path))

    (out_dir / "labels_map.json").write_text(json.dumps(labels_map, indent=2), encoding="utf-8")
    (out_dir / "bank_index.json").write_text(json.dumps({str(k): v for k, v in index.items()}, indent=2), encoding="utf-8")

    counts = {int(k): len(v) for k, v in index.items()}
    kept = sum(1 for k, c in counts.items() if c > 0)
    print("[bank] saved:", out_dir)
    print("[bank] labels total:", len(struct_names), "labels with samples:", kept)
    if use_components and cc_label is None:
        print("[bank] WARNING: use_components=True but scipy unavailable, used whole-mask crops.")
    return out_dir / "bank_index.json"


def load_totalseg_bank(bank_dir: Path) -> Tuple[Dict[int, List[Path]], Dict[int, str]]:
    labels_map_p = bank_dir / "labels_map.json"
    idx_p = bank_dir / "bank_index.json"
    if not labels_map_p.exists() or not idx_p.exists():
        raise FileNotFoundError(f"Missing labels_map.json or bank_index.json under: {bank_dir}")

    labels_map_raw = json.loads(labels_map_p.read_text(encoding="utf-8"))
    labels_map: Dict[int, str] = {}
    for k, v in labels_map_raw.items():
        try:
            labels_map[int(k)] = str(v)
        except Exception:
            pass

    idx_raw = json.loads(idx_p.read_text(encoding="utf-8"))
    bank: Dict[int, List[Path]] = {}
    for k, v in idx_raw.items():
        lab = int(k)
        files = []
        for p in v:
            pp = Path(p)
            if not pp.is_absolute():
                pp = bank_dir / pp
            files.append(pp)
        if files:
            bank[lab] = files

    if not bank:
        raise RuntimeError(f"Empty bank at: {bank_dir}")
    return bank, labels_map


# ======================================================================================
# Utils: seeds, rotate, morphology
# ======================================================================================
def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def random_rotate(box, target, rotate):
    if rotate == "stay":
        return box, target
    elif rotate == "side1":
        return np.rot90(box, 1, axes=(0, 2)), np.rot90(target, 1, axes=(0, 2))
    elif rotate == "side2":
        return np.rot90(box, -1, axes=(0, 2)), np.rot90(target, -1, axes=(0, 2))
    elif rotate == "side3":
        return np.rot90(box, 1, axes=(1, 2)), np.rot90(target, 1, axes=(1, 2))
    elif rotate == "side4":
        return np.rot90(box, -1, axes=(1, 2)), np.rot90(target, -1, axes=(1, 2))
    elif rotate == "upsidedown":
        return np.rot90(box, 2, axes=(0, 2)), np.rot90(target, 2, axes=(0, 2))
    else:
        raise ValueError(f"Unknown rotate mode: {rotate}")


def _erode_numpy_once(m: np.ndarray) -> np.ndarray:
    m = (m > 0).astype(np.uint8)
    sx, sy, sz = m.shape
    p = np.pad(m, 1, mode="constant", constant_values=0)
    out = p[1:1 + sx, 1:1 + sy, 1:1 + sz].copy()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                out = np.minimum(out, p[1 + dx:1 + dx + sx, 1 + dy:1 + dy + sy, 1 + dz:1 + dz + sz])
    return out


def binary_erosion(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    iters = max(1, int(iters))
    if _HAS_SCIPY:
        m = ndimage.binary_erosion(m, structure=np.ones((3, 3, 3), dtype=np.uint8), iterations=iters).astype(np.uint8)
        return m
    for _ in range(iters):
        m = _erode_numpy_once(m)
    return m


def binary_dilation(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    iters = max(1, int(iters))
    if _HAS_SCIPY:
        m = ndimage.binary_dilation(m, structure=np.ones((3, 3, 3), dtype=np.uint8), iterations=iters).astype(np.uint8)
        return m.astype(np.uint8)
    # numpy fallback: very rough dilation via max of neighbors (slow)
    sx, sy, sz = m.shape
    p = np.pad(m, 1, mode="constant", constant_values=0)
    out = np.zeros_like(m)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                out = np.maximum(out, p[1 + dx:1 + dx + sx, 1 + dy:1 + dy + sy, 1 + dz:1 + dz + sz])
    if iters > 1:
        for _ in range(iters - 1):
            out = binary_dilation(out, 1)
    return out.astype(np.uint8)


def mask_to_contour(mask: np.ndarray, thickness: int = 1) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    thickness = max(1, int(thickness))
    core = binary_erosion(m, iters=thickness)
    return (m & (1 - core)).astype(np.uint8)


def _zoom_binary(mask: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return mask
    if not _HAS_SCIPY:
        return mask
    z = ndimage.zoom(mask.astype(np.uint8), zoom=(scale, scale, scale), order=0)
    return (z > 0).astype(np.uint8)


# ======================================================================================
# NEW: structure-aware priors (anchor + relation graph)
# ======================================================================================
def _default_anchor_by_name(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (mu, sigma) in normalized [0,1]^3 for the organ center.
    Axis convention here is arbitrary but consistent within synthetic canvas:
      x: cranial->caudal (0 top, 1 bottom)
      y: left->right     (0 left, 1 right)
      z: posterior->anterior (0 back, 1 front)
    """
    n = name.lower()

    # broad defaults
    mu = np.array([0.55, 0.50, 0.50], dtype=np.float32)
    sg = np.array([0.22, 0.22, 0.22], dtype=np.float32)

    # superior thorax / neck
    if "thyroid" in n:
        mu = np.array([0.12, 0.50, 0.55]); sg = np.array([0.08, 0.18, 0.18])
    if "trachea" in n or "esophagus" in n:
        mu = np.array([0.25, 0.50, 0.55]); sg = np.array([0.15, 0.12, 0.18])

    if "lung" in n:
        mu = np.array([0.25, 0.50, 0.55]); sg = np.array([0.15, 0.20, 0.18])

    # upper abdomen
    if n in ("liver", "gallbladder"):
        mu = np.array([0.48, 0.63, 0.55]); sg = np.array([0.18, 0.18, 0.18])
    if n == "spleen":
        mu = np.array([0.48, 0.35, 0.55]); sg = np.array([0.18, 0.18, 0.18])
    if n == "stomach":
        mu = np.array([0.50, 0.42, 0.58]); sg = np.array([0.18, 0.18, 0.18])
    if n == "pancreas":
        mu = np.array([0.52, 0.50, 0.55]); sg = np.array([0.14, 0.18, 0.16])
    if "adrenal_gland" in n:
        mu = np.array([0.48, 0.50, 0.52]); sg = np.array([0.16, 0.18, 0.18])

    if n == "aorta":
        mu = np.array([0.45, 0.50, 0.45]); sg = np.array([0.20, 0.10, 0.18])

    if "kidney_right" in n:
        mu = np.array([0.58, 0.62, 0.52]); sg = np.array([0.16, 0.16, 0.16])
    if "kidney_left" in n:
        mu = np.array([0.58, 0.38, 0.52]); sg = np.array([0.16, 0.16, 0.16])

    # bowel / pelvis
    if "small_bowel" in n or "colon" in n or "duodenum" in n:
        mu = np.array([0.68, 0.50, 0.55]); sg = np.array([0.22, 0.22, 0.18])
    if "urinary_bladder" in n:
        mu = np.array([0.86, 0.50, 0.60]); sg = np.array([0.10, 0.18, 0.16])
    if "prostate" in n:
        mu = np.array([0.88, 0.50, 0.58]); sg = np.array([0.08, 0.14, 0.12])

    # bones: posterior-ish, distributed
    if "vertebrae" in n or "sacrum" in n:
        mu = np.array([0.60, 0.50, 0.30]); sg = np.array([0.22, 0.14, 0.14])
    if "rib" in n or "scapula" in n or "clavicle" in n:
        mu = np.array([0.25, 0.50, 0.35]); sg = np.array([0.18, 0.25, 0.18])
    if "femur" in n or "hip" in n:
        mu = np.array([0.92, 0.50, 0.40]); sg = np.array([0.10, 0.22, 0.16])

    return mu.astype(np.float32), sg.astype(np.float32)


def build_anchor_priors(labels_map: Dict[int, str]) -> Dict[int, Dict[str, Any]]:
    priors = {}
    for lid, name in labels_map.items():
        if int(lid) == 0:
            continue
        mu, sg = _default_anchor_by_name(str(name))
        priors[int(lid)] = {"mu": mu, "sigma": sg}
    return priors


def build_relation_graph(labels_map: Dict[int, str]) -> Dict[str, List[Tuple[int, int]]]:
    """
    Return relation graph as pairs of label_ids.
    Keep it minimal + robust (won't crash if some classes missing).
    """
    name_to_id = {str(v): int(k) for k, v in labels_map.items() if str(k) != "0"}

    def _has(n): return n in name_to_id
    def _id(n): return name_to_id[n]

    contain = []
    contact = []
    avoid = []

    # contain: lung contains trachea-ish (soft), stomach contains duodenum? (too strong, skip)
    if _has("trachea"):
        for ln in ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"]:
            if _has(ln):
                contain.append((_id(ln), _id("trachea")))  # parent, child

    # contact: aorta tends to be adjacent to many abdominal organs (soft)
    if _has("aorta"):
        for on in ["liver", "spleen", "pancreas", "kidney_left", "kidney_right", "stomach"]:
            if _has(on):
                contact.append((_id(on), _id("aorta")))

    # contact: bladder / prostate
    if _has("urinary_bladder") and _has("prostate"):
        contact.append((_id("urinary_bladder"), _id("prostate")))

    # avoid: bones should not heavily overlap with bowel/bladder (hard-ish)
    bone_like = [n for n in ["sacrum", "vertebrae_L1", "vertebrae_C1", "rib_left_1", "rib_right_1", "scapula_left", "clavicle_left", "femur_left", "hip_left"] if _has(n)]
    soft_like = [n for n in ["small_bowel", "colon", "urinary_bladder", "stomach"] if _has(n)]
    for b in bone_like:
        for s in soft_like:
            avoid.append((_id(b), _id(s)))

    return {"contain": contain, "contact": contact, "avoid": avoid}


# ======================================================================================
# Augment + sample shapes
# ======================================================================================
def augment_shape_mask(mask: np.ndarray, cfg: dict, SPACE) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)

    if random.random() < 0.5:
        m = m[::-1, :, :]
    if random.random() < 0.5:
        m = m[:, ::-1, :]
    if random.random() < 0.5:
        m = m[:, :, ::-1]

    axes_choices = [(0, 1), (0, 2), (1, 2)]
    ax = random.choice(axes_choices)
    k = random.randint(0, 3)
    if k:
        m = np.rot90(m, k, axes=ax)

    lo, hi = 0.85, 1.25
    scale = random.uniform(float(lo), float(hi))
    m = _zoom_binary(m, scale)

    w, h, d = m.shape
    if w > SPACE[0] or h > SPACE[1] or d > SPACE[2]:
        shrink = min(SPACE[0] / w, SPACE[1] / h, SPACE[2] / d) * 0.95
        m = _zoom_binary(m, shrink)

    return m.astype(np.uint8)


def sample_totalseg_shape(
    label_id: int,
    bank: Dict[int, List[Path]],
    cfg: dict,
    SPACE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = bank.get(int(label_id), [])
    if not files:
        raise RuntimeError(f"No shapes for label {label_id} in bank.")
    npz_path = random.choice(files)
    obj = np.load(str(npz_path), allow_pickle=True)
    mask = obj["mask"].astype(np.uint8)

    mask = augment_shape_mask(mask, cfg, SPACE)
    filled_mask = (mask > 0).astype(np.uint8)

    if bool(cfg["use_contour"]):
        contour_mask = mask_to_contour(filled_mask, thickness=int(cfg["contour_thickness"]))
    else:
        contour_mask = filled_mask

    box = (contour_mask * int(cfg["intensity"])).astype(np.int16)
    target_box = (filled_mask * int(label_id)).astype(np.int16)
    return box, target_box, filled_mask


# ======================================================================================
# NEW: structure-aware placement (candidate sampling + scoring)
# ======================================================================================
def _place_mask_into_canvas(mask: np.ndarray, x: int, y: int, z: int, SPACE: Tuple[int, int, int]) -> np.ndarray:
    w, h, d = mask.shape
    canvas = np.zeros(SPACE, dtype=np.uint8)
    canvas[x:x + w, y:y + h, z:z + d] = mask.astype(np.uint8)
    return canvas


def _sample_anchor_center(label_id: int, anchor_priors: Dict[int, Dict[str, Any]], cfg: dict) -> np.ndarray:
    pr = anchor_priors.get(int(label_id), None)
    if pr is None:
        mu = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        sg = np.array([0.25, 0.25, 0.25], dtype=np.float32)
    else:
        mu = pr["mu"].astype(np.float32)
        sg = pr["sigma"].astype(np.float32)

    # jittered sample
    jitter = float(cfg["anchor_jitter"])
    c = np.random.normal(loc=mu, scale=sg * 0.35 + jitter, size=(3,)).astype(np.float32)
    c = np.clip(c, 0.05, 0.95)
    return c


def _compute_overlap_ratio(new_occ: np.ndarray, occ: np.ndarray) -> float:
    inter = int(np.sum((new_occ > 0) & (occ > 0)))
    denom = int(np.sum(new_occ > 0))
    return inter / max(denom, 1)


def _contact_voxels(a: np.ndarray, b: np.ndarray, dilate: int) -> int:
    if dilate <= 0:
        return int(np.sum((a > 0) & (b > 0)))
    a_d = binary_dilation(a, iters=dilate)
    return int(np.sum((a_d > 0) & (b > 0)))


def _contain_ratio(child: np.ndarray, parent: np.ndarray) -> float:
    inside = int(np.sum((child > 0) & (parent > 0)))
    denom = int(np.sum(child > 0))
    return inside / max(denom, 1)


def _score_candidate(
    label_id: int,
    new_occ: np.ndarray,
    occ: np.ndarray,
    placed: List[Dict[str, Any]],
    anchor_center_norm: np.ndarray,
    new_center_norm: np.ndarray,
    relations: Dict[str, List[Tuple[int, int]]],
    cfg: dict
) -> Tuple[float, bool]:
    """
    Returns (score, is_valid). Hard rejects for severe avoid overlap violations.
    """
    # base overlap penalty
    overlap = _compute_overlap_ratio(new_occ, occ)

    # anchor penalty (distance from sampled anchor center)
    anchor_dist = float(np.linalg.norm(new_center_norm - anchor_center_norm))

    score = 0.0
    score += float(cfg["relation_weight_overlap"]) * overlap
    score += float(cfg["relation_weight_anchor"]) * (-anchor_dist)

    # build quick dict of already-placed by label_id (might be multiple instances per label; keep list)
    by_lid: Dict[int, List[np.ndarray]] = {}
    for it in placed:
        by_lid.setdefault(int(it["label_id"]), []).append(it["occ"])

    # hard avoid constraints
    hard_thr = float(cfg["hard_avoid_overlap"])
    for (a, b) in relations.get("avoid", []):
        if int(label_id) == int(a) and int(b) in by_lid:
            for occ_b in by_lid[int(b)]:
                ov = _compute_overlap_ratio(new_occ, occ_b)
                if ov > hard_thr:
                    return -1e9, False
        if int(label_id) == int(b) and int(a) in by_lid:
            for occ_a in by_lid[int(a)]:
                ov = _compute_overlap_ratio(new_occ, occ_a)
                if ov > hard_thr:
                    return -1e9, False

    # contact rewards
    dil = int(cfg["contact_dilate"])
    cmin = int(cfg["contact_min_vox"])
    w_contact = float(cfg["relation_weight_contact"])
    for (a, b) in relations.get("contact", []):
        if int(label_id) == int(a) and int(b) in by_lid:
            best = 0
            for occ_b in by_lid[int(b)]:
                best = max(best, _contact_voxels(new_occ, occ_b, dil))
            if best >= cmin:
                score += w_contact
        if int(label_id) == int(b) and int(a) in by_lid:
            best = 0
            for occ_a in by_lid[int(a)]:
                best = max(best, _contact_voxels(new_occ, occ_a, dil))
            if best >= cmin:
                score += w_contact

    # contain rewards (soft)
    w_contain = float(cfg["relation_weight_contain"])
    cthr = float(cfg["contain_min_ratio"])
    for (parent, child) in relations.get("contain", []):
        if int(label_id) == int(child) and int(parent) in by_lid:
            best = 0.0
            for occ_p in by_lid[int(parent)]:
                best = max(best, _contain_ratio(new_occ, occ_p))
            if best >= cthr:
                score += w_contain
        if int(label_id) == int(parent) and int(child) in by_lid:
            # parent added after child: reward if child inside new parent
            best = 0.0
            for occ_c in by_lid[int(child)]:
                best = max(best, _contain_ratio(occ_c, new_occ))
            if best >= cthr:
                score += w_contain

    return score, True


def _choose_coordinate_structure_aware(
    label_id: int,
    mask_shape: Tuple[int, int, int],
    filled_tensor: np.ndarray,
    SPACE: Tuple[int, int, int],
    anchor_priors: Dict[int, Dict[str, Any]],
    relations: Dict[str, List[Tuple[int, int]]],
    placed: List[Dict[str, Any]],
    cfg: dict
) -> Optional[Tuple[int, int, int, np.ndarray]]:
    """
    Returns (x,y,z,new_occ) or None if no valid candidate.
    """
    w, h, d = mask_shape
    if w > SPACE[0] or h > SPACE[1] or d > SPACE[2]:
        return None

    anchor_center = _sample_anchor_center(label_id, anchor_priors, cfg)

    best = None
    best_score = -1e18

    C = int(cfg["placement_candidates"])
    for _ in range(C):
        # sample target center around anchor
        c = np.random.normal(loc=anchor_center, scale=float(cfg["anchor_jitter"]), size=(3,)).astype(np.float32)
        c = np.clip(c, 0.05, 0.95)

        # convert to voxel coordinate for the *center* then compute corner
        cx = int(round(float(c[0]) * (SPACE[0] - 1)))
        cy = int(round(float(c[1]) * (SPACE[1] - 1)))
        cz = int(round(float(c[2]) * (SPACE[2] - 1)))

        x = int(np.clip(cx - w // 2, 0, SPACE[0] - w))
        y = int(np.clip(cy - h // 2, 0, SPACE[1] - h))
        z = int(np.clip(cz - d // 2, 0, SPACE[2] - d))

        # quick occlusion global constraint with current occupancy
        # build occ for this candidate (binary)
        # NOTE: we don't know mask content here; caller will pass filled_mask later via new_occ.
        # Here we only handle coordinate; occ built outside.
        # We'll temporarily accept coordinate; scoring happens after new_occ built.

        # store candidate
        # we'll compute score in caller after new_occ available
        # But we can at least approximate center for anchor distance:
        new_center = np.array([(x + w / 2) / SPACE[0], (y + h / 2) / SPACE[1], (z + d / 2) / SPACE[2]], dtype=np.float32)

        # placeholder new_occ will be filled in caller; return coordinate + new_center norm
        # We'll compute score in caller so we can use exact occupancy.
        # Instead, we just keep coordinate, new_center for later.
        if best is None:
            best = (x, y, z, new_center)
            best_score = -1e17  # will be updated later

    if best is None:
        return None

    # We'll return coordinate and the anchor_center for later scoring in caller
    # encode anchor_center in new_center slot? better return separately
    return best[0], best[1], best[2], anchor_center


# ======================================================================================
# FDSL generation: make one sample
# ======================================================================================
def make_data_one(cfg: dict,
                  bank: Dict[int, List[Path]],
                  label_ids: List[int],
                  labels_map: Dict[int, str],
                  SPACE: Tuple[int, int, int],
                  anchor_priors: Dict[int, Dict[str, Any]],
                  relations: Dict[str, List[Tuple[int, int]]]):
    boxes = []
    targets = []
    filleds = []
    volumes = []
    lab_ids = []

    for _ in range(int(cfg["num_per_image"])):
        lab_id = random.choice(label_ids)
        box, target_box, filled_mask = sample_totalseg_shape(lab_id, bank, cfg, SPACE)
        boxes.append(box)
        targets.append(target_box)
        filleds.append(filled_mask)
        volumes.append(int(np.sum(target_box > 0)))
        lab_ids.append(int(lab_id))

    filled_tensor = np.zeros(SPACE, dtype=np.uint8)   # occupancy of current label map
    space = np.zeros(SPACE, dtype=np.int16)           # image canvas
    space_target = np.zeros(SPACE, dtype=np.int16)    # label canvas

    lo, hi = NOISE_RANGE_DICT.get(int(cfg["noise_strength"]), (0, 0))
    if int(cfg["noise_strength"]) > 0:
        noise_level = random.randint(int(lo), int(hi))
        noise = np.random.randint(0, noise_level, SPACE, dtype=np.int16)
    else:
        noise = np.zeros(SPACE, dtype=np.int16)

    rotate_list = ["stay"] if bool(cfg["no_rotate"]) else ROTATE_L_DEFAULT[:]

    # place big objects first
    sorted_pairs = sorted([(volumes[i], i) for i in range(len(volumes))], reverse=True)

    placed: List[Dict[str, Any]] = []

    for vol, idx in sorted_pairs:
        box = boxes[idx]
        target = targets[idx]
        filled_mask = filleds[idx]
        label_id = lab_ids[idx]

        # random rotate both
        rotate = random.choice(rotate_list)
        box_r, target_r = random_rotate(box, target, rotate)
        filled_r, _ = random_rotate(filled_mask, filled_mask, rotate)

        w, h, d = filled_r.shape
        if w > SPACE[0] or h > SPACE[1] or d > SPACE[2]:
            continue

        # ============== choose coordinate ==============
        if bool(cfg["structure_aware"]):
            picked = _choose_coordinate_structure_aware(
                label_id=label_id,
                mask_shape=filled_r.shape,
                filled_tensor=filled_tensor,
                SPACE=SPACE,
                anchor_priors=anchor_priors,
                relations=relations,
                placed=placed,
                cfg=cfg
            )
            if picked is None:
                continue
            x, y, z, anchor_center = picked

            # build new occupancy
            new_occ = _place_mask_into_canvas((filled_r > 0).astype(np.uint8), x, y, z, SPACE)

            # global occlusion constraint (your original knob)
            overlap_ratio = _compute_overlap_ratio(new_occ, filled_tensor)
            if overlap_ratio > float(cfg["occlusion_r"]):
                # try a few random retries around anchor
                ok = False
                for _ in range(10):
                    picked2 = _choose_coordinate_structure_aware(
                        label_id=label_id,
                        mask_shape=filled_r.shape,
                        filled_tensor=filled_tensor,
                        SPACE=SPACE,
                        anchor_priors=anchor_priors,
                        relations=relations,
                        placed=placed,
                        cfg=cfg
                    )
                    if picked2 is None:
                        break
                    x, y, z, anchor_center = picked2
                    new_occ = _place_mask_into_canvas((filled_r > 0).astype(np.uint8), x, y, z, SPACE)
                    overlap_ratio = _compute_overlap_ratio(new_occ, filled_tensor)
                    if overlap_ratio <= float(cfg["occlusion_r"]):
                        ok = True
                        break
                if not ok:
                    continue

            # score candidate with relations
            new_center_norm = np.array([(x + w / 2) / SPACE[0], (y + h / 2) / SPACE[1], (z + d / 2) / SPACE[2]], dtype=np.float32)
            score, valid = _score_candidate(
                label_id=label_id,
                new_occ=new_occ,
                occ=filled_tensor,
                placed=placed,
                anchor_center_norm=anchor_center,
                new_center_norm=new_center_norm,
                relations=relations,
                cfg=cfg
            )
            if not valid:
                continue
        else:
            # fallback to original random placement with occlusion threshold via rejection
            tries = 0
            while True:
                x = random.randint(0, SPACE[0] - w)
                y = random.randint(0, SPACE[1] - h)
                z = random.randint(0, SPACE[2] - d)
                new_occ = _place_mask_into_canvas((filled_r > 0).astype(np.uint8), x, y, z, SPACE)
                overlap_ratio = _compute_overlap_ratio(new_occ, filled_tensor)
                if overlap_ratio <= float(cfg["occlusion_r"]):
                    break
                tries += 1
                if tries > 100:
                    new_occ = None
                    break
            if new_occ is None:
                continue
            score = 0.0

        # ============== commit placement with overwrite label semantics ==============
        # update image canvas: add contour-intensity where box_r > 0 in the placed region
        # update label canvas: overwrite where filled_r > 0 with label_id
        tmp_img = np.zeros(SPACE, dtype=np.int16)
        tmp_lab = np.zeros(SPACE, dtype=np.int16)

        tmp_img[x:x + w, y:y + h, z:z + d] = box_r.astype(np.int16)
        tmp_lab[x:x + w, y:y + h, z:z + d] = target_r.astype(np.int16)

        # overwrite labels where tmp_lab > 0
        space_target = np.where(tmp_lab > 0, tmp_lab, space_target).astype(np.int16)

        # image: we will later binarize to constant intensity, so here just accumulate
        space = (space + tmp_img).astype(np.int16)

        # occupancy = where labels exist
        filled_tensor = (space_target > 0).astype(np.uint8)

        placed.append({
            "label_id": int(label_id),
            "occ": (tmp_lab > 0).astype(np.uint8),  # exact new occ for this placed instance
            "score": float(score),
            "name": str(labels_map.get(int(label_id), "unknown")),
        })

    # render intensity: constant foreground + noise
    space = np.where(space > 0, int(cfg["intensity"]), 0).astype(np.int16)
    space = (space + noise).astype(np.int16)
    return space, space_target


# ======================================================================================
# Worker globals
# ======================================================================================
_G_CFG = None
_G_BANK = None
_G_LABEL_IDS = None
_G_SPACE = None
_G_SAVE = None
_G_LABELS_MAP = None
_G_ANCHOR_PRIORS = None
_G_RELATIONS = None


def _init_worker(cfg, bank, label_ids, SPACE, save_dict, labels_map, anchor_priors, relations):
    global _G_CFG, _G_BANK, _G_LABEL_IDS, _G_SPACE, _G_SAVE, _G_LABELS_MAP, _G_ANCHOR_PRIORS, _G_RELATIONS
    _G_CFG = cfg
    _G_BANK = bank
    _G_LABEL_IDS = label_ids
    _G_SPACE = SPACE
    _G_SAVE = save_dict
    _G_LABELS_MAP = labels_map
    _G_ANCHOR_PRIORS = anchor_priors
    _G_RELATIONS = relations


def _save_case(case_id: int, split: str, img: np.ndarray, lab: Optional[np.ndarray]):
    if split in ("train", "val"):
        img_p = Path(_G_SAVE["imagesTr"]) / f"img{case_id:04d}.nii.gz"
        lab_p = Path(_G_SAVE["labelsTr"]) / f"label{case_id:04d}.nii.gz"
        nib.save(nib.Nifti1Image(img.astype(np.float32), np.eye(4)), str(img_p))
        if lab is None:
            raise RuntimeError("train/val requires label, got None")
        nib.save(nib.Nifti1Image(lab.astype(np.int16), np.eye(4)), str(lab_p))
    elif split == "test":
        img_p = Path(_G_SAVE["imagesTs"]) / f"img{case_id:04d}.nii.gz"
        nib.save(nib.Nifti1Image(img.astype(np.float32), np.eye(4)), str(img_p))
    else:
        raise ValueError(f"Unknown split: {split}")


def _worker_generate(task: Tuple[int, str]) -> int:
    case_id, split = task
    seed = int(_G_CFG["seed"]) + int(case_id)
    set_seeds(seed)
    img, lab = make_data_one(
        _G_CFG, _G_BANK, _G_LABEL_IDS, _G_LABELS_MAP, _G_SPACE, _G_ANCHOR_PRIORS, _G_RELATIONS
    )
    if split == "test":
        _save_case(case_id, split, img, None)
    else:
        _save_case(case_id, split, img, lab)
    return case_id


# ======================================================================================
# dataset.json (nnUNet style)
# ======================================================================================
def write_dataset_json(out_dir: Path, labels_map: Dict[int, str], train_ids: List[int], val_ids: List[int], test_ids: List[int]):
    labels = {str(k): str(v) for k, v in sorted(labels_map.items(), key=lambda x: int(x[0]) if isinstance(x[0], str) and x[0].isdigit() else int(x[0]))}
    js = {
        "name": "totalseg_fdsl_only_synth_structure_aware",
        "description": "FDSL synthetic dataset; shapes sampled only from TotalSegmentator shapebank; structure-aware placement enabled",
        "tensorImageSize": "3D",
        "reference": "TotalSegmentator (meta.csv split used for bank restriction)",
        "licence": "research",
        "release": "v201-small",
        "modality": {"0": "CT"},
        "labels": labels,
        "numTraining": len(train_ids),
        "numTest": len(test_ids),
        "training": [{"image": f"imagesTr/img{cid:04d}.nii.gz", "label": f"labelsTr/label{cid:04d}.nii.gz"} for cid in train_ids],
        "validation": [{"image": f"imagesTr/img{cid:04d}.nii.gz", "label": f"labelsTr/label{cid:04d}.nii.gz"} for cid in val_ids],
        "test": [f"imagesTs/img{cid:04d}.nii.gz" for cid in test_ids],
    }
    (out_dir / "dataset.json").write_text(json.dumps(js, indent=4), encoding="utf-8")
    print("[json] wrote:", out_dir / "dataset.json")


# ======================================================================================
# CLI + main
# ======================================================================================
def _parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="FDSL synthetic dataset generator using TotalSegmentator shapes only (structure-aware placement)."
    )

    p.add_argument("--totalseg_root", type=str, default=USER_PATHS["TOTALSEG_ROOT"])
    p.add_argument("--meta_csv", type=str, default=USER_PATHS["TOTALSEG_META_CSV"])
    p.add_argument("--bank_dir", type=str, default=USER_PATHS["BANK_DIR"])
    p.add_argument("--out_dir", type=str, default=USER_PATHS["OUT_DIR"])

    # bank control
    p.add_argument("--bank_split", type=str, default=DEFAULT_CFG["bank_split"], choices=["train", "val", "trainval", "all"])
    p.add_argument("--bank_case_limit", type=int, default=DEFAULT_CFG["bank_case_limit"])
    p.add_argument("--rebuild_bank", action="store_true")
    p.add_argument("--skip_build_bank", action="store_true")

    # generation
    p.add_argument("--data_num", type=int, default=DEFAULT_CFG["data_num"])
    p.add_argument("--val_rate", type=float, default=DEFAULT_CFG["val_rate"])
    p.add_argument("--gen_test", action="store_true", default=DEFAULT_CFG["gen_test"])
    p.add_argument("--space", type=int, default=DEFAULT_CFG["space"])
    p.add_argument("--num_per_image", type=int, default=DEFAULT_CFG["num_per_image"])
    p.add_argument("--pool_workers", type=int, default=10)
    p.add_argument("--seed", type=int, default=DEFAULT_CFG["seed"])
    p.add_argument("--noise_strength", type=int, default=DEFAULT_CFG["noise_strength"], choices=[0, 1, 2])
    p.add_argument("--occlusion_r", type=float, default=DEFAULT_CFG["occlusion_r"])

    # image style
    p.add_argument("--intensity", type=int, default=DEFAULT_CFG["intensity"])
    p.add_argument("--use_contour", action="store_true", default=DEFAULT_CFG["use_contour"])
    p.add_argument("--contour_thickness", type=int, default=DEFAULT_CFG["contour_thickness"])

    # bank crop params
    p.add_argument("--bank_margin", type=int, default=DEFAULT_CFG["bank_margin"])
    p.add_argument("--bank_min_voxels", type=int, default=DEFAULT_CFG["bank_min_voxels"])
    p.add_argument("--bank_max_samples_per_case", type=int, default=DEFAULT_CFG["bank_max_samples_per_case"])
    p.add_argument("--bank_use_components", action="store_true", default=DEFAULT_CFG["bank_use_components"])

    # structure-aware placement
    p.add_argument("--structure_aware", action="store_true", default=DEFAULT_CFG["structure_aware"])
    p.add_argument("--placement_candidates", type=int, default=DEFAULT_CFG["placement_candidates"])
    p.add_argument("--anchor_jitter", type=float, default=DEFAULT_CFG["anchor_jitter"])
    p.add_argument("--hard_avoid_overlap", type=float, default=DEFAULT_CFG["hard_avoid_overlap"])
    p.add_argument("--contact_dilate", type=int, default=DEFAULT_CFG["contact_dilate"])
    p.add_argument("--contact_min_vox", type=int, default=DEFAULT_CFG["contact_min_vox"])
    p.add_argument("--contain_min_ratio", type=float, default=DEFAULT_CFG["contain_min_ratio"])

    return p.parse_args()


def main():
    args = _parse_args()

    cfg = deepcopy(DEFAULT_CFG)
    cfg["data_num"] = int(args.data_num)
    cfg["val_rate"] = float(args.val_rate)
    cfg["gen_test"] = bool(args.gen_test)
    cfg["space"] = int(args.space)
    cfg["num_per_image"] = int(args.num_per_image)
    cfg["seed"] = int(args.seed)
    cfg["noise_strength"] = int(args.noise_strength)
    cfg["occlusion_r"] = float(args.occlusion_r)
    cfg["intensity"] = int(args.intensity)
    cfg["use_contour"] = bool(args.use_contour)
    cfg["contour_thickness"] = int(args.contour_thickness)
    cfg["bank_margin"] = int(args.bank_margin)
    cfg["bank_min_voxels"] = int(args.bank_min_voxels)
    cfg["bank_max_samples_per_case"] = int(args.bank_max_samples_per_case)
    cfg["bank_use_components"] = bool(args.bank_use_components)
    cfg["bank_split"] = str(args.bank_split)
    cfg["bank_case_limit"] = int(args.bank_case_limit)

    cfg["structure_aware"] = bool(args.structure_aware)
    cfg["placement_candidates"] = int(args.placement_candidates)
    cfg["anchor_jitter"] = float(args.anchor_jitter)
    cfg["hard_avoid_overlap"] = float(args.hard_avoid_overlap)
    cfg["contact_dilate"] = int(args.contact_dilate)
    cfg["contact_min_vox"] = int(args.contact_min_vox)
    cfg["contain_min_ratio"] = float(args.contain_min_ratio)

    root_dir = Path(args.totalseg_root)
    meta_csv = Path(args.meta_csv)
    bank_dir = Path(args.bank_dir)
    out_dir = Path(args.out_dir)

    splits = read_totalseg_meta(meta_csv)
    bank_subjects = select_bank_subjects(splits, cfg["bank_split"], cfg["bank_case_limit"], cfg["seed"])

    out_dir.mkdir(parents=True, exist_ok=True)
    split_report = {
        "totalseg_meta_csv": str(meta_csv),
        "totalseg_root": str(root_dir),
        "meta_split": splits,
        "bank_split": cfg["bank_split"],
        "bank_case_limit": cfg["bank_case_limit"],
        "bank_subjects": bank_subjects,
        "note": "Shape bank is built ONLY from bank_subjects to reduce leakage risk.",
    }

    # rebuild bank if requested
    if bool(args.rebuild_bank) and bank_dir.exists():
        for fp in [bank_dir / "labels_map.json", bank_dir / "bank_index.json"]:
            if fp.exists():
                try:
                    fp.unlink()
                except Exception:
                    pass
        for sub in bank_dir.glob("label_*"):
            if sub.is_dir():
                for f in sub.glob("*"):
                    try:
                        f.unlink()
                    except Exception:
                        pass
                try:
                    sub.rmdir()
                except Exception:
                    pass

    if not args.skip_build_bank:
        need = not (bank_dir / "labels_map.json").exists() or not (bank_dir / "bank_index.json").exists()
        if need:
            print("[main] building shapebank from TotalSegmentator...")
            build_totalseg_shape_bank(
                root_dir=root_dir,
                subjects=bank_subjects,
                out_dir=bank_dir,
                margin=cfg["bank_margin"],
                min_voxels=cfg["bank_min_voxels"],
                max_samples_per_case=cfg["bank_max_samples_per_case"],
                use_components=cfg["bank_use_components"],
            )
        else:
            print("[main] using existing bank:", bank_dir)
            print("[main] NOTE: if you changed --bank_split/--bank_case_limit, run with --rebuild_bank to guarantee restriction.")
    else:
        print("[main] skip_build_bank=True -> assume bank exists:", bank_dir)

    # load full bank
    full_bank, full_labels_map = load_totalseg_bank(bank_dir)

    # filter & remap to 32
    print(f"[main] Filtering bank to keep only {len(TARGET_32_CLASSES)} classes...")
    name_to_old_id = {v: k for k, v in full_labels_map.items() if int(k) != 0}

    final_bank = {}
    final_labels_map = {0: "background"}

    new_id = 1
    for target_name in TARGET_32_CLASSES:
        if target_name not in name_to_old_id:
            print(f"  [WARN] Target class '{target_name}' not found in built bank. Skipping.")
            continue
        old_id = int(name_to_old_id[target_name])
        if old_id not in full_bank or len(full_bank[old_id]) == 0:
            print(f"  [WARN] Target class '{target_name}' (old_id={old_id}) has no samples. Skipping.")
            continue
        final_bank[new_id] = full_bank[old_id]
        final_labels_map[new_id] = target_name
        new_id += 1

    if len(final_bank) == 0:
        raise RuntimeError("Filtered bank is empty! Check TARGET_32_CLASSES names vs TotalSegmentator names.")

    bank = final_bank
    labels_map = final_labels_map

    label_ids = sorted([lid for lid in bank.keys() if int(lid) != 0])
    split_report["bank_labels_with_samples"] = len(label_ids)
    split_report["bank_labels_total"] = len(labels_map) - 1

    # NEW: build priors + relations based on kept labels
    anchor_priors = build_anchor_priors(labels_map)
    relations = build_relation_graph(labels_map)
    split_report["structure_aware"] = {
        "enabled": bool(cfg["structure_aware"]),
        "relations": relations,
        "anchor_priors": {str(k): {"mu": v["mu"].tolist(), "sigma": v["sigma"].tolist()} for k, v in anchor_priors.items()},
    }

    # nnUNet style output folders
    imagesTr = out_dir / "imagesTr"
    labelsTr = out_dir / "labelsTr"
    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)
    imagesTs = out_dir / "imagesTs"
    if bool(cfg["gen_test"]):
        imagesTs.mkdir(parents=True, exist_ok=True)

    # synthetic split
    n = int(cfg["data_num"])
    if n <= 0:
        raise ValueError("--data_num must be > 0")
    ids = list(range(1, n + 1))
    vr = float(cfg["val_rate"])
    if vr <= 0:
        val_ids = []
        train_ids = ids
    else:
        val_n = int(round(n * vr))
        val_n = max(1, min(val_n, n - 1))
        val_ids = ids[-val_n:]
        train_ids = ids[:-val_n]

    test_ids = []
    if bool(cfg["gen_test"]):
        test_n = max(1, int(round(n * 0.1)))
        test_ids = list(range(n + 1, n + test_n + 1))

    split_report["synthetic_split"] = {"train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids}

    tasks = [(cid, "train") for cid in train_ids] + [(cid, "val") for cid in val_ids] + [(cid, "test") for cid in test_ids]

    SPACE = (int(cfg["space"]), int(cfg["space"]), int(cfg["space"]))
    save_dict = dict(imagesTr=str(imagesTr), labelsTr=str(labelsTr), imagesTs=str(imagesTs))

    # run generation
    if int(args.pool_workers) <= 1:
        _init_worker(cfg, bank, label_ids, SPACE, save_dict, labels_map, anchor_priors, relations)
        for t in tqdm(tasks, desc="[gen]"):
            _worker_generate(t)
    else:
        with ProcessPoolExecutor(
            max_workers=int(args.pool_workers),
            initializer=_init_worker,
            initargs=(cfg, bank, label_ids, SPACE, save_dict, labels_map, anchor_priors, relations),
        ) as pool:
            for _ in tqdm(pool.map(_worker_generate, tasks), total=len(tasks), desc="[gen]"):
                pass

    write_dataset_json(out_dir, labels_map, train_ids, val_ids, test_ids)

    record = deepcopy(cfg)
    record["totalseg_root"] = str(root_dir)
    record["meta_csv"] = str(meta_csv)
    record["bank_dir"] = str(bank_dir)
    record["out_dir"] = str(out_dir)
    record["has_scipy"] = _HAS_SCIPY
    record["labels_total"] = len(labels_map) - 1
    record["labels_with_samples"] = len(label_ids)

    (out_dir / "config_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (out_dir / "split_report.json").write_text(json.dumps(split_report, indent=2), encoding="utf-8")

    print("[done] synthetic dataset saved to:", out_dir)
    print("[done] split_report.json saved (leakage audit):", out_dir / "split_report.json")


if __name__ == "__main__":
    main()