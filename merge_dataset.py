# -*- coding: utf-8 -*-
"""合并 10 类番茄检测数据集（旧手动标注优先 + PlantVillage 自动整叶框）

数据源:
    1) E:\\PlantVillage-Dataset\\tomato-ready\\tomato_dataset  旧 2 类手动标注（train/val，类 0/1）
    2) E:\\tomato\\<类文件夹> + E:\\tomato\\labels\\<类文件夹>  PlantVillage 自动整叶框（类 0~9）

处理:
    - 按图片 MD5 去重（重复时保留先加入的样本，因此旧手动标注优先）
    - 按标签中的主类别做 8:1:1 分层划分（每个类都进 train/val/test）
    - 输出到 E:\\tomato\\tomato_multi\\ 并生成 dataset.yaml

用法:
    python merge_dataset.py
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auto_boxes import CLASS_ORDER, FOLDER_MAP  # noqa: E402

OLD_DATASET = r"E:\PlantVillage-Dataset\tomato-ready\tomato_dataset"
PV_ROOT = r"E:\tomato"
PV_LABELS = r"E:\tomato\labels"
OUT_ROOT = r"E:\tomato\tomato_multi"
SEED = 0

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def primary_class(label_path: str) -> int:
    counts: dict[int, int] = {}
    with open(label_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if parts:
                try:
                    cls = int(parts[0])
                except ValueError:
                    continue
                counts[cls] = counts.get(cls, 0) + 1
    if not counts:
        return -1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def collect_old(seen: dict, samples: list) -> int:
    added = 0
    for split in ("train", "val"):
        img_dir = os.path.join(OLD_DATASET, split, "images")
        lbl_dir = os.path.join(OLD_DATASET, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        for name in sorted(os.listdir(img_dir)):
            if os.path.splitext(name)[1].lower() not in IMAGE_EXTS:
                continue
            img = os.path.join(img_dir, name)
            lbl = os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt")
            if not os.path.isfile(lbl):
                continue
            digest = md5(img)
            if digest in seen:
                continue
            cls = primary_class(lbl)
            if cls < 0:
                continue
            seen[digest] = True
            samples.append({"img": img, "lbl": lbl, "cls": cls, "src": "old", "name": name})
            added += 1
    return added


def collect_pv(seen: dict, samples: list) -> int:
    added = 0
    for folder, cls_idx in FOLDER_MAP.items():
        img_dir = os.path.join(PV_ROOT, folder)
        lbl_dir = os.path.join(PV_LABELS, folder)
        if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
            continue
        for name in sorted(os.listdir(img_dir)):
            if os.path.splitext(name)[1].lower() not in IMAGE_EXTS:
                continue
            img = os.path.join(img_dir, name)
            lbl = os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt")
            if not os.path.isfile(lbl):
                continue
            digest = md5(img)
            if digest in seen:
                continue
            seen[digest] = True
            samples.append({"img": img, "lbl": lbl, "cls": cls_idx, "src": f"pv{cls_idx}", "name": name})
            added += 1
    return added


def main() -> None:
    ap = argparse.ArgumentParser(description="合并 10 类番茄检测数据集")
    ap.add_argument("--out", default=OUT_ROOT, help="输出目录")
    ap.add_argument("--skip-old", action="store_true", help="不并入旧手动标注，统一使用整叶框")
    args = ap.parse_args()
    out_root = args.out
    os.makedirs(out_root, exist_ok=True)
    seen: dict = {}
    samples: list = []
    n_old = 0
    if args.skip_old:
        print("[跳过] 旧手动标注（--skip-old），统一使用 PlantVillage 整叶框标注")
    else:
        n_old = collect_old(seen, samples)
    n_pv = collect_pv(seen, samples)
    print(f"旧手动标注: {n_old}  PlantVillage 自动框: {n_pv}  去重后合计: {len(samples)}")

    # 按主类别 8:1:1 分层划分
    rng = random.Random(SEED)
    buckets: dict[int, list] = {}
    for s in samples:
        buckets.setdefault(s["cls"], []).append(s)
    assign = {"train": [], "val": [], "test": []}
    for cls, group in sorted(buckets.items()):
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        assign["train"].extend(group[:n_train])
        assign["val"].extend(group[n_train : n_train + n_val])
        assign["test"].extend(group[n_train + n_val :])

    # 复制输出（源前缀避免同名冲突）
    stats: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        img_dir = os.path.join(out_root, split, "images")
        lbl_dir = os.path.join(out_root, split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        stats[split] = {name: 0 for name in CLASS_ORDER}
        for s in assign[split]:
            stem = os.path.splitext(s["name"])[0]
            new_img = os.path.join(img_dir, f"{s['src']}_{stem}{os.path.splitext(s['name'])[1]}")
            new_lbl = os.path.join(lbl_dir, f"{s['src']}_{stem}.txt")
            if not os.path.exists(new_img):
                shutil.copy2(s["img"], new_img)
            if not os.path.exists(new_lbl):
                shutil.copy2(s["lbl"], new_lbl)
            stats[split][CLASS_ORDER[s["cls"]]] += 1
        print(f"[{split}] {sum(stats[split].values())} 张")

    yaml_text = (
        "# 10 类番茄检测数据集（merge_dataset.py 生成）\n"
        f"path: {out_root.replace(os.sep, '/')}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "nc: 10\n"
        "names: " + json.dumps(CLASS_ORDER, ensure_ascii=False) + "\n"
    )
    with open(os.path.join(out_root, "dataset.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml_text)
    with open(os.path.join(out_root, "merge_report.json"), "w", encoding="utf-8") as f:
        json.dump({"total": len(samples), "old_annotated": n_old, "pv_auto": n_pv,
                   "splits": stats, "classes": CLASS_ORDER}, f, ensure_ascii=False, indent=2)

    print("\n===== 各类别分布 (train/val/test) =====")
    header = "类别\t" + "\t".join(("train", "val", "test"))
    print(header)
    for cls in CLASS_ORDER:
        print(f"{cls}\t{stats['train'].get(cls, 0)}\t{stats['val'].get(cls, 0)}\t{stats['test'].get(cls, 0)}")
    print(f"\n数据集: {out_root}")
    print(f"配置: {os.path.join(out_root, 'dataset.yaml')}")
    print(f"报告: {os.path.join(out_root, 'merge_report.json')}")


if __name__ == "__main__":
    main()
