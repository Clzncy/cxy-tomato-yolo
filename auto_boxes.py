# -*- coding: utf-8 -*-
"""PlantVillage 番茄分类图 -> YOLO 检测标签（自动整叶框，零人工标注）

用法:
    python auto_boxes.py --src E:\\tomato --out E:\\tomato\\labels
    python auto_boxes.py --src E:\\tomato --out E:\\tomato\\labels --limit 50   # 小批量试跑

说明:
    - 每个类文件夹对应一个固定类别索引（与 10 类 dataset.yaml 保持一致）
    - 用 绿色掩膜 + LAB a 通道 Otsu 找叶片轮廓，取最大轮廓的外接框
    - 找不到可靠轮廓时回退为"整图缩进 3%"的框
    - 已存在的标签默认跳过（--overwrite 可强制重算）
"""

import argparse
import json
import os

import cv2
import numpy as np

# 10 类顺序：healthy / early_blight 必须在前两位，与旧标注(0/1)兼容
CLASS_ORDER = [
    "healthy",
    "early_blight",
    "late_blight",
    "leaf_mold",
    "bacterial_spot",
    "septoria_leaf_spot",
    "target_spot",
    "tomato_mosaic_virus",
    "yellow_leaf_curl_virus",
    "spider_mites",
]

FOLDER_MAP = {
    "Tomato___healthy": 0,
    "Tomato___Early_blight": 1,
    "Tomato___Late_blight": 2,
    "Tomato___Leaf_Mold": 3,
    "Tomato___Bacterial_spot": 4,
    "Tomato___Septoria_leaf_spot": 5,
    "Tomato___Target_Spot": 6,
    "Tomato___Tomato_mosaic_virus": 7,
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": 8,
    "Tomato___Spider_mites Two-spotted_spider_mite": 9,
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
FALLBACK_INSET = 0.03  # 兜底框：整图四周缩进 3%
MIN_AREA_RATIO = 0.03  # 轮廓小于画面 3% 视为失败


def leaf_bbox(img_bgr):
    """返回叶片/前景的最大外接框 (x, y, w, h)，失败返回 None。"""
    h, w = img_bgr.shape[:2]
    color = img_bgr if img_bgr.ndim == 3 else cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    # 1) 绿色叶片掩膜
    green = cv2.inRange(hsv, (25, 50, 50), (95, 255, 255))
    # 2) LAB a 通道 Otsu（黄叶/病斑、浅色背景也能分出来），正反两个方向都试
    lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)
    a_ch = lab[:, :, 1]
    _, otsu = cv2.threshold(a_ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_or(green, otsu)
    mask = cv2.bitwise_or(mask, cv2.bitwise_not(otsu))
    # 3) 形态学清理小噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)
    if bw < 5 or bh < 5:
        return None
    if (bw * bh) / (w * h) < MIN_AREA_RATIO:
        return None
    return x, y, bw, bh


def fallback_bbox(w, h):
    """兜底：整图缩进 3%。"""
    ix = int(w * FALLBACK_INSET)
    iy = int(h * FALLBACK_INSET)
    return ix, iy, max(w - 2 * ix, 1), max(h - 2 * iy, 1)


def to_yolo_line(cls_idx, box, w, h):
    x, y, bw, bh = box
    cx = (x + bw / 2.0) / w
    cy = (y + bh / 2.0) / h
    nw = bw / w
    nh = bh / h
    cx = float(np.clip(cx, 0.0, 1.0))
    cy = float(np.clip(cy, 0.0, 1.0))
    nw = float(np.clip(nw, 0.0, 1.0))
    nh = float(np.clip(nh, 0.0, 1.0))
    return f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="PlantVillage 分类图自动生成 YOLO 整叶框标签")
    ap.add_argument("--src", required=True, help="PlantVillage 类文件夹所在目录，如 E:\\tomato")
    ap.add_argument("--out", required=True, help="标签输出目录，如 E:\\tomato\\labels")
    ap.add_argument("--limit", type=int, default=0, help="每个类最多处理多少张（0=全部，用于试跑）")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已存在的标签")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    total_done = total_skip = total_fail = 0
    class_stats = {}

    for folder, cls_idx in FOLDER_MAP.items():
        folder_path = os.path.join(args.src, folder)
        if not os.path.isdir(folder_path):
            print(f"[跳过] 找不到文件夹: {folder_path}")
            continue
        images = [n for n in os.listdir(folder_path) if os.path.splitext(n)[1].lower() in IMAGE_EXTS]
        images.sort()
        if args.limit > 0:
            images = images[: args.limit]
        out_dir = os.path.join(args.out, folder)
        os.makedirs(out_dir, exist_ok=True)
        done = skip = fail = 0
        for name in images:
            stem = os.path.splitext(name)[0]
            label_path = os.path.join(out_dir, stem + ".txt")
            if os.path.exists(label_path) and not args.overwrite:
                skip += 1
                continue
            img = cv2.imread(os.path.join(folder_path, name))
            if img is None:
                fail += 1
                continue
            h, w = img.shape[:2]
            box = leaf_bbox(img)
            if box is None:
                box = fallback_bbox(w, h)
                fail += 1  # 统计为"用了兜底框"
            with open(label_path, "w", encoding="utf-8") as f:
                f.write(to_yolo_line(cls_idx, box, w, h) + "\n")
            done += 1
            total_done += 1
            if total_done % 500 == 0:
                print(f"[进度] 已生成 {total_done} 个标签 ...")
        class_stats[folder] = {"class": CLASS_ORDER[cls_idx], "done": done, "skipped": skip, "fallback": fail}
        print(f"[完成] {folder} -> {CLASS_ORDER[cls_idx]} (index {cls_idx}) 生成 {done} 跳过 {skip} 兜底 {fail}")
        total_skip += skip
        total_fail += fail

    # 类别映射与 yaml 骨架，方便后面合并数据集
    with open(os.path.join(args.out, "class_map.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"classes": CLASS_ORDER, "folder_map": FOLDER_MAP, "stats": class_stats},
            f, ensure_ascii=False, indent=2,
        )
    yaml_text = (
        "# 10 类番茄检测数据集（由 auto_boxes.py 生成骨架）\n"
        "path: E:/tomato\n"
        "train: tomato_multi/train/images\n"
        "val: tomato_multi/val/images\n"
        "test: tomato_multi/test/images\n"
        "nc: 10\n"
        "names: " + json.dumps(CLASS_ORDER, ensure_ascii=False) + "\n"
    )
    with open(os.path.join(args.out, "dataset_10class.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml_text)

    print("\n===== 汇总 =====")
    print(f"生成: {total_done}  跳过(已存在): {total_skip}  兜底框: {total_fail}")
    print(f"类别映射: {os.path.join(args.out, 'class_map.json')}")
    print(f"10 类 yaml 骨架: {os.path.join(args.out, 'dataset_10class.yaml')}")


if __name__ == "__main__":
    main()
