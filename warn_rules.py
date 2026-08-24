# -*- coding: utf-8 -*-
"""番茄病害预警规则脚本：批量扫描图片，输出 健康/观察/处理 三级预警。

用法:
    python warn_rules.py --model <best.pt> --source E:\\tomato\\test_pics \
        --out E:\\tomato\\warn_out --conf 0.25 --area 0.05 --confirm 1

输出:
    - 控制台逐张结果 + 汇总
    - 带框标注图: <out>/annotated/
    - 结构化结果: <out>/warnings.json

预警规则:
    绿: 未检出任何非 healthy 类别
    黄: 检出病害/虫害，且非高危害类别、面积占比 < 20%
    红: 高危害类别(晚疫病/病毒病)，或 患病叶片占比 >= 阈值(默认 0.5) 且患病叶片 >= 2；
        连续帧确认后生效
"""

import argparse
import json
import os
import re

import cv2


def _ensure_ultralytics_settings() -> None:
    """权限受限环境下，把 ultralytics 配置目录指到脚本旁，避免启动失败。"""
    cfg = os.path.join(os.environ.get("APPDATA", ""), "Ultralytics", "settings.json")
    if os.path.exists(cfg):
        try:
            with open(cfg, "r", encoding="utf-8"):
                pass
            return
        except OSError:
            pass
    os.environ.setdefault(
        "YOLO_CONFIG_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".yolo_config"),
    )


_ensure_ultralytics_settings()

from ultralytics import YOLO  # noqa: E402

# 与 auto_boxes.py / dataset_10class.yaml 保持一致的 10 类顺序
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

# 危害等级：3=高（立即处理） 2=中（观察/尽快处理） 0=健康
SEVERITY = {0: 0, 1: 2, 2: 3, 3: 2, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 2}

ADVICE = {
    1: "早疫病：摘除病叶，喷施代森锰锌/苯醚甲环唑，注意轮换用药。",
    2: "晚疫病：高危害！立即喷施烯酰吗啉/氟吡菌胺，清除病株病叶，加强通风降湿。",
    3: "叶霉病：控湿通风，喷施嘧菌酯/异菌脲，摘除下部老叶病叶。",
    4: "细菌性斑点病：铜制剂（氢氧化铜/春雷霉素），避免雨天农事操作。",
    5: "斑枯病：清除病残体，喷施苯醚甲环唑/代森锰锌。",
    6: "靶斑病：轮换使用嘧菌酯、戊唑醇，改善通风透光。",
    7: "花叶病毒病：高危害！拔除重病株，防治蚜虫（传毒媒介），工具消毒。",
    8: "黄化曲叶病毒病：高危害！先治烟粉虱/蚜虫，拔除重病株，选用抗病品种。",
    9: "红蜘蛛：喷施阿维菌素/联苯肼酯，重点喷叶背，天敌保护。",
}

GENERIC_ADVICE = "建议尽快人工复核病斑，拍照记录并咨询当地植保站。"
LEVELS = ["绿", "黄", "红"]
FRAME_RE = re.compile(r"^(.*?)[_\-]?(\d{2,})(\.[a-zA-Z]+)$")


def frame_group(name: str):
    """把 cam_0001.jpg 归到前缀 cam；没有帧号的返回原文件名。"""
    m = FRAME_RE.match(name)
    if m:
        return m.group(1)
    return os.path.splitext(name)[0]


def pick_advice(classes: dict) -> str:
    top = max(classes.items(), key=lambda kv: kv[1])
    return ADVICE.get(top[0], GENERIC_ADVICE)


def box_iou(a, b) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def dedup_boxes(boxes: list, thr: float = 0.5) -> list:
    """按置信度排序，重叠超过 thr 的框只保留一个（整叶框场景：同一片叶子算一个）。"""
    ordered = sorted(boxes, key=lambda b: b["conf"], reverse=True)
    kept: list = []
    for b in ordered:
        if all(box_iou(b["xyxy"], k["xyxy"]) < thr for k in kept):
            kept.append(b)
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description="番茄病害/虫害预警规则引擎")
    ap.add_argument("--model", required=True, help="YOLO 模型权重 .pt")
    ap.add_argument("--source", required=True, help="单张图片或图片文件夹")
    ap.add_argument("--out", default=r"E:\tomato\warn_out", help="报告输出目录")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    ap.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    ap.add_argument("--area", type=float, default=0.5,
                    help="患病叶片占比阈值：患病框数/总框数（整叶标注下用叶片占比）")
    ap.add_argument("--confirm", type=int, default=1, help="连续 N 帧确认才升级为红色预警")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "annotated"), exist_ok=True)
    model = YOLO(args.model)

    rows = []
    group_status = {}  # 前缀 -> 非绿图片数
    results = model.predict(source=args.source, conf=args.conf, imgsz=args.imgsz, stream=True, verbose=False)
    for r in results:
        name = os.path.basename(r.path)
        img_w, img_h = r.orig_shape[1], r.orig_shape[0]
        cls_counts: dict[int, int] = {}
        max_conf = 0.0
        max_area = 0.0
        max_sev = 0
        total_boxes = 0
        diseased_boxes = 0
        raw_boxes: list = []
        if r.boxes is not None:
            for b in r.boxes:
                raw_boxes.append({
                    "cls": int(b.cls.item()),
                    "conf": float(b.conf.item()),
                    "xyxy": b.xyxy[0].tolist(),
                })
        for b in dedup_boxes(raw_boxes, 0.5):
            cls = b["cls"]
            conf = b["conf"]
            total_boxes += 1
            if cls == 0:  # healthy 不算异常
                continue
            diseased_boxes += 1
            cls_counts[cls] = cls_counts.get(cls, 0) + 1
            max_conf = max(max_conf, conf)
            x1, y1, x2, y2 = b["xyxy"]
            area_ratio = max((x2 - x1) * (y2 - y1) / (img_w * img_h), 0.0)
            max_area = max(max_area, area_ratio)
            max_sev = max(max_sev, SEVERITY.get(cls, 2))
        leaf_ratio = diseased_boxes / total_boxes if total_boxes else 0.0

        if not cls_counts:
            level = "绿"
            classes = "healthy"
            advice = "无需处理，继续保持监测。"
        else:
            if max_sev >= 3 or (leaf_ratio >= args.area and diseased_boxes >= 2):
                level = "红"
            else:
                level = "黄"
            classes = ", ".join(f"{CLASS_ORDER[c] if c < len(CLASS_ORDER) else str(c)}x{n}"
                                for c, n in sorted(cls_counts.items()))
            advice = pick_advice(cls_counts)

        g = frame_group(name)
        group_status[g] = group_status.get(g, 0) + (1 if level != "绿" else 0)
        rows.append({"file": name, "group": g, "level": level, "classes": classes,
                     "max_conf": round(max_conf, 4), "area_ratio": round(max_area, 4),
                     "leaf_ratio": round(leaf_ratio, 4), "advice": advice})

        # 保存带框标注图
        ann = r.plot()
        cv2.imwrite(os.path.join(args.out, "annotated", name), ann)

    # 连续帧确认：红 需要同组 >= confirm 张非绿；不足则降为黄并标注待确认
    if args.confirm > 1:
        for row in rows:
            if row["level"] == "红" and group_status.get(row["group"], 0) < args.confirm:
                row["level"] = "黄"
                row["advice"] = "[待连续帧确认] " + row["advice"]

    # 汇总与输出
    summary = {"绿": 0, "黄": 0, "红": 0}
    for row in rows:
        summary[row["level"]] += 1
        print(f"{row['file']}\t{row['level']}\t{row['classes']}\tconf={row['max_conf']:.2f}\t"
              f"患病叶片={row['leaf_ratio']:.0%}\t{row['advice']}")

    print("\n===== 预警汇总 =====")
    print(f"图片总数: {len(rows)}  绿: {summary['绿']}  黄: {summary['黄']}  红: {summary['红']}")
    with open(os.path.join(args.out, "warnings.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "images": rows}, f, ensure_ascii=False, indent=2)
    print(f"报告已保存: {os.path.join(args.out, 'warnings.json')}")
    print(f"标注图: {os.path.join(args.out, 'annotated')}")


if __name__ == "__main__":
    main()
