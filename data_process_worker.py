"""数据处理后台：自动数据分类（用模型预测类别，把图片归入类别文件夹）。"""

import argparse
import os
import shutil
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="data process worker")
    parser.add_argument("--mode", required=True, choices=["classify"])
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from ultralytics import YOLO

    model = YOLO(args.model)
    names = model.names
    out_root = os.path.join(args.source, "自动分类")
    os.makedirs(out_root, exist_ok=True)

    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        verbose=False,
    )
    counts: dict[str, int] = {}
    total = 0
    for r in results:
        path = getattr(r, "path", "")
        if not path or not os.path.isfile(path):
            continue
        cls_name = "unknown"
        if r.boxes is not None and len(r.boxes) > 0:
            i = int(r.boxes.conf.argmax())
            c = int(r.boxes.cls[i])
            cls_name = names.get(c, str(c))
        dst_dir = os.path.join(out_root, cls_name)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(path, os.path.join(dst_dir, os.path.basename(path)))
        counts[cls_name] = counts.get(cls_name, 0) + 1
        total += 1

    print("CLASSIFY_DONE total=", total)
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print("OUTPUT=", out_root)


if __name__ == "__main__":
    main()
