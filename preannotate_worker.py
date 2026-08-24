"""Pre-annotation worker: batch-annotate images with a trained model."""

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO pre-annotation worker")
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", required=True)
    parser.add_argument("--save_txt", type=int, default=1, help="1=also write YOLO label txt")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from ultralytics import YOLO

    model = YOLO(args.model)
    os.makedirs(args.out, exist_ok=True)
    model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,
        save_txt=bool(args.save_txt),
        project=args.out,
        name="pre",
        exist_ok=True,
    )
    print("PREANNOTATE_DONE")


if __name__ == "__main__":
    main()
