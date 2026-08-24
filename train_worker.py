"""Training worker: launched by yolo_trainer_app.py as a subprocess."""

import argparse
import re
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO training worker")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="")
    parser.add_argument("--extra", default="")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from ultralytics import YOLO

    model = YOLO(args.model)
    kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
    }
    if args.name:
        kwargs["name"] = args.name

    for token in re.split(r"[,\s]+", args.extra.strip()):
        if "=" in token:
            key, value = token.split("=", 1)
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
            kwargs[key] = value

    model.train(**kwargs)
    print("TRAINING_DONE")


if __name__ == "__main__":
    main()
