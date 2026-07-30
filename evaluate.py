"""
Evaluate a trained SolarFCD checkpoint on the held-out Test split.

Produces every number reported in the paper:
    accuracy, macro precision / recall / F1, per-class report,
    Cohen's kappa, confusion matrix (CSV + PNG), inference time,
    and (optionally) FLOPs.

Example
-------
    python src/evaluate.py --data-root data/SolarFCD \
                           --model runs/vgg19/V19.keras \
                           --out-dir runs/vgg19
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tensorflow import keras
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from data_loader import CLASSES, build_generators
from utils import hms_string, plot_confusion_matrix


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a SolarFCD checkpoint.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--model", required=True, help="path to .keras / .h5 checkpoint")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--out-dir", default="runs/exp")
    p.add_argument("--flops", action="store_true",
                   help="also report FLOPs (requires keras-flops, TF2/Keras2)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_gen = build_generators(
        args.data_root,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        augment=False,
    )

    print(f"[i] Loading {args.model}")
    model = keras.models.load_model(args.model, compile=False)

    start = time.time()
    probs = model.predict(test_gen, verbose=1)
    elapsed = time.time() - start
    y_pred = np.argmax(probs, axis=1)
    y_true = test_gen.classes

    acc = accuracy_score(y_true, y_pred) * 100
    results = {
        "model": Path(args.model).name,
        "n_test_images": int(len(y_true)),
        "accuracy_percent": round(float(acc), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro")), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro")), 4),
        "cohen_kappa": round(float(cohen_kappa_score(y_true, y_pred)), 4),
        "inference_seconds": round(float(elapsed), 2),
        "ms_per_image": round(float(elapsed / max(len(y_true), 1) * 1000), 2),
    }

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASSES,
                                   digits=4, output_dict=True)
    results["per_class"] = {
        c: {k: round(float(v), 4) for k, v in report[c].items()} for c in CLASSES
    }

    if args.flops:
        try:
            from keras_flops import get_flops
            flops = get_flops(model, batch_size=1)
            results["flops_G"] = round(flops / 1e9, 4)
            results["flops_M"] = round(flops / 1e6, 2)
        except Exception as exc:                       # pragma: no cover
            print(f"[!] FLOPs skipped: {exc}")

    print("\n=== Test-set results ===")
    print(f"Accuracy        : {acc:.4f} %")
    print(f"F1 (macro)      : {results['f1_macro']}")
    print(f"Recall (macro)  : {results['recall_macro']}")
    print(f"Precision(macro): {results['precision_macro']}")
    print(f"Cohen's kappa   : {results['cohen_kappa']}")
    print(f"Inference time  : {hms_string(elapsed)}  ({results['ms_per_image']} ms/img)")
    print("\nConfusion matrix (rows = true, cols = predicted):")
    print(cm)
    print("\n" + classification_report(y_true, y_pred, target_names=CLASSES, digits=4))

    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",",
               header=",".join(CLASSES), comments="")
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    plot_confusion_matrix(cm, CLASSES, out_dir / "confusion_matrix.png")

    print(f"\n[=] Saved metrics.json, confusion_matrix.csv/.png to {out_dir}")


if __name__ == "__main__":
    main()
