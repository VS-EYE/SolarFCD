"""
Train a classifier on SolarFCD.

Example
-------
    python src/train.py --data-root data/SolarFCD \
                        --backbone vgg19 --epochs 20 --batch-size 64 \
                        --out-dir runs/vgg19

The defaults reproduce the setting reported in the paper:
    224x224 input, batch 64, Adam(1e-3), 20 epochs, frozen ImageNet backbone,
    ModelCheckpoint on val_accuracy, ReduceLROnPlateau on val_loss.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from data_loader import CLASSES, build_generators
from models import create_model
from utils import hms_string, set_global_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train a SolarFCD classifier.")
    p.add_argument("--data-root", required=True,
                   help="folder containing Train/ Valid/ Test/")
    p.add_argument("--backbone", default="vgg19",
                   help="vgg19 | densenet121 | mobilenet | resnet50 | "
                        "efficientnetb0 | inceptionv3 | xception | vit_b16 ...")
    p.add_argument("--attention", default="none",
                   choices=["none", "channel", "se", "cbam"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--no-augment", action="store_true",
                   help="disable shift/flip augmentation on the training split")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="runs/exp")
    return p.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpus = tf.config.list_physical_devices("GPU")
    print(f"[i] TensorFlow {tf.__version__} | GPUs visible: {len(gpus)}")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass

    train_gen, valid_gen, test_gen = build_generators(
        args.data_root,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        augment=not args.no_augment,
        seed=args.seed,
    )
    print(f"[i] class_indices = {train_gen.class_indices}")

    model = create_model(
        backbone=args.backbone,
        n_classes=len(CLASSES),
        optimizer=Adam(learning_rate=args.lr),
        input_shape=(args.img_size, args.img_size, 3),
        attention=args.attention,
    )
    model.summary()

    ckpt_path = out_dir / f"{model.name}.keras"
    callbacks = [
        ModelCheckpoint(filepath=str(ckpt_path), save_best_only=True,
                        monitor="val_accuracy", mode="max", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3,
                          mode="auto", min_delta=1e-5, min_lr=0.0, verbose=1),
    ]

    start = time.time()
    history = model.fit(train_gen,
                        epochs=args.epochs,
                        validation_data=valid_gen,
                        callbacks=callbacks,
                        verbose=1)
    elapsed = time.time() - start
    print(f"Training time: {hms_string(elapsed)}")

    (out_dir / "history.json").write_text(
        json.dumps({k: [float(v) for v in vals]
                    for k, vals in history.history.items()}, indent=2))
    (out_dir / "train_config.json").write_text(
        json.dumps({**vars(args), "classes": CLASSES,
                    "model_name": model.name,
                    "training_seconds": round(elapsed, 2),
                    "tf_version": tf.__version__}, indent=2))

    print(f"\n[=] Best checkpoint : {ckpt_path}")
    print(f"[=] Next step       : python src/evaluate.py --data-root {args.data_root} "
          f"--model {ckpt_path} --out-dir {out_dir}")


if __name__ == "__main__":
    main()
