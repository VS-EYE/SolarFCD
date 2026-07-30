# SolarFCD — Solar Panel Fault Classification

Official code for the paper *"<Paper title>"*.
This repository contains the complete pipeline used to produce every result
reported in the paper: dataset download, data loading, model construction,
training, and evaluation.

| | |
|---|---|
| **Task** | 4-class image classification of solar-panel faults |
| **Dataset** | SolarFCD (3,742 images, ~5 GB) — hosted on DFKI Cloud |
| **Classes** | `Electrical_Fault`, `Healthy`, `Structural_Fault`, `Surface_Obstruction` |
| **Input** | 224 × 224 × 3 |
| **Backbones** | VGG16/19, ResNet50, DenseNet121/201, MobileNet(V2), InceptionV3, Xception, EfficientNetB0, ViT-B16 |
| **Attention** | none / channel / SE / CBAM |
| **License** | MIT (code) — see [`LICENSE`](LICENSE) |

---

## 1. Dataset

### 1.1 Availability

The **SolarFCD** dataset is publicly available from the DFKI Cloud:

**https://cloud.dfki.de/owncloud/index.php/s/SZYo97gdwY2k92Z**

The dataset is *not* stored in this repository (≈5 GB); it is downloaded on demand.

### 1.2 Automatic download (recommended)

```bash
python src/download_dataset.py --out data
```

This downloads the archive (resumable), extracts it, locates the folder that
contains `Train/ Valid/ Test/`, and prints a per-class image count so you can
confirm the copy is complete. Expect ~11 GB of free disk space during extraction
(the archive is deleted afterwards unless you pass `--keep-zip`).

### 1.3 Manual download

1. Open the link above in a browser and click **Download all files**, or run:
   ```bash
   wget -O SolarFCD.zip "https://cloud.dfki.de/owncloud/index.php/s/SZYo97gdwY2k92Z/download"
   # or:  curl -L -o SolarFCD.zip "https://cloud.dfki.de/owncloud/index.php/s/SZYo97gdwY2k92Z/download"
   ```
2. Unzip it into `data/`.
3. Verify the layout:
   ```bash
   python src/download_dataset.py --out data --verify-only
   ```

### 1.4 Expected directory structure

```
data/SolarFCD/
├── Train/                       # 3,030 images
│   ├── Electrical_Fault/
│   ├── Healthy/
│   ├── Structural_Fault/
│   └── Surface_Obstruction/
├── Valid/                       #   337 images
│   └── ... (same four folders)
└── Test/                        #   375 images
    └── ... (same four folders)
```

The class list is hard-coded in `src/data_loader.py` and passed explicitly to
every generator, so the label indices are identical on every machine:

| index | class |
|---|---|
| 0 | `Electrical_Fault` |
| 1 | `Healthy` |
| 2 | `Structural_Fault` |
| 3 | `Surface_Obstruction` |

If your copy lives elsewhere, pass `--data-root /path/to/SolarFCD` to every
script — no absolute paths are hard-coded anywhere in this repository.

---

## 2. Installation

Tested on Ubuntu 22.04 and Windows 11, Python 3.10–3.11, NVIDIA GPU with CUDA 12
(CPU-only also works, roughly 15× slower).

```bash
git clone https://github.com/<your-username>/SolarFCD-Classification.git
cd SolarFCD-Classification

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

Check that TensorFlow sees your GPU:

```bash
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
```

<details>
<summary><b>ViT-B16 and FLOPs (optional, separate environment)</b></summary>

`vit-keras` and `keras-flops` have not been ported to Keras 3. Run those two
experiments in a legacy environment:

```bash
python -m venv .venv-vit
source .venv-vit/bin/activate
pip install -r requirements-legacy-vit.txt
python src/train.py --data-root data/SolarFCD --backbone vit_b16 --out-dir runs/vit_b16
```
</details>

---

## 3. Running the code

### 3.1 Quick start (three commands)

```bash
python src/download_dataset.py --out data
python src/train.py    --data-root data/SolarFCD --backbone vgg19 --out-dir runs/vgg19
python src/evaluate.py --data-root data/SolarFCD --model runs/vgg19/V19.keras --out-dir runs/vgg19
```

### 3.2 Training

```bash
python src/train.py --data-root data/SolarFCD \
                    --backbone vgg19 \
                    --attention none \
                    --epochs 20 \
                    --batch-size 64 \
                    --lr 0.001 \
                    --seed 42 \
                    --out-dir runs/vgg19
```

| Argument | Default | Meaning |
|---|---|---|
| `--data-root` | *required* | folder containing `Train/ Valid/ Test/` |
| `--backbone` | `vgg19` | `vgg16`, `vgg19`, `resnet50`, `densenet121`, `densenet201`, `mobilenet`, `mobilenetv2`, `inceptionv3`, `xception`, `efficientnetb0`, `vit_b16` |
| `--attention` | `none` | `none`, `channel`, `se`, `cbam` |
| `--epochs` | `20` | training epochs |
| `--batch-size` | `64` | mini-batch size (lower it to 32/16 if you hit OOM) |
| `--lr` | `0.001` | Adam learning rate |
| `--img-size` | `224` | input resolution |
| `--no-augment` | off | disable shift/flip augmentation |
| `--seed` | `42` | seeds Python, NumPy and TensorFlow |
| `--out-dir` | `runs/exp` | where checkpoints and logs are written |

Training settings match the paper: ImageNet backbone **frozen**, head =
`Flatten → Dense(4, softmax)`, `categorical_crossentropy`, Adam(1e-3),
`ModelCheckpoint` on `val_accuracy` (best only), and `ReduceLROnPlateau` on
`val_loss` (factor 0.1, patience 3).

Outputs written to `--out-dir`:

```
runs/vgg19/
├── V19.keras            # best checkpoint (highest val_accuracy)
├── history.json         # per-epoch loss/accuracy/precision/recall/AUC
└── train_config.json    # every hyper-parameter + TF version + wall-clock time
```

### 3.3 Evaluation

```bash
python src/evaluate.py --data-root data/SolarFCD \
                       --model runs/vgg19/V19.keras \
                       --out-dir runs/vgg19
```

Prints and saves accuracy, macro precision / recall / F1, per-class report,
Cohen's κ, the confusion matrix, and inference time. Add `--flops` (legacy
environment only) to report FLOPs.

```
runs/vgg19/
├── metrics.json
├── confusion_matrix.csv
└── confusion_matrix.png
```

### 3.4 Reproducing the full comparison table

```bash
for BACKBONE in vgg16 vgg19 resnet50 densenet121 mobilenet efficientnetb0; do
  python src/train.py    --data-root data/SolarFCD --backbone $BACKBONE --out-dir runs/$BACKBONE
  python src/evaluate.py --data-root data/SolarFCD --out-dir runs/$BACKBONE \
                         --model "$(ls runs/$BACKBONE/*.keras | head -1)"
done
```

Attention variants (e.g. the MobileNet + channel-attention model, `MBCA`):

```bash
python src/train.py --data-root data/SolarFCD --backbone mobilenet --attention channel \
                    --out-dir runs/mobilenet_ca
```

### 3.5 Notebook

`notebooks/Base.ipynb` is the reference notebook, identical in content to the
scripts but laid out cell-by-cell. Set `DATA_ROOT` in the second cell (or export
`SOLARFCD_ROOT=/path/to/SolarFCD`) and run all cells.

```bash
jupyter notebook notebooks/Base.ipynb
```

---

## 4. Repository layout

```
SolarFCD-Classification/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt                # main environment (Keras 3)
├── requirements-legacy-vit.txt     # optional env for ViT-B16 / FLOPs
├── configs/
│   └── default.yaml                # all hyper-parameters in one place
├── notebooks/
│   └── Base.ipynb                  # reference notebook
├── src/
│   ├── download_dataset.py         # fetch + verify SolarFCD from DFKI Cloud
│   ├── data_loader.py              # generators, fixed class order
│   ├── blocks.py                   # SE / CBAM / inception / residual blocks
│   ├── models.py                   # model factory (backbone × attention)
│   ├── train.py                    # training entry point
│   ├── evaluate.py                 # metrics, confusion matrix, FLOPs
│   └── utils.py                    # seeding, timing, plotting
└── docs/
    └── REVIEWER_RESPONSE.md        # data & code availability statements
```

---

## 5. Reproducibility notes

* `--seed 42` seeds Python, NumPy and TensorFlow. Exact bit-for-bit
  reproducibility on GPU also requires `TF_DETERMINISTIC_OPS=1`; small
  variations (≲0.5 % accuracy) between runs are expected otherwise.
* The test generator uses `shuffle=False`; do not change this, or predictions
  will no longer align with `test_gen.classes` and the confusion matrix will be
  meaningless.
* Checkpoints are saved in the native `.keras` format. Legacy `.h5` files still
  load via `keras.models.load_model(path)`.
* Reference environment: Python 3.11, TensorFlow 2.17, CUDA 12.3, NVIDIA RTX GPU.

---

## 6. Citation

```bibtex
@article{<key>,
  title   = {<Paper title>},
  author  = {<Authors>},
  journal = {<Journal>},
  year    = {2026},
  doi     = {<DOI>}
}
```

Dataset: SolarFCD, DFKI Cloud, https://cloud.dfki.de/owncloud/index.php/s/SZYo97gdwY2k92Z

---

## 7. Contact

Questions and issues: please open a
[GitHub issue](https://github.com/<your-username>/SolarFCD-Classification/issues)
or contact `<your.email@dfki.de>`.
