"""Data acquisition and preprocessing for handwritten digits. 
Every image (training data, uploads, and the image sent to /predict) goes through the
same `canonicalize()` step, so the model always sees digits in one consistent format.
"""
import io
import random
import shutil
import uuid
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

from src import config

RAW_DIR = config.DATA_DIR / "raw"
NATIVE_SIZE = (8, 8)      


# 1. Data acquisition

def class_counts(folder: Path) -> dict:
    folder = Path(folder)
    if not folder.exists():
        return {}
    return {
        d.name: sum(1 for f in d.iterdir() if f.suffix.lower() in config.IMAGE_EXTS)
        for d in sorted(folder.iterdir()) if d.is_dir()
    }


def acquire_data(test_split: float = 0.2) -> dict:
    """Use the images in data/raw if present; otherwise export scikit-learn's digits
    dataset (1,797 real handwritten 8x8 digits) to data/raw as PNG files."""
    if any((RAW_DIR / "train").glob("*/*.png")):
        print("Using existing images in data/raw")
    else:
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split

        digits = load_digits()
        X, y = digits.images, digits.target
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_split,
                                                  stratify=y, random_state=config.SEED)
        for split, XS, ys in (("train", X_tr, y_tr), ("test", X_te, y_te)):
            for i, (img, label) in enumerate(zip(XS, ys)):
                d = RAW_DIR / split / str(label)
                d.mkdir(parents=True, exist_ok=True)
                pixels = (img / 16 * 255).astype("uint8")          # 0-16 -> 0-255
                Image.fromarray(pixels).resize((64, 64), Image.NEAREST).save(d / f"img_{i}.png")
        print("Exported scikit-learn digits to data/raw")
    counts = {"train": class_counts(RAW_DIR / "train"), "test": class_counts(RAW_DIR / "test")}
    print({k: sum(v.values()) for k, v in counts.items()}, "images")
    return counts


# 2. Data processing

def canonicalize(image) -> Image.Image:
    """Any digit image -> 32x32 grayscale, white digit on black, cropped and centred."""
    if isinstance(image, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image))
    elif isinstance(image, (str, Path)):
        image = Image.open(image)
    img = image.convert("L")
    if np.asarray(img).mean() > 127:
        img = ImageOps.invert(img)
    arr = np.asarray(img, dtype=np.float32)
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max() * 255
    mask = arr > 0.25 * 255
    if mask.any():
        rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
        arr = arr[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    h, w = arr.shape
    side = max(h, w)                        #
    canvas = np.zeros((side, side), dtype=np.float32)
    top, left = (side - h) // 2, (side - w) // 2
    canvas[top:top + h, left:left + w] = arr
    img = Image.fromarray(canvas.astype("uint8"))
    # Pass through the dataset's native 8x8 resolution so sharp photos look like training data
    img = img.resize(NATIVE_SIZE, Image.BOX)
    return img.resize(config.IMG_SIZE, Image.BILINEAR)


def process_dataset() -> dict:
    """Canonicalize every raw image into data/train and data/test."""
    for split, out in (("train", config.TRAIN_DIR), ("test", config.TEST_DIR)):
        if out.exists():
            shutil.rmtree(out)
        for f in sorted((RAW_DIR / split).glob("*/*")):
            if f.suffix.lower() not in config.IMAGE_EXTS:
                continue
            dest = out / f.parent.name
            dest.mkdir(parents=True, exist_ok=True)
            canonicalize(f).save(dest / (f.stem + ".png"))
    return {"train": class_counts(config.TRAIN_DIR), "test": class_counts(config.TEST_DIR)}


def load_datasets(train_dir: Path = config.TRAIN_DIR, val_split: float = config.VAL_SPLIT,
                  batch_size: int = config.BATCH_SIZE):
    """Return (train_ds, val_ds, class_names) as batched tf.data pipelines (pixels 0-255)."""
    import tensorflow as tf

    common = dict(validation_split=val_split, seed=config.SEED, image_size=config.IMG_SIZE,
                  color_mode="grayscale", batch_size=batch_size, label_mode="int")
    train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, subset="training", **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(train_dir, subset="validation", **common)
    class_names = train_ds.class_names
    train_ds = train_ds.cache().shuffle(2000, seed=config.SEED).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, class_names


def load_test_dataset(test_dir: Path = config.TEST_DIR, class_names=None,
                      batch_size: int = config.BATCH_SIZE):
    import tensorflow as tf

    ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=config.IMG_SIZE, color_mode="grayscale", batch_size=batch_size,
        shuffle=False, label_mode="int", class_names=class_names)
    return ds.cache().prefetch(tf.data.AUTOTUNE)


def augmentation_layer():
    """Small random shifts, rotations and zooms (no flips: a flipped 2 is not a 2)."""
    import tensorflow as tf

    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomTranslation(0.12, 0.12),
        tf.keras.layers.RandomZoom(0.12),
    ], name="augmentation")


def preprocess_image(image) -> np.ndarray:
    """Bytes, path or PIL image -> float32 array (1, 32, 32, 1), pixels 0-255."""
    arr = np.asarray(canonicalize(image), dtype=np.float32)
    return arr[None, :, :, None]


# Features for visualisation

def image_features(path: Path) -> dict:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255
    ink = arr > 0.25
    rows, cols = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
    width = (cols[-1] - cols[0] + 1) if len(cols) else 0
    height = (rows[-1] - rows[0] + 1) if len(rows) else 0
    return {
        "ink_coverage": float(ink.mean()),                 # share of pixels that are stroke
        "stroke_intensity": float(arr[ink].mean()) if ink.any() else 0.0,
        "aspect_ratio": float(width / height) if height else 0.0,   # digit width / height
        "top_heaviness": float(arr[:16].sum() / max(arr.sum(), 1e-6)),  # ink in top half
    }


def dataset_features(folder: Path = config.TRAIN_DIR, max_per_class: int = 500) -> pd.DataFrame:
    rows = []
    for class_dir in sorted(p for p in Path(folder).iterdir() if p.is_dir()):
        for f in sorted(class_dir.iterdir())[:max_per_class]:
            if f.suffix.lower() in config.IMAGE_EXTS:
                rows.append({"label": class_dir.name, **image_features(f)})
    return pd.DataFrame(rows)


def mean_images(folder: Path = config.TRAIN_DIR, size: int = 16) -> dict:
    """The average image of each digit, as a size x size grid of 0-1 values."""
    out = {}
    for class_dir in sorted(p for p in Path(folder).iterdir() if p.is_dir()):
        arrs = [np.asarray(Image.open(f).convert("L").resize((size, size)), dtype=np.float32) / 255
                for f in class_dir.iterdir() if f.suffix.lower() in config.IMAGE_EXTS]
        if arrs:
            out[class_dir.name] = np.mean(arrs, axis=0).round(3).tolist()
    return out


# Upload handling for retraining

def _valid_label(label) -> str:
    label = str(label).strip()
    if label not in {str(d) for d in range(10)}:
        raise ValueError(f"'{label}' is not a digit. Use a class name from 0 to 9.")
    return label


def _save_image_bytes(data: bytes, label: str) -> bool:
    try:
        img = canonicalize(data)               # store uploads in the same format as training data
    except (UnidentifiedImageError, OSError, ValueError):
        return False
    dest = config.UPLOAD_DIR / _valid_label(label)
    dest.mkdir(parents=True, exist_ok=True)
    img.save(dest / f"{uuid.uuid4().hex[:12]}.png")
    return True


def save_uploads(files, label=None) -> dict:
    """files: list of (filename, bytes). Loose images need `label`; a .zip may hold digit folders."""
    saved, rejected = 0, 0
    for name, data in files:
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for member in z.infolist():
                    p = Path(member.filename)
                    if member.is_dir() or p.suffix.lower() not in config.IMAGE_EXTS or "__MACOSX" in p.parts:
                        continue
                    try:
                        ok = _save_image_bytes(z.read(member), p.parent.name or label)
                    except ValueError:
                        ok = False
                    saved, rejected = saved + ok, rejected + (not ok)
        elif Path(name).suffix.lower() in config.IMAGE_EXTS:
            if label is None:
                raise ValueError("Choose which digit these images show.")
            _valid_label(label)
            ok = _save_image_bytes(data, label)
            saved, rejected = saved + ok, rejected + (not ok)
        else:
            rejected += 1
    return {"saved": saved, "rejected": rejected, "pending": pending_upload_count()}


def pending_upload_count() -> int:
    return sum(class_counts(config.UPLOAD_DIR).values())


def merge_uploads_into_train(test_fraction: float = 0.2) -> dict:
    """Move pending uploads into train, with a share into test so evaluation covers them.
    A copy of each file is also kept in data/raw so the processed folders can be rebuilt."""
    moved = {}
    for class_dir in [p for p in config.UPLOAD_DIR.iterdir() if p.is_dir()]:
        files = [f for f in class_dir.iterdir() if f.suffix.lower() in config.IMAGE_EXTS]
        random.shuffle(files)
        n_test = max(1, int(len(files) * test_fraction)) if len(files) >= 5 else 0
        for i, f in enumerate(files):
            split = "test" if i < n_test else "train"
            target = (config.TEST_DIR if split == "test" else config.TRAIN_DIR) / class_dir.name
            raw_target = RAW_DIR / split / class_dir.name
            target.mkdir(parents=True, exist_ok=True)
            raw_target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, raw_target / f.name)
            shutil.move(str(f), target / f.name)
        moved[class_dir.name] = len(files)
        shutil.rmtree(class_dir, ignore_errors=True)
    return moved


def add_mnist(per_class: int = 600, test_per_class: int = 100) -> dict:
    """Add a sample of MNIST digits to data/raw so the model sees many more writing styles."""
    import tensorflow as tf

    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
    rng = np.random.default_rng(config.SEED)
    for split, X, Y, n in (("train", x_tr, y_tr, per_class), ("test", x_te, y_te, test_per_class)):
        for d in range(10):
            idx = rng.choice(np.where(Y == d)[0], n, replace=False)
            out = RAW_DIR / split / str(d)
            out.mkdir(parents=True, exist_ok=True)
            for i in idx:
                Image.fromarray(X[i]).save(out / f"mnist_{i}.png")
    return {"train": class_counts(RAW_DIR / "train"), "test": class_counts(RAW_DIR / "test")}