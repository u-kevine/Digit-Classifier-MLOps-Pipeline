"""Model creation, training, evaluation and retraining for the digit CNN.
CLI:
    python -m src.model train     
    python -m src.model retrain   
    python -m src.model evaluate   
"""
import json
import shutil
import sys
import time
from datetime import datetime, timezone

import numpy as np

from src import config
from src.preprocessing import (augmentation_layer, class_counts, load_datasets,
                               load_test_dataset, merge_uploads_into_train,
                               pending_upload_count)



# Metadata and shared retraining status (JSON files on the models volume)

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default if default is not None else {}


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=float))
    tmp.replace(path)                      


def acquire_retrain_lock() -> bool:
    """Create the lock file atomically. Only one process (in any container sharing the models
    volume) can hold it, so two replicas can never retrain at the same time."""
    import os
    path = config.RETRAIN_LOCK_PATH
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            age = time.time() - path.stat().st_mtime
        except FileNotFoundError:
            return acquire_retrain_lock()
        if age < config.RETRAIN_LOCK_MAX_AGE_S:
            return False
        path.unlink(missing_ok=True)          
        return acquire_retrain_lock()
    with os.fdopen(fd, "w") as fh:
        fh.write(_now())
    return True


def release_retrain_lock():
    config.RETRAIN_LOCK_PATH.unlink(missing_ok=True)


def retrain_running() -> bool:
    return config.RETRAIN_LOCK_PATH.exists()


def read_status() -> dict:
    return read_json(config.STATUS_PATH, {"state": "idle"})


def write_status(**fields):
    status = read_status()
    status.update(fields, updated_at=_now())
    write_json(config.STATUS_PATH, status)


# 3. Model creation

def build_model(num_classes: int = 10, learning_rate: float = 1e-3):
    """A small CNN: two convolution blocks, then a dense classifier."""
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    inputs = tf.keras.Input(shape=config.IMG_SIZE + (config.CHANNELS,), name="image")
    x = augmentation_layer()(inputs)
    x = layers.Rescaling(1.0 / 255, name="scale_0_1")(x)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)                      # 32 -> 16
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)                      # 16 -> 8
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="digit")(x)

    model = tf.keras.Model(inputs, outputs, name="digit_cnn")
    compile_model(model, learning_rate)
    return model


def compile_model(model, learning_rate: float):
    import tensorflow as tf

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])


def get_callbacks(patience: int = 5):
    import tensorflow as tf
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]


def history_dict(*histories) -> dict:
    merged = {}
    for h in histories:
        for k, v in h.history.items():
            merged.setdefault(k, []).extend(float(x) for x in v)
    return merged


def train_model(epochs: int = 40, save: bool = True, trigger: str = "initial"):
    train_ds, val_ds, class_names = load_datasets()
    model = build_model(len(class_names))
    h = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=get_callbacks(), verbose=2)
    history = history_dict(h)
    metrics = evaluate_model(model, class_names=class_names)
    if save:
        save_model(model, class_names, metrics, history, trigger=trigger)
    return model, history, metrics


# 4. Model testing

def predict_dataset(model, ds):
    y_true, y_prob = [], []
    for images, labels in ds:
        y_prob.append(model.predict(images, verbose=0))
        y_true.append(labels.numpy())
    return np.concatenate(y_true), np.concatenate(y_prob)


def evaluate_model(model, test_ds=None, class_names=None) -> dict:
    """Accuracy, precision, recall, F1 (macro and weighted), log loss, ROC-AUC, confusion matrix."""
    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                                 log_loss, precision_recall_fscore_support, roc_auc_score)

    if test_ds is None:
        test_ds = load_test_dataset(class_names=class_names)
    y_true, y_prob = predict_dataset(model, test_ds)
    y_pred = y_prob.argmax(axis=1)
    labels = list(range(len(class_names)))
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", labels=labels)
    except ValueError:
        auc = None
    return {
        "n_test": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": p_m, "recall_macro": r_m, "f1_macro": f_m,
        "precision_weighted": p_w, "recall_weighted": r_w, "f1_weighted": f_w,
        "log_loss": log_loss(y_true, y_prob, labels=labels),
        "roc_auc_ovr": auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                           output_dict=True, zero_division=0),
    }


def load_current_model():
    import tensorflow as tf
    return tf.keras.models.load_model(config.MODEL_PATH)


def copy_weights(src, dst):
    """Copy weights layer by layer into a freshly built model (warm start)."""
    for ls, ld in zip(src.layers, dst.layers):
        if getattr(ls, "layers", None):
            copy_weights(ls, ld)
        elif ls.weights:
            ld.set_weights(ls.get_weights())


def save_model(model, class_names, metrics, history, trigger="manual", extra=None):
    meta = read_json(config.META_PATH, {"versions": []})
    if config.MODEL_PATH.exists():
        shutil.copy2(config.MODEL_PATH, config.ARCHIVE_DIR / f"digit_cnn_v{meta.get('version', 0)}.h5")
    version = meta.get("version", 0) + 1
    tmp = config.MODEL_DIR / "digit_cnn.tmp.h5"
    model.save(tmp)
    tmp.replace(config.MODEL_PATH)
    meta.update({
        "version": version, "class_names": list(class_names), "trained_at": _now(),
        "metrics": metrics, "history": history,
        "train_counts": class_counts(config.TRAIN_DIR), "test_counts": class_counts(config.TEST_DIR),
    })
    meta["versions"].append({
        "version": version, "trained_at": meta["trained_at"], "trigger": trigger,
        "accuracy": metrics["accuracy"], "f1_macro": metrics["f1_macro"],
        "n_train": sum(meta["train_counts"].values()), **(extra or {}),
    })
    write_json(config.META_PATH, meta)
    return version


# 5. Retraining

def should_retrain() -> bool:
    """Trigger condition: enough new labelled digits have been uploaded."""
    return pending_upload_count() >= config.RETRAIN_THRESHOLD


def retrain(trigger: str = "manual", epochs: int = 10) -> dict:
    """Merge uploads into the training set, fine-tune the current model, and promote the
    new model unless its test accuracy is more than PROMOTION_TOLERANCE below the current
    model's on the same test set."""
    if not acquire_retrain_lock():
        return {"state": "running", "message": "A retraining job is already running."}
    start = time.time()
    write_status(state="running", trigger=trigger, started_at=_now(), message="Merging uploads")
    try:
        moved = merge_uploads_into_train()
        train_ds, val_ds, class_names = load_datasets()
        test_ds = load_test_dataset(class_names=class_names)

        old_metrics = None
        model = build_model(len(class_names))
        if config.MODEL_PATH.exists():
            write_status(message="Scoring the current model")
            old_model = load_current_model()
            old_metrics = evaluate_model(old_model, test_ds, class_names)
            copy_weights(old_model, model)                 
            compile_model(model, learning_rate=3e-4)       
        write_status(message="Training on old and new data")
        h = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=get_callbacks(3), verbose=2)

        write_status(message="Evaluating retrained model")
        new_metrics = evaluate_model(model, test_ds, class_names)
        promoted = (old_metrics is None or
                    new_metrics["accuracy"] >= old_metrics["accuracy"] - config.PROMOTION_TOLERANCE)
        result = {
            "trigger": trigger, "new_images": moved, "promoted": promoted,
            "old_accuracy": old_metrics and old_metrics["accuracy"],
            "new_accuracy": new_metrics["accuracy"], "new_f1_macro": new_metrics["f1_macro"],
            "duration_s": round(time.time() - start, 1),
        }
        if promoted:
            result["version"] = save_model(model, class_names, new_metrics, history_dict(h),
                                           trigger, extra={"new_images": sum(moved.values())})
        write_status(state="done", finished_at=_now(), result=result,
                     message="New model deployed" if promoted
                     else "Kept current model (new one scored more than the allowed margin lower)")
        return result
    except Exception as exc:  # noqa: BLE001
        write_status(state="failed", finished_at=_now(), message=str(exc))
        raise
    finally:
        release_retrain_lock()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        _, _, m = train_model()
        print(json.dumps({k: v for k, v in m.items() if k not in ("per_class", "confusion_matrix")}, indent=2))
    elif cmd == "retrain":
        print(json.dumps(retrain(trigger="cli"), indent=2))
    elif cmd == "evaluate":
        meta = read_json(config.META_PATH)
        m = evaluate_model(load_current_model(), class_names=meta["class_names"])
        print(json.dumps({k: v for k, v in m.items() if k != "per_class"}, indent=2))