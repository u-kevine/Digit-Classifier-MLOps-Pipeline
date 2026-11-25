"""Single-image prediction. The Predictor reloads itself when a retrained model is
written to the shared models folder, so every container picks up new versions."""
import threading
import time

import numpy as np

from src import config
from src.model import read_json
from src.preprocessing import preprocess_image


class Predictor:
    def __init__(self):
        self._lock = threading.Lock()
        self.model = None
        self.class_names = []
        self.version = None
        self._loaded_mtime = None
        self.reload_if_updated()

    def reload_if_updated(self) -> bool:
        if not config.MODEL_PATH.exists():
            return False
        mtime = config.MODEL_PATH.stat().st_mtime
        if mtime == self._loaded_mtime:
            return False
        import tensorflow as tf
        with self._lock:
            if mtime == self._loaded_mtime:
                return False
            model = tf.keras.models.load_model(config.MODEL_PATH)
            meta = read_json(config.META_PATH)
            self.model, self.class_names = model, meta.get("class_names", [])
            self.version, self._loaded_mtime = meta.get("version"), mtime
        return True

    @property
    def ready(self) -> bool:
        return self.model is not None

    def predict(self, image) -> dict:
        """image: bytes, file path or PIL image."""
        self.reload_if_updated()
        if not self.ready:
            raise RuntimeError("No trained model found. Train one first.")
        start = time.perf_counter()
        x = preprocess_image(image)
        probs = self.model(x, training=False).numpy()[0]
        top = int(np.argmax(probs))
        return {
            "label": self.class_names[top],
            "confidence": float(probs[top]),
            "probabilities": {c: float(p) for c, p in zip(self.class_names, probs)},
            "model_version": self.version,
            "inference_ms": round((time.perf_counter() - start) * 1000, 2),
        }


_predictor = None


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


def predict_image(image) -> dict:
    return get_predictor().predict(image)


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(predict_image(sys.argv[1]), indent=2))