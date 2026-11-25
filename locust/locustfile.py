"""Flood the API with prediction requests."""
import io
import random
from pathlib import Path

from locust import HttpUser, between, task

TEST_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "test"
EXTS = {".jpg", ".jpeg", ".png"}


def _load_images(limit=60):
    files = [p for p in TEST_DIR.rglob("*") if p.suffix.lower() in EXTS]
    random.shuffle(files)
    images = [(p.name, p.read_bytes()) for p in files[:limit]]
    if not images:  # no local data: send synthetic images so the test still runs
        from PIL import Image
        for i in range(10):
            buf = io.BytesIO()
            Image.new("L", (64, 64), random.randint(0, 255)).save(buf, "PNG")
            images.append((f"synthetic_{i}.png", buf.getvalue()))
    return images


IMAGES = _load_images()


class PredictionUser(HttpUser):
    wait_time = between(0.5, 2)

    @task(10)
    def predict(self):
        name, data = random.choice(IMAGES)
        self.client.post("/predict", files={"file": (name, data, "image/png")}, name="/predict")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
