"""REST API + dashboard. """
import os
import socket
import threading
import time
from pathlib import Path
from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src import config
from src.model import (evaluate_model, read_json, read_status, retrain, should_retrain)
from src.prediction import get_predictor
from src.preprocessing import (class_counts, dataset_features, mean_images,
                               pending_upload_count, save_uploads)

STATIC = Path(__file__).parent / "static"
START_TIME = time.time()
HOSTNAME = socket.gethostname()        # shows which container answered (load balancing)
_counter = {"requests": 0, "predictions": 0}
_stats_cache = {"key": None, "data": None}

app = FastAPI(title="Digit Classifier", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def count_requests(request, call_next):
    _counter["requests"] += 1
    response = await call_next(request)
    response.headers["X-Served-By"] = HOSTNAME
    return response


@app.on_event("startup")
def warm_up():
    get_predictor()        


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(STATIC / "index.html")


# ----------------- uptime ----------------------#
@app.get("/health")
def health():
    p = get_predictor()
    return {
        "status": "ok" if p.ready else "no_model",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "started_at": START_TIME,
        "container": HOSTNAME,
        "model_version": p.version,
        "requests_served": _counter["requests"],
        "predictions_served": _counter["predictions"],
        "retraining": read_status().get("state"),
    }


# ------------------ prediction -----------------#
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    try:
        result = await run_in_threadpool(get_predictor().predict, data)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception:
        raise HTTPException(400, "Could not read that file as an image.")
    _counter["predictions"] += 1
    result["container"] = HOSTNAME
    return result


# --------------------- data & stats -------------#
@app.get("/stats")
def stats():
    """Feature summaries for the dashboard charts (cached until the data changes)."""
    counts = {"train": class_counts(config.TRAIN_DIR), "test": class_counts(config.TEST_DIR),
              "uploads": class_counts(config.UPLOAD_DIR)}
    key = str(counts["train"])
    if _stats_cache["key"] != key:
        df = dataset_features(config.TRAIN_DIR, max_per_class=200)
        if df.empty:
            _stats_cache["data"] = {}
        else:
            g = df.groupby("label")
            _stats_cache["data"] = {
                "ink_coverage": {k: v.round(3).tolist() for k, v in g["ink_coverage"]},
                "aspect_ratio": {k: v.round(3).tolist() for k, v in g["aspect_ratio"]},
                "top_heaviness_mean": g["top_heaviness"].mean().round(3).to_dict(),
                "mean_images": mean_images(config.TRAIN_DIR),
            }
        _stats_cache["key"] = key
    return {"counts": counts, "features": _stats_cache["data"]}


@app.get("/metrics")
def metrics():
    meta = read_json(config.META_PATH)
    if not meta:
        raise HTTPException(404, "No trained model yet.")
    return meta


@app.post("/upload")
async def upload(background: BackgroundTasks, files: List[UploadFile] = File(...),
                 label: Optional[str] = Form(None)):
    """Bulk upload: many images with one class label, or a .zip with class sub-folders."""
    payload = [(f.filename, await f.read()) for f in files]
    try:
        result = await run_in_threadpool(save_uploads, payload, label)
    except ValueError as e:
        raise HTTPException(400, str(e))
    result["auto_retrain_started"] = False
    if config.AUTO_RETRAIN and should_retrain() and read_status().get("state") != "running":
        background.add_task(_run_retrain, "auto: upload threshold reached")
        result["auto_retrain_started"] = True
    result["threshold"] = config.RETRAIN_THRESHOLD
    return result


@app.get("/uploads")
def uploads():
    return {"pending": pending_upload_count(), "by_class": class_counts(config.UPLOAD_DIR),
            "threshold": config.RETRAIN_THRESHOLD, "auto_retrain": config.AUTO_RETRAIN}


# --------------------- retraining --------------------------------#
def _run_retrain(trigger: str):
    try:
        retrain(trigger=trigger)
    except Exception as e:  # status file already records the failure
        print("Retraining failed:", e)


@app.post("/retrain")
def trigger_retrain(background: BackgroundTasks):
    if read_status().get("state") == "running":
        raise HTTPException(409, "Retraining is already running.")
    if pending_upload_count() == 0 and os.getenv("ALLOW_EMPTY_RETRAIN", "false") != "true":
        raise HTTPException(400, "Upload new images before retraining.")
    background.add_task(_run_retrain, "manual: dashboard button")
    return {"state": "started"}


@app.get("/retrain/status")
def retrain_status():
    return read_status()


# --------------------- evaluation in production -------------------#
_eval_lock = threading.Lock()


@app.post("/evaluate")
async def evaluate():
    """Re-score the live model on the held-out test set (includes new data merged in)."""
    p = get_predictor()
    if not p.ready:
        raise HTTPException(503, "No trained model found.")
    if not _eval_lock.acquire(blocking=False):
        raise HTTPException(409, "An evaluation is already running.")
    try:
        m = await run_in_threadpool(evaluate_model, p.model, None, p.class_names)
    finally:
        _eval_lock.release()
    m["model_version"] = p.version
    return m