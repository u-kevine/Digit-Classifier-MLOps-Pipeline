# Handwritten Digit Classifier: MLOps Pipeline

| **Video demo** |  |
| **Live app** | http://51.20.14.115:8000 |
| **API docs** | http://51.20.14.115:8000/docs |


## Project description

Many clinics and community health workers still record patient information by hand: ages, weights,
temperatures and doses written in paper registers, vaccination cards and lab forms. Turning those
numbers into digital records is slow and error-prone. This project builds the first component of a
tool that reads handwritten numbers: a model that recognises a single handwritten digit (0–9) from an image.

Because a misread dose is dangerous, the model is designed to assist, not replace, a health worker:
every prediction comes with a confidence score so uncertain readings can be checked by a person.

The project covers the full machine learning lifecycle: data acquisition and preprocessing, a
convolutional neural network, evaluation, a REST API with a monitoring dashboard, retraining on newly
uploaded data, deployment in Docker containers behind a load balancer on AWS, and load testing with Locust.

## Dataset

- **UCI / scikit-learn handwritten digits:** 1,797 images at 8×8 pixels.
- **MNIST sample:** 600 training and 100 test images per digit, added so the model learns many more writing styles.
- **Total:** 7,437 training images and 1,360 test images, balanced at about 740 and 136 per digit.

## Features

- **Prediction:** draw a digit or upload a photo of one; the model returns the digit and all ten probabilities.
- **Visualisations:** class balance, digit width and the average image of each digit, each with an interpretation.
- **Model up-time:** live uptime clock, response-time chart and the container that answered.
- **Bulk upload:** many images of one digit, or a .zip with folders named 0–9.
- **Retraining:** a button, plus an automatic trigger once 50 new images are uploaded. A retrained model
  only goes live if it scores at least as well as the current one on the test set.
- **Evaluation in production:** re-score the live model on the test set at any time; every version is
  listed with its accuracy.

## Repository structure

```text
├── README.md
├── notebook/digit_classifier.ipynb    preprocessing, training, evaluation, prediction, retraining
├── src/
│   ├── config.py                      paths and settings
│   ├── preprocessing.py               data acquisition, canonicalize(), data pipelines, uploads
│   ├── model.py                       CNN, evaluation, retraining and its trigger
│   ├── prediction.py                  single-image prediction
│   ├── api.py                         FastAPI app
│   └── static/index.html              dashboard
├── data/raw, data/train, data/test    images
├── models/digit_cnn.h5                trained model
├── locust/                            load test and results
├── scripts/summarize_locust.py        builds the load-test table and chart
├── deploy/nginx.conf                  load balancer
├── Dockerfile, docker-compose.yml
└── requirements.txt, requirements-dev.txt
```

## Setup

### Run locally (Python 3.10–3.12)

```bash
git clone https://github.com/u-kevine/Digit-Classifier-MLOps-Pipeline.git
cd Digit-Classifier-MLOps-Pipeline
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn src.api:app --port 8000          
```

### Retrain the model from scratch

Run `notebook/digit_classifier.ipynb` top to bottom, or `python -m src.model train`.

### Run with Docker (load-balanced)

```bash
docker compose up -d --build
docker compose up -d --scale api=3 && docker compose restart nginx
```

### Deploy on AWS

1. Launch an EC2 instance: Ubuntu 24.04, m7i-flex.large (free-tier eligible), 30 GB disk.
2. Open TCP port 8000 in its security group.
3. Install Docker: `curl -fsSL https://get.docker.com | sudo sh`
4. Clone this repo and run `docker compose up -d --build`.

## Model evaluation (1,360 test images)

| Metric | Value |
|---|---|
| Accuracy | 97.7% (1,329 / 1,360) |
| Precision (macro) | 97.4% |
| Recall (macro) | 97.4% |
| F1 (macro) | 97.4% |
| ROC-AUC (one-vs-rest) | 0.998–1.000 per digit |

Digit 8 is the hardest (126 of 135 correct), mostly confused with 3 and 9. The live site has since been
retrained with new uploads; see the version table on the dashboard.

## Flood request simulation (Locust)

100 simulated users, spawn rate 10 per second, 2 minutes per run, against the AWS deployment.
Each API container is limited to 0.5 CPU and 1.5 GB memory on a 2-vCPU instance.

|   Containers |   Requests |   Failures |   Avg (ms) |   Median (ms) |   p95 (ms) |   p99 (ms) |   Max (ms) |   Req/s |
|-------------:|-----------:|-----------:|-----------:|--------------:|-----------:|-----------:|-----------:|--------:|
|            1 |       3077 |         93 |       2268 |          2700 |       4100 |       4600 |       5525 |    25.8 |
|            2 |       4568 |          0 |       1086 |           650 |       2600 |       3000 |       3927 |    38.3 |
|            4 |       7229 |          0 |        202 |           180 |        380 |        600 |       1140 |    60.7 |

![Load test results](locust/results/load_test_summary.png)

**Observations:** Adding containers made the API dramatically faster and more reliable under the
same load. With **1 container** (0.5 CPU), requests queued up: the median response took 2.7 s,
the slowest 5% took over 4 s, and 93 requests (3%) failed because they waited too long to be
answered. With **2 containers**, the median dropped to 650 ms, failures fell to zero and throughput
rose from 25.8 to 38.3 requests per second. With **4 containers**, using the instance's full
2 vCPUs, the median fell to 180 ms (15× faster than one container), 95% of requests finished
within 380 ms, and throughput reached 60.7 requests per second. At that point the API kept up with
everything the 100 simulated users sent: each user pauses 0.5–2 s between requests, which caps
demand at roughly 65–70 requests per second, so the remaining limit is the test load rather than
the server. Since 4 containers already use all of the instance's CPU, scaling further would need a
larger instance or more instances behind the load balancer.

## Data credits

- UCI Optical Recognition of Handwritten Digits, via scikit-learn.
- MNIST database of handwritten digits (LeCun, Cortes and Burges).