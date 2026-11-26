"""Turn Locust CSVs (locust/results/c<N>_stats.csv) into a markdown table and a chart."""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "locust" / "results"
rows = []
for f in sorted(RESULTS.glob("c*_stats.csv"), key=lambda p: int(re.findall(r"\d+", p.name)[0])):
    n = int(re.findall(r"\d+", f.name)[0])
    df = pd.read_csv(f)
    r = df[df["Name"] == "/predict"].iloc[0]
    rows.append({
        "Containers": n,
        "Requests": int(r["Request Count"]),
        "Failures": int(r["Failure Count"]),
        "Avg (ms)": round(r["Average Response Time"]),
        "Median (ms)": round(r["Median Response Time"]),
        "p95 (ms)": round(r["95%"]),
        "p99 (ms)": round(r["99%"]),
        "Max (ms)": round(r["Max Response Time"]),
        "Req/s": round(r["Requests/s"], 1),
    })

if not rows:
    raise SystemExit("No results found in locust/results.")

table = pd.DataFrame(rows)
md = table.to_markdown(index=False)
(RESULTS / "summary.md").write_text(md + "\n")
print(md)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(table["Containers"], table["Median (ms)"], "o-", label="Median")
ax[0].plot(table["Containers"], table["p95 (ms)"], "s-", label="p95")
ax[0].set(xlabel="Containers", ylabel="Latency (ms)", title="/predict latency", xticks=table["Containers"])
ax[0].legend()
ax[1].bar(table["Containers"].astype(str), table["Req/s"], color="#1E5B4F")
ax[1].set(xlabel="Containers", ylabel="Requests / second", title="Throughput")
fig.tight_layout()
fig.savefig(RESULTS / "load_test_summary.png", dpi=150)
print("Saved", RESULTS / "load_test_summary.png")
