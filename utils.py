import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path


def create_run_dir(prefix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"{timestamp}_{prefix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(run_dir, **configs):
    payload = {name: asdict(config) for name, config in configs.items()}
    path = Path(run_dir) / "config.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_csv_row(path, row, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if should_write_header:
            writer.writeheader()
        writer.writerow(row)
