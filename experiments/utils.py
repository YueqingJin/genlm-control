
import datasets
import time, csv, signal, contextlib, io, sys, types, textwrap
from datasets import load_dataset
from pathlib import Path

# time record
@contextlib.contextmanager
def stopwatch():
    start = time.time()
    yield lambda: time.time() - start
# load mbpp
def load_mbpp(split: str = "validation"):
    ds = load_dataset("mbpp", split=split)
    return [
        dict(
            prompt=row["text"],
            tests=row["test_list"],
            reference=row["code"],
            task_id=row.get("task_id", i),
        )
        for i, row in enumerate(ds)
    ]

# test
def run_tests(pred_code: str, tests: list[str], func_name: str = "") -> bool:
    #tests one by one; Any anomaly is regarded as a failure
    module = types.ModuleType("__pred__")
    try:
        exec(pred_code, module.__dict__)
    except Exception:
        return False

    for t in tests:
        try:
            exec(textwrap.dedent(t), module.__dict__)
        except Exception:                             # A single test triggers any exception
            return False

    return True

# Result
def record_result(csv_path: str, row: dict):
    new = not Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if new:
            w.writeheader()
        w.writerow(row)
