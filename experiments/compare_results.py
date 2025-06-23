import csv
import pathlib

NOSIM_CSV = pathlib.Path("nosim_results.csv")
SIM_CSV   = pathlib.Path("sim_results.csv")

def ensure_header(path: pathlib.Path):
    """If the first line of the file is not a header, insert the header at the very beginning"""
    text = path.read_text().splitlines()
    if text and ("task_id" not in text[0] or "pass1" not in text[0]):
        new = ["task_id,pass1,time"] + text
        path.write_text("\n".join(new) + "\n")

def load(csv_path: pathlib.Path):
    rows = []
    ensure_header(csv_path)
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((int(r["pass1"]), float(r["time"])))
    return rows

def summarize(rows):
    n_total = len(rows)
    n_pass  = sum(p for p, _ in rows)
    tot_t   = sum(t for _, t in rows)
    return dict(pass1=n_pass / n_total if n_total else 0, tot=tot_t)

def main():
    nosim = summarize(load(NOSIM_CSV))
    sim   = summarize(load(SIM_CSV))

    print("\nMBPP 90-questions comparison")
    print(f"without simulation  pass = {nosim['pass1']:.1%}   time = {nosim['tot']:.1f}s")
    print(f"with simulation     pass = {sim['pass1']:.1%}   time = {sim['tot']:.1f}s")

if __name__ == "__main__":
    main()

