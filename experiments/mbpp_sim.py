

import asyncio
import torch
from experiments.utils import load_mbpp, stopwatch, run_tests, record_result
import sys
sys.setrecursionlimit(10000)   # Raise the upper limit of recursion from 1000 to 10000
#simulation
import genlm.control.simulation as sim_mod
_orig_sim = sim_mod.simulated_completion_generate

# open simulation
import genlm.control.sampler.sequence as seq_mod
seq_mod.USE_SIM = True

# load model
from genlm.control.simulation import load_llama
MODEL, TOK = load_llama()

# generate
from experiments.mbpp_nosim import generate as _orig_generate

CSV_PATH = "sim_results.csv"
from genlm.control.sampler.sequence import SequenceModel as _SeqModelOrig
from numbers import Integral
SENTINEL = "\n### CODE START ###\n"

def _patched_init(self, *args, **kwargs):
    # convert the positional parameter tuple to a modifiable list
    args = list(args)

    # Find the entry point of max tokens
    if "max_tokens" in kwargs:
        mt = kwargs["max_tokens"]
        kw_place = True
    elif len(args) >= 3:              # SequenceModel
        mt = args[2]
        kw_place = False
    else:
        mt = None
        kw_place = False

    # if list[str | int] / str  / int integer
    if mt is not None:
        # str → int
        if isinstance(mt, str):
            mt = int(mt)

        # list[str | int] → list[int]
        if isinstance(mt, list):
            mt = [int(x) if isinstance(x, str) else x for x in mt]

            assert all(isinstance(x, Integral) and x > 0 for x in mt), \
                "max_tokens The list contains illegal elements"
            mt_single = max(mt)
        else:
            # single int
            assert isinstance(mt, Integral) and mt > 0, "max tokens must be positive integers"
            mt_single = mt

        # 4. mt_single back
        if kw_place:
            kwargs["max_tokens"] = mt_single
        else:
            args[2] = mt_single

    #  __init__
    _SeqModelOrig.__old_init__(self, *args, **kwargs)

if not hasattr(_SeqModelOrig, "__old_init__"):
    _SeqModelOrig.__old_init__ = _SeqModelOrig.__init__
    _SeqModelOrig.__init__ = _patched_init


async def main():
    split = load_mbpp("validation")
    sim_counts = []

    for task in split:
        sc = [0]

        # First accumulate, and then call the original completion
        def counted_sim(prefix, model, tokenizer, max_new_tokens=400):
            sc[0] += 1
            print(">>> called counted_sim, now =", sc[0])
            # f what is passed in is Shim, take the.model attribute of it
            hf_model = getattr(model, "model", model)
            return _orig_sim(prefix, hf_model, tokenizer, max_new_tokens)
        sim_mod.simulated_completion_generate = counted_sim

        print(f"=== Task {task['task_id']} ===")
        with stopwatch() as t:
            # best_code, n_sol, n_ok = await _orig_generate(
            #     task["prompt"]
            #     , 800
            # )
            prompt = task["prompt"] + SENTINEL  # 题目 + 哨兵 拼成新 prompt
            best_code, n_sol, n_ok = await _orig_generate(
                prompt,  # ← 用新的 prompt
                800
            )

        elapsed = t()
        passed = run_tests(best_code, task["tests"], "") if best_code.strip() else False
        record_result(CSV_PATH, {
            "task_id": task["task_id"],
            "pass1": int(passed),
            "time": elapsed,
            "n_solutions": n_sol,
            "n_syntax_ok": n_ok
        })
        sim_counts.append(sc[0])
        # Print the number of simulation completion calls for this question
        print(
            f"Task {task['task_id']} | sims={sc[0]} | "
            f"solutions={n_sol} | syntax_ok={n_ok} | pass={passed}"
        )
    # Restore
    sim_mod.simulated_completion_generate = _orig_sim

    # Print the average number of calls
    avg = sum(sim_counts) / len(sim_counts) if sim_counts else 0.0
    print(f"\nOn average, the simulation completion for each question was invoked {avg:.2f} ")


if __name__ == "__main__":
    asyncio.run(main())
