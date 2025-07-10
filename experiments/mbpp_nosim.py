import asyncio, ast
from pathlib import Path
import sys
sys.setrecursionlimit(10000)
import urllib.request
import re
import pathlib
from experiments.utils import stopwatch, record_result, run_tests, load_mbpp
from genlm.control.simulation import load_llama
MODEL, TOK = load_llama()
# turn off
import genlm.control.sampler.sequence as seq_mod
# seq_mod.USE_SIM = False
from genlm.control.potential import PromptedLLM
from genlm.control.sampler.token  import DirectTokenSampler
from genlm.control.sampler.sequence import SMC
from genlm.control.constant import EOS
from genlm.backend.tokenization.vocab import decode_vocab
# from genlm.control.potential.built_in.wcfg import WCFG
from genlm.control.potential.built_in.wcfg import BoolCFG
from importlib import resources

# from genlm.control.potential.built_in.json import JsonSchema
TOK.eos_token = "</s>"
TOK.eos_token_id = 2
MODEL.config.eos_token_id = 2
# A function for cleaning up the redundant text in the model output
def clean_code_try_parse(p: str, max_strip: int = 10) -> str:
    """
    Try ast.parse line by line until it can be successfully parsed.
Discard at most the first max_strip row to prevent infinite loops.
    """
    lines = p.splitlines()
    for strip in range(0, min(max_strip, len(lines)) + 1):
        candidate = "\n".join(lines[strip:])
        try:
            ast.parse(candidate, mode="exec")
            return candidate  # If the parsing is successful, return
        except SyntaxError:
            continue
    # If all the previous max strip lines have been tried and still don't work, return to the original text
    return p

class Shim:
    def __init__(self, mdl, tok):
        self.model, self.tokenizer = mdl, tok
        self.device = next(mdl.parameters()).device
        self.byte_vocab, _ = decode_vocab(tok)
        vocab_size = len(self.byte_vocab)
        self.str_vocab = [""] * vocab_size
        self.str_vocab[2] = "</s>"

    async def next_token_logprobs(self, ids):
        import torch, torch.nn.functional as F
        t = torch.tensor([ids], device=self.device)
        logits = self.model(input_ids=t).logits[0, -1]
        return F.log_softmax(logits, dim=-1).detach()

    def generate(self, *a, **kw):
        return self.model.generate(*a, **kw)

# generate
async def generate(prompt: str,
                   max_tokens: int = 600
                  ) -> tuple[str, int, int]:

    # sampler
    llm = Shim(MODEL, TOK)
    eos_bytes = llm.byte_vocab[2]
    prefix_ids = TOK(prompt, return_tensors="pt").input_ids[0].tolist()
    pllm = PromptedLLM(
        llm,
        prefix_ids,
        eos_tokens=[eos_bytes]
    )
    pllm.tokenizer = TOK
    sampler = DirectTokenSampler(pllm)

    here = Path(__file__).parent  # experiments
    py_lark = (here / "python_simple.lark").read_text()

    # Only let CFG check the code after the sentinel
    SENTINEL = "\n### CODE START ###\n"  # It must be exactly the same as Document A
    SENT_BYTES = SENTINEL.encode("utf-8")

    def strip_prompt(chunks: list[bytes]) -> bytes:
        joined = b"".join(chunks)
        idx = joined.find(SENT_BYTES)  # Find the sentry
        if idx == -1:  # The Sentinel has not been fully generated yet
            return b""

        code = joined[idx + len(SENT_BYTES):]  # Only the part behind the sentry is left

        # If the code starts with ' ' ', discard this entire line
        if code.startswith(b"```"):
            nl = code.find(b"\n")  # Find the end of the line
            code = b"" if nl == -1 else code[nl + 1:]
        # ----------------------------------------------------
        if code.startswith(b"###"):
            nl = code.find(b"\n")
            code = b"" if nl == -1 else code[nl + 1:]
        if not code.startswith(b"\n"):
            code = b"\n" + code

        return code

    python_cfg = (
        BoolCFG.from_lark(py_lark)
        .coerce(pllm, f=strip_prompt)
    )
    critic = python_cfg
    #  Count the number of optional bytes before and after the CFG
    # Debugging patch: Print CFG filtering results in real time

    # Back up the original prefix first
    orig_prefix = python_cfg.prefix

    async def debug_allowed(ctx_bytes, step):
        """
Print the current context tail and the number of valid bytes after filtering
        """
        ok = [b for b in range(256)
              if await orig_prefix(ctx_bytes + [bytes([b])])]
        tail = b"".join(ctx_bytes[-20:])  # Take the last 20 bytes
        print(f"[step {step:04d}] allowed = {len(ok):3d} | ctx_tail = {tail!r}")

    async def patched_prefix(ctx):
        step = len(ctx)  # How many bytes have been generated = which step
        if step % 10 == 0:  # Print once every 10 steps; I want to remove this line at every step
            await debug_allowed(ctx, step)
        return await orig_prefix(ctx)  # Continue to use the original judgment

    python_cfg.prefix = patched_prefix

    seqs = await SMC(sampler, critic)(
        n_particles=5,
        ess_threshold=0.5,
        max_tokens=max_tokens
    )


    # total
    decoded_unique = {}
    for ctx, _ in seqs:
        if ctx and ctx[-1] is EOS:
            try:
                prog = b"".join(ctx[:-1]).decode("utf-8")
                decoded_unique[prog] = None
            except UnicodeDecodeError:
                pass

    n_solutions  = len(decoded_unique)
    n_syntax_ok  = sum(1 for p in decoded_unique
                       # Use a function to remove the redundant prefixes first, and then perform a syntax check
                       if not isinstance(ast.parse(clean_code_try_parse(p), mode="exec"), Exception))
    best = max(seqs.decoded_posterior,
               key=seqs.decoded_posterior.get,
               default="")
    # print(len(allowed), "bytes allowed at start")
    return best, n_solutions, n_syntax_ok


# main
async def main():
    split    = load_mbpp("validation")
    csv_path = "nosim_results.csv"

    for task in split:
        with stopwatch() as t:
            code, n_sol, n_ok = await generate(task["prompt"])
            passed = run_tests(code, task["tests"], "") if code.strip() else False

        record_result(csv_path, {
            "task_id": task["task_id"],
            "pass1":  int(passed),
            "time":   t(),
            "n_solutions": n_sol,
            "n_syntax_ok": n_ok
        })
        print(f"Task {task['task_id']:>3} | solutions={n_sol:2d} "
              f"| syntax_ok={n_ok:2d} | pass={passed}")

if __name__ == "__main__":
    asyncio.run(main())
