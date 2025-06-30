import asyncio, ast
from pathlib import Path

from experiments.utils import stopwatch, record_result, run_tests, load_mbpp
from genlm.control.simulation import load_llama
MODEL, TOK = load_llama()
# turn off
import genlm.control.sampler.sequence as seq_mod
seq_mod.USE_SIM = False
from genlm.control.potential import PromptedLLM
from genlm.control.sampler.token  import DirectTokenSampler
from genlm.control.sampler.sequence import SMC
from genlm.control.constant import EOS
from genlm.backend.tokenization.vocab import decode_vocab
from genlm.control.potential.built_in.json import JsonSchema
TOK.eos_token = "</s>"
TOK.eos_token_id = 2
MODEL.config.eos_token_id = 2


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

    # local
    schema = {
        "type": "object",
        "properties": {"return": {"type": "integer"}},
        "required": ["return"],
        "additionalProperties": False,
    }

    class LenientJSON(JsonSchema):
        async def prefix(self, context):
            try:
                return await super().prefix(context=context)
            except StopIteration:  #  StopIteration
                return 0.0
            except RuntimeError as e:  # RuntimeError
                if "StopIteration" in str(e):  # coroutine raised StopIteration
                    return 0.0
                raise
    critic = (
        LenientJSON(schema)
        .coerce(pllm, f=b"".join)  # int → bytes
    )
    #SMC
    seqs = await SMC(sampler, critic)(
        n_particles=80,
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
                       if not isinstance(ast.parse(p, mode="exec"), Exception))

    best = max(seqs.decoded_posterior,
               key=seqs.decoded_posterior.get,
               default="")

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
