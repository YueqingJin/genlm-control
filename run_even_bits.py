# import asyncio
# import ast
#
# from genlm.control.potential import PromptedLLM, Potential
# # from genlm.control.potential.testing import UnitTestPotential
# # from genlm.control.potential.grammar import PythonGrammar
# # from genlm.grammar.python import PythonGrammar
#
#
# # from genlm.control.potential import Coerce
# from genlm.control.sampler.token import DirectTokenSampler
# from genlm.control.sampler.sequence import SMC
#
#
# async def main():
#     # 1) 提示和基础模型
#     prompt = "Write a python function to set all even bits of a given number."
#     p = PromptedLLM(prompt, model_name="gpt2")
#
#     # 2) 局部势 Φ_eff：强制 Python 语法合法
#     # grammar = PythonGrammar.from_builtin()
#     # eff = Coerce(p.token_type, grammar) * p
#     eff = p
#
#     # 3a) 语义全局势：单测断言
#     tests = [
#         "assert even_bit_set_number(10) == 10",
#         "assert even_bit_set_number(20) == 30",
#         "assert even_bit_set_number(30) == 30",
#     ]
#     glob_tests = UnitTestPotential(
#         module_path="genlm.control.simulation_utils",
#         test_expressions=tests,
#     )
#
#     # 3b) 语法全局势：AST 解析检查
#     class SyntaxOK(Potential):
#         async def score(self, tokens):
#             src = self.decode(tokens)
#             try:
#                 ast.parse(src)
#                 return 0.0
#             except SyntaxError:
#                 return float('-inf')
#
#     glob_syntax = SyntaxOK()
#
#     # 3c) 组合全局势：单测 + 语法
#     glob = glob_tests * glob_syntax
#
#     # 4) 构造局部 TokenSampler
#     sampler = DirectTokenSampler(eff)
#
#     # 5) 运行 SMC（含早期模拟补全 + 全局势）
#     sequences = await SMC(sampler, critic=glob)(
#         n_particles=20,
#         ess_threshold=0.5,
#         max_tokens=100,
#     )
#
#     # 6) 输出 top 候选
#     print("--- Top candidates ---")
#     for code, logw in sequences.decoded_posterior.items():
#         print(f"[weight={logw:.3f}]")
#         print(code)
#         print()
#
#
# if __name__ == "__main__":
#     asyncio.run(main())




#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# coding: utf-8
# """
# SMC 示例：自动生成并单元测试 even_bit_set_number
# ──────────────────────────────────────────────
# • 局部 target  = PromptedLLM                （不做语法过滤）
# • 全局 critic  = SyntaxOK × UnitTestPotential（语法 + 语义校验）
# """
#
# import asyncio, ast, sys, types, textwrap
# from genlm.control.simulation import load_gpt2
# from genlm.control.potential import PromptedLLM, Potential
# from genlm.control.sampler.token import DirectTokenSampler
# from genlm.control.sampler.sequence import SMC
#
#
# # ─────────────────── 1. 语义势：单元测试 ───────────────────
# class UnitTestPotential(Potential):
#     def __init__(self, module_name: str, tests: list[str], tok):
#         super().__init__(vocabulary=[0])          # 任意非空占位即可
#         self.module_name, self.tests, self.tok = module_name, tests, tok
#
#     async def prefix(self, t):   return 0.0
#     async def complete(self, t): return await self.score(t)
#
#     async def score(self, tokens):
#         ids  = [x for x in tokens if isinstance(x, int)]
#         code = self.tok.decode(ids, skip_special_tokens=True)
#
#         # 把生成的源码放进一个临时模块里跑
#         mod = types.ModuleType(self.module_name)
#         try:
#             exec(code, mod.__dict__)
#         except Exception:
#             return float("-inf")
#
#         sys.modules[self.module_name] = mod
#         try:
#             for expr in self.tests:
#                 exec(textwrap.dedent(expr), mod.__dict__)
#         except AssertionError:
#             return float("-inf")
#         finally:
#             sys.modules.pop(self.module_name, None)
#
#         return 0.0
#
#
# # ─────────────────── 2. 语法势 ──────────────────────────────
# class SyntaxOK(Potential):
#     def __init__(self, tok):
#         super().__init__(vocabulary=[0])
#         self.tok = tok
#
#     async def prefix(self, t):   return 0.0
#     async def complete(self, t): return await self.score(t)
#
#     async def score(self, tokens):
#         try:
#             code = self.tok.decode([i for i in tokens if isinstance(i, int)],
#                                    skip_special_tokens=True)
#             ast.parse(code)
#             return 0.0
#         except SyntaxError:
#             return float("-inf")
#
#
# # ─────────────────── 3. 主流程 ──────────────────────────────
# async def main():
#     # 3-1 载入 GPT-2
#     model, tok = load_gpt2()
#
#     # 3-2 LLMShim：适配 llm.byte_vocab / .next_token_logprobs
#     class Shim:
#         def __init__(self, mdl, tokenizer):
#             self.model, self.tokenizer = mdl, tokenizer
#             self.byte_vocab = {tid: txt.encode("utf-8", "ignore")
#                                for txt, tid in tokenizer.get_vocab().items()}
#             self.device = next(mdl.parameters()).device
#
#         async def next_token_logprobs(self, ids):
#             import torch, torch.nn.functional as F
#             t = torch.tensor([ids], device=self.device)
#             return F.log_softmax(self.model(t).logits[0, -1], dim=-1).detach()
#
#         def generate(self, *a, **kw): return self.model.generate(*a, **kw)
#
#     llm = Shim(model, tok)
#
#     # 3-3 PromptedLLM
#     prompt = (
#         "Implement the Python function below.\n\n"
#         "def even_bit_set_number(n: int) -> int:\n"
#         '    """Set every even-indexed bit (0-based) of n to 1 and return the result."""\n'
#     )
#     p_ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
#     pllm  = PromptedLLM(llm, p_ids, eos_tokens=[tok.eos_token_id])
#
#     # ★ 补 tokenizer，供 DirectTokenSampler 使用
#     pllm.tokenizer = tok
#
#     # 3-4 构造局部 sampler（这里只用 LLM 本身）
#     sampler = DirectTokenSampler(pllm)
#
#     # 3-5 构造全局 critic = 语法检查 × 单元测试
#     tests = [
#         "assert even_bit_set_number(10) == 10",
#         "assert even_bit_set_number(20) == 30",
#         "assert even_bit_set_number(30) == 30",
#     ]
#     critic = SyntaxOK(tok) * UnitTestPotential("tmp_mod", tests, tok)
#
#     # 3-6 运行 SMC
#     seqs = await SMC(sampler, critic=critic)(
#         n_particles=10,
#         ess_threshold=0.5,
#         max_tokens=80,
#         verbosity=1,
#     )
#
#     # 3-7 输出
#     if seqs.decoded_posterior:
#         print(f"\n找到 {len(seqs.decoded_posterior)} 个通过全部测试的候选：")
#         for code, w in seqs.decoded_posterior.items():
#             print(f"\n[weight={w:.3f}]\n{code}")
#     else:
#         print("\n没有任何候选通过断言 :(")
#
#
# if __name__ == "__main__":
#     asyncio.run(main())
# run_even_bits.py  （与 genlm-control 同级）
import asyncio, ast, sys, textwrap, types, torch
from genlm.control.simulation import load_gpt2, simulated_completion_generate
from genlm.control.potential import PromptedLLM, Potential
from genlm.control.sampler.token import DirectTokenSampler
from genlm.control.sampler.sequence import SMC

# ─── 1. 单元测试势 ───────────────────────────────────────────────
class UnitTestPotential(Potential):
    def __init__(self, mod_name: str, tests: list[str], tok):
        super().__init__(vocabulary=[0])                        # 占位
        self.mod_name, self.tests, self.tok = mod_name, tests, tok

    async def prefix(self, t):   return 0.0
    async def complete(self, t): return await self.score(t)

    async def score(self, tokens):
        code = self.tok.decode([x for x in tokens if isinstance(x, int)],
                               skip_special_tokens=True)
        mod  = types.ModuleType(self.mod_name)
        try:
            exec(code, mod.__dict__)
        except Exception:
            return float("-inf")
        sys.modules[self.mod_name] = mod
        try:
            for expr in self.tests:
                exec(textwrap.dedent(expr), mod.__dict__)
        except AssertionError:
            return float("-inf")
        finally:
            sys.modules.pop(self.mod_name, None)
        return 0.0

# ─── 2. 语法检查势 ──────────────────────────────────────────────
class SyntaxOK(Potential):
    def __init__(self, tok):
        super().__init__(vocabulary=[0])                        # 占位
        self.tok = tok

    async def prefix(self, t):   return 0.0
    async def complete(self, t): return await self.score(t)

    async def score(self, tokens):
        try:
            code = self.tok.decode([i for i in tokens if isinstance(i, int)],
                                   skip_special_tokens=True)
            ast.parse(code)
            return 0.0
        except SyntaxError:
            return float("-inf")

# ─── 3. 主流程 ──────────────────────────────────────────────────
async def main():
    model, tok = load_gpt2()

    # 3-1 LLM “垫片”
    class Shim:
        def __init__(self, mdl, tokenizer):
            self.model, self.tokenizer = mdl, tokenizer
            self.device = next(mdl.parameters()).device
            # byte_vocab & 反查表
            self.byte_vocab = {tid: txt.encode("utf-8", "ignore")
                               for txt, tid in tokenizer.get_vocab().items()}
            self.rev_byte_vocab = {v: k for k, v in self.byte_vocab.items()}

        # ---- utils ---------------------------------------------------------
        def _to_ids(self, seq):
            ids = []
            for t in seq:
                if isinstance(t, int):
                    ids.append(t)
                elif isinstance(t, bytes):
                    ids.append(self.rev_byte_vocab[t])
                elif torch.is_tensor(t):
                    ids.extend(t.view(-1).tolist())
                else:
                    raise TypeError(f"Unexpected token type {type(t)}")
            return ids

        # ---- 必要接口 -------------------------------------------------------
        async def next_token_logprobs(self, ids):
            import torch.nn.functional as F
            t = torch.tensor([ids], device=self.device)
            logits = self.model(t).logits[0, -1]
            return F.log_softmax(logits, dim=-1).detach()

        def generate(self, input_ids, **kw):
            tensor_in = torch.tensor([self._to_ids(input_ids)], device=self.device)
            return self.model.generate(tensor_in, **kw)

    llm = Shim(model, tok)

    # 3-2 PromptedLLM
    prompt = (
        "Implement the Python function below.\n\n"
        "def even_bit_set_number(n: int) -> int:\n"
        '    """Set every even-indexed bit (0-based) of n to 1 and return the result."""\n'
    )
    p_ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
    pllm  = PromptedLLM(llm, p_ids, eos_tokens=[tok.eos_token_id])
    pllm.tokenizer = tok                      # DirectTokenSampler

    # 3-3 Sampler & Critic
    sampler = DirectTokenSampler(pllm)
    tests   = [
        "assert even_bit_set_number(10) == 10",
        "assert even_bit_set_number(20) == 30",
        "assert even_bit_set_number(30) == 30",
    ]
    critic  = SyntaxOK(tok) * UnitTestPotential("tmp_mod", tests, tok)

    # 3-4 SMC
    seqs = await SMC(sampler, critic=critic)(
        n_particles   =  6,
        ess_threshold = .5,
        max_tokens    = 80,
        verbosity     = 1,
    )

    # 3-5 输出
    if seqs.decoded_posterior:
        print(f"\n find {len(seqs.decoded_posterior)} candidate who has passed all the tests：")
        for code, w in seqs.decoded_posterior.items():
            print(f"\n[weight={w:.3f}]\n{code}")
    else:
        print("\n No candidate passed the assertion")

if __name__ == "__main__":
    asyncio.run(main())
