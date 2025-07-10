
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase
def load_gpt2(model_name: str = "gpt2"):
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained(model_name)
    mdl = GPT2LMHeadModel.from_pretrained(model_name)
    mdl.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return mdl, tok

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_llama(
    repo = "meta-llama/Llama-3.2-1B-Instruct",
):
    """
    First, load it on the CPU with FP16
    then MPS
    """
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        repo,
        use_fast=True,
        trust_remote_code=True,
    )

    # CPU load FP16
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
        trust_remote_code=True,
    )

    # change MPS and eval
    model = model.to("mps").eval()

    return model, tokenizer
# Determine when to trigger simulation
# def is_simulation_point(text: str) -> bool:
#     """
#     Simulate only at natural breakpoints:
#     End with a newline character "\n"
#    With a colon ":" or a semicolon "; The end
#    If True is returned, simulation completion will be performed.
#     """
#     return text.endswith("\n") or text.rstrip().endswith((":",";"))
def is_simulation_point(text: str) -> bool:
    """
    The simulation completion is triggered once every n tokens received.
    """
    # Add a counter
    if not hasattr(is_simulation_point, "counter"):
        is_simulation_point.counter = 0

    # Accumulate one step
    is_simulation_point.counter += 1

    # It is triggered when it reaches a multiple of a certain number
    return is_simulation_point.counter % 50 == 0


def simulated_completion_generate(prefix, model, tokenizer, max_new_tokens=600):
    inputs = tokenizer(prefix, return_tensors='pt')
    input_ids = inputs['input_ids']          # Stay in the CPU

    orig_device = next(model.parameters()).device   # It might be "mps"
    model_cpu   = model.to("cpu")                   # Migrate to the CPU

    with torch.no_grad():
        output_ids = model_cpu.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=600,
            top_p=0.95,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    model.to(orig_device)    # Put it back to the original device and continue to use the SMC later

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# def simulated_completion_manual(
#     prefix: str,
#     model: PreTrainedModel,
#     tokenizer: PreTrainedTokenizerBase,
#     max_new_tokens: int = 600,
#     temperature: float = 1.0,
#     top_k: int = 0,
#     top_p: float = 1.0
# ) -> str:
#     """
#     Manually call the model to calculate logits, and then use F.softmax + torch.multinomial for sampling.
# prefix: The text prefix has been generated
# temperature: Temperature coefficient. Results are more concentrated when <1.0, and more random when >1.0
# top_k: Sample only among the top_k tokens with the highest probability. 0 indicates that it is not enabled
# top_p: nucleus sampling threshold. When the cumulative probability exceeds top_p, the remaining logits are masked
#     """
#     inputs = tokenizer(prefix, return_tensors='pt')
#     generated = inputs['input_ids'][0].tolist()
#
#     for _ in range(max_new_tokens):
#         #  logits
#         outputs = model(torch.tensor([generated]))
#         logits = outputs.logits[0, -1, :]
#
#         # change logits：
#         scaled_logits = logits / temperature
#         #    top_k
#         if top_k > 0:
#             values, indices = torch.topk(scaled_logits, top_k)
#             min_values = values[-1]
#             scaled_logits[scaled_logits < min_values] = float('-inf')
#
#         if top_p < 1.0:
#             # rank
#             sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
#             cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
#
#             # Set the logits whose cumulative probability exceeds top p to -inf
#             sorted_logits[cumulative_probs > top_p] = float("-inf")
#
#             # Write the result back to the original logits
#             scaled_logits[sorted_indices] = sorted_logits
#
#         # Softmax - population
#         probs = F.softmax(scaled_logits, dim=-1)
#         # next token
#         next_id = torch.multinomial(probs, num_samples=1).item()
#         generated.append(next_id)
#         if next_id == tokenizer.eos_token_id:
#             break
#
#     return tokenizer.decode(generated, skip_special_tokens=True)