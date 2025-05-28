r"""Evaluates the number of constraints satisfied for each candidate response.

Example usage:

INPUT_PATH=data/sampled_evals_w_contexts_Qwen_Qwen2-72B-Instruct-judge_claude-3.5-sonnet_vs_meta-llama_Meta-Llama-3.1-405B-Instruct-Turbo.jsonl
OUTPUT_PATH=data/sampled_evals_w_contexts_Qwen_Qwen2-72B-Instruct-judge_claude-3.5-sonnet_vs_meta-llama_Meta-Llama-3.1-405B-Instruct-Turbo_constraints.jsonl
python3.10 main/eval_constraints.py \
--input_path=${INPUT_PATH} \
--output_path=${OUTPUT_PATH}
"""

import sys

import models
from absl import app, flags

sys.path.append("contexteval/common")
import jsonl_utils
import tqdm
import tsv_utils

_INPUT_PATH = flags.DEFINE_string("input_path", "", "Path to the input file.")
_OUTPUT_PATH = flags.DEFINE_string("output_path", "", "Path to the output file.")


def main(unused_argv) -> None:
    prompt = "\n".join(tsv_utils.read_txt("prompts/eval_constraints_prompt.txt"))
    examples = jsonl_utils.read(_INPUT_PATH.value)
    model = models.GPT4()
    for ex in tqdm.tqdm(examples):
        cur_prompt = (prompt + ".")[:-1]
        cur_prompt = cur_prompt.replace("[QUERY]", ex["query"])
        cur_prompt = cur_prompt.replace("[CONTEXT]", ex["context"])
        cur_prompt = cur_prompt.replace("[RESPONSE]", ex["candidate_one_response"])
        num_constraints_output_one = model.generate(cur_prompt)

        cur_prompt = (prompt + ".")[:-1]
        cur_prompt = cur_prompt.replace("[QUERY]", ex["query"])
        cur_prompt = cur_prompt.replace("[CONTEXT]", ex["context"])
        cur_prompt = cur_prompt.replace("[RESPONSE]", ex["candidate_two_response"])
        num_constraints_output_two = model.generate(cur_prompt)
        ex["candidate_one_constraints"] = num_constraints_output_one
        ex["candidate_two_constraints"] = num_constraints_output_two

    jsonl_utils.write(_OUTPUT_PATH.value, examples)


if __name__ == "__main__":
    app.run(main)
