r"""Generate codes for rating justifications from model evaluations.

Example usage:
MODEL_PAIR=claude-3.5-sonnet_vs_meta-llama_Meta-Llama-3.1-405B-Instruct-Turbo
MODEL_PAIR=gpt-4_vs_gemini-1.5-flash-exp-0827
MODEL_PAIR=google_gemma-2-27b-it_vs_jamba-1.5-large

OUTPUT_PATH=data/model_evals_tagged_${MODEL_PAIR}.json
python3 main/code_model_judgements.py \
--model_pair=${MODEL_PAIR} \
--output_path=${OUTPUT_PATH}
"""

import random
import sys

import models
from absl import app, flags

sys.path.append("contexteval/common")
import jsonl_utils
import tqdm
import tsv_utils

random.seed(423)

_MODEL_PAIR = flags.DEFINE_string("model_pair", "", "Name of the model pair.")
_OUTPUT_PATH = flags.DEFINE_string("output_path", "", "Path to the output file.")


def get_eval_justifications(eval_models, filename_template):
    eval_justifications = []
    eval_data = {}
    for model in eval_models:
        filepath = filename_template.replace("EVAL_MODEL", model)
        eval_data[model] = jsonl_utils.read(filepath)
    for i, ex in enumerate(eval_data["Qwen_Qwen2-72B-Instruct"]):
        qualified_models = [
            model
            for model in eval_models
            if "Justification" in eval_data[model][i]["eval_judgement"]
        ]
        if len(qualified_models) == 0:
            continue
        rand_model = random.choice(qualified_models)
        if "context" in ex:
            context = ex["context"]
        else:
            context = ""
        eval_justifications.append(
            {
                "judgement": eval_data[rand_model][i]["eval_judgement"],
                "query": ex["query"],
                "context": context,
                "setting": filename_template.split("/")[-1]
                .split("_EVAL_MODEL")[0]
                .replace("sampled_evals_", ""),
                "eval_model": rand_model,
            }
        )
    return eval_justifications


def main(unused_argv) -> None:
    text_prompt = "\n".join(tsv_utils.read_txt("prompts/codify_judgements.txt"))

    if _MODEL_PAIR.value == "claude-3.5-sonnet_vs_meta-llama_Meta-Llama-3.1-405B-Instruct-Turbo":
        eval_models = [
            "gpt-4",
            "gemini-1.5-pro",
            "google_gemma-2-27b-it",
            "jamba-1.5-large",
            "Qwen_Qwen2-72B-Instruct",
        ]
    elif _MODEL_PAIR.value == "google_gemma-2-27b-it_vs_jamba-1.5-large":
        eval_models = [
            "claude-3.5-sonnet",
            "meta-llama_Meta-Llama-3.1-405B-Instruct-Turbo",
            "gpt-4",
            "gemini-1.5-pro",
            "Qwen_Qwen2-72B-Instruct",
        ]
    elif _MODEL_PAIR.value == "gpt-4_vs_gemini-1.5-flash-exp-0827":
        eval_models = [
            "claude-3.5-sonnet",
            "meta-llama_Meta-Llama-3.1-405B-Instruct-Turbo",
            "google_gemma-2-27b-it",
            "jamba-1.5-large",
            "Qwen_Qwen2-72B-Instruct",
        ]

    filename_template = f"data/sampled_evals_wo_contexts_EVAL_MODEL-judge_{_MODEL_PAIR.value}.jsonl"
    eval_justifications = get_eval_justifications(eval_models, filename_template)
    filename_template = f"data/sampled_evals_gen_wo_contexts_eval_w_contexts_EVAL_MODEL-judge_{_MODEL_PAIR.value}.jsonl"
    eval_justifications += get_eval_justifications(eval_models, filename_template)
    filename_template = f"data/sampled_evals_w_contexts_EVAL_MODEL-judge_{_MODEL_PAIR.value}.jsonl"
    eval_justifications += get_eval_justifications(eval_models, filename_template)

    eval_justifications = random.sample(eval_justifications, 1000)

    model = models.GPT4()
    outputs = []
    for ex in tqdm.tqdm(eval_justifications):
        cur_prompt = (text_prompt + ".")[:-1]
        cur_prompt = cur_prompt.replace("[QUERY]", ex["query"] + ex["context"])
        cur_prompt = cur_prompt.replace("[PREFERENCE]", ex["judgement"])
        cur_prompt = cur_prompt.replace("Justification: [JUSTIFICATION]\n", "")
        codes = model.generate(cur_prompt)
        ex["codes"] = codes
        outputs.append(ex)

    jsonl_utils.write(_OUTPUT_PATH.value, outputs)


if __name__ == "__main__":
    app.run(main)
