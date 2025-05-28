r"""Generate codes for human justifications.

Example usage:
MODEL_PAIR=google_gemma-2-27b-it_vs_jamba-1.5-large
MODEL_PAIR=claude-3.5-sonnet_vs_meta-llama_Meta-Llama-3.1-405B-Instruct-Turbo
MODEL_PAIR=gpt-4_vs_gemini-1.5-flash-exp-0827

OUTPUT_PATH=data/human_judgements_tagged_${MODEL_PAIR}.json
python3 main/code_judgements.py \
--model_pair=${MODEL_PAIR} \
--output_path=${OUTPUT_PATH}
"""

import json
import random
import sys

import models
from absl import app, flags

sys.path.append('contexteval/common')
import jsonl_utils
import tqdm
import tsv_utils

random.seed(423)

_MODEL_PAIR = flags.DEFINE_string(
    "model_pair", "", "Name of the model pair."
)
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path", "", "Path to the output file."
)


def main(unused_argv) -> None:
  text_prompt = "\n".join(tsv_utils.read_txt("prompts/codify_judgements.txt"))

  with open(f"data/human_judgements/{_MODEL_PAIR.value}.jsonl") as f:
    human_judgements = json.load(f)

  model = models.GPT4()
  outputs = []
  for ex in tqdm.tqdm(human_judgements):
    cur_prompt = (text_prompt + ".")[:-1]
    if ex["follow_up_qas"] == []:
      follow_up_qas = "None"
    else:
      follow_up_qas = "\n".join([qa["qa"] for qa in ex["follow_up_qas"]])

    cur_prompt = cur_prompt.replace("[QUERY]", ex["query"].split("Follow-Up Questions and Answers")[0].strip() + "\nFollow-Up Questions: " + follow_up_qas)
    cur_prompt = cur_prompt.replace("[PREFERENCE]", ex["overall_preference"])
    cur_prompt = cur_prompt.replace("[JUSTIFICATION]", ex["justification"])
    codes = model.generate(cur_prompt)
    ex["codes"] = codes
    outputs.append(ex)

  jsonl_utils.write(_OUTPUT_PATH.value, outputs)


if __name__ == "__main__":
  app.run(main)
