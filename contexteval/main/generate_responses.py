r"""Generate responses for a set of queries.

Example usage:

INPUT_PATH=data/sampled_qa_contexts_individual.jsonl
W_CONTEXT=True
OUTPUT_PATH=data/sampled_responses_w_contexts_gpt4.jsonl
python3.10 main/generate_responses.py \
--input_path=${INPUT_PATH} \
--output_path=${OUTPUT_PATH} \
--w_context=${W_CONTEXT}
"""

from absl import app
from absl import flags

import random
import sys
sys.path.append('contexteval/common')
import example_utils
import jsonl_utils
import tsv_utils
import models
import tqdm

random.seed(423)

_INPUT_PATH = flags.DEFINE_string(
    "input_path", "", "Path to the input file."
)
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path", "", "Path to the output file."
)
_MODEL_NAME = flags.DEFINE_string(
    "model_name", "", "Model name."
)
_W_CONTEXT = flags.DEFINE_boolean(
    "w_context", True, "Whether to provide queries with context."
)


def main(unused_argv) -> None:
  if _W_CONTEXT.value:
    response_prompt = "\n".join(tsv_utils.read_txt("prompts/response_w_context_prompt.txt"))
  else:
    response_prompt = "\n".join(tsv_utils.read_txt("prompts/response_wo_context_prompt.txt"))

  examples = example_utils.read_examples(_INPUT_PATH.value)

  if "gpt" in _MODEL_NAME.value:
    model = models.GPT4(model_name=_MODEL_NAME.value)
  elif "gemini" in _MODEL_NAME.value:
    model = models.Gemini(model_name=_MODEL_NAME.value)
  elif "claude" in _MODEL_NAME.value:
    model = models.Claude(model_name=_MODEL_NAME.value)
  elif "jamba" in _MODEL_NAME.value:
    model = models.Jamba(model_name=_MODEL_NAME.value)
  else:
    model = models.TogetherAI(model_name=_MODEL_NAME.value)

  outputs = []
  idx = 0
  for ex in tqdm.tqdm(examples):
    cur_prompt = (response_prompt + ".")[:-1]
    cur_prompt = cur_prompt.replace("[QUERY]", ex.query)
    if _W_CONTEXT.value:
      cur_prompt = cur_prompt.replace("[CONTEXT]", ex.sampled_context)
      context_response = model.generate(input_text=cur_prompt, max_len=2048)
      outputs.append({"query": ex.query, "context": ex.sampled_context, "response": context_response, "all_contexts": ex.contexts})
    else:
      response = model.generate(input_text=cur_prompt, max_len=2048)
      outputs.append({"query": ex.query, "response": response})
    idx += 1

    jsonl_utils.write(_OUTPUT_PATH.value, [outputs[-1]], append=True)


if __name__ == "__main__":
  app.run(main)
