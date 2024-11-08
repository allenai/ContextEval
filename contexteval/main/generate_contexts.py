r"""Generate contexts for a set of queries.

Example usage:

INPUT_PATH=data/all_data_latest_filtered.jsonl
OUTPUT_PATH=data/sampled_qa_contexts_gpt-4.jsonl
python3.10 main/generate_contexts.py \
--input_path=${INPUT_PATH} \
--output_path=${OUTPUT_PATH} \
--model_name=gpt-4
"""

from absl import app
from absl import flags

import models
import random
import sys
sys.path.append('contexteval/common')
import example_utils
import tsv_utils
import tqdm


random.seed(42)

_MODEL_NAME = flags.DEFINE_string(
    "model_name", "", "Model name."
)
_INPUT_PATH = flags.DEFINE_string(
    "input_path", "", "Path to the input file."
)
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path", "", "Path to the output file."
)


def main(unused_argv) -> None:
  context_prompt = "\n".join(tsv_utils.read_txt("prompts/context_qa_prompt.txt"))
  examples = example_utils.read_examples(_INPUT_PATH.value)

  if _MODEL_NAME.value == "gpt-4":
    model = models.GPT4()
  elif _MODEL_NAME.value == "gemini-1.5-pro":
    model = models.Gemini()
  elif _MODEL_NAME.value == "claude-3.5-sonnet":
    model = models.Claude(model_name="claude-3.5-sonnet")
  else:
    raise ValueError(f"Unsupported model name: {_MODEL_NAME.value}")
  outputs = []
  for ex in tqdm.tqdm(examples):
    cur_prompt = (context_prompt + ".")[:-1]
    cur_prompt = cur_prompt.replace("[QUERY]", ex.query)
    context_output = model.generate(cur_prompt)
    ex.contexts = context_output
    outputs.append(ex)
    example_utils.write_examples(_OUTPUT_PATH.value, [ex], append=True)


if __name__ == "__main__":
  app.run(main)
