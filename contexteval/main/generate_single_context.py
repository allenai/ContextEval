r"""Generate contexts for a set of queries.

Example usage:

INPUT_PATH=data/sampled_qa_contexts_with_validation.jsonl
OUTPUT_PATH=data/sampled_qa_contexts_individual.jsonl
python3.10 main/generate_single_context.py \
--input_path=${INPUT_PATH} \
--output_path=${OUTPUT_PATH}
"""

from absl import app
from absl import flags

import random
import sys
sys.path.append('contexteval/common')
import example_utils
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


def main(unused_argv) -> None:
  context_prompt = "\n".join(tsv_utils.read_txt("prompts/single_context_prompt.txt"))
  examples = {}
  examples = example_utils.read_examples(_INPUT_PATH.value)
  model = models.GPT4()

  outputs = []
  for ex in tqdm.tqdm(examples):
    cur_prompt = (context_prompt + ".")[:-1]
    cur_prompt = cur_prompt.replace("[QUERY]", ex.query)
    cur_prompt = cur_prompt.replace("[CONTEXT]", ex.contexts)
    sampled_context = model.generate(input_text=cur_prompt, max_len=2048)
    ex.sampled_context = sampled_context
    outputs.append(ex)
    example_utils.write_examples(_OUTPUT_PATH.value, [outputs[-1]], append=True)


if __name__ == "__main__":
  app.run(main)
