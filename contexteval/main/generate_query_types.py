r"""Generate types for a set of queries.

Example usage:

INPUT_PATH=data/sampled_qa_contexts_gpt-4.jsonl
OUTPUT_PATH=data/all_queries_qtypes.jsonl
python3.10 main/generate_query_types.py \
--input_path=${INPUT_PATH} \
--output_path=${OUTPUT_PATH}
"""

from absl import app
from absl import flags
import models
import random
import sys
import tqdm
sys.path.append('contexteval/common')
import example_utils
import jsonl_utils
import tsv_utils


random.seed(423)

_INPUT_PATH = flags.DEFINE_string(
    "input_path", "", "Path to the input file."
)
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path", "", "Path to the output file."
)


def main(unused_argv) -> None:
  context_prompt = "\n".join(tsv_utils.read_txt("prompts/query_type_prompt.txt"))
  examples = example_utils.read_examples(_INPUT_PATH.value)
  model = models.GPT4()
  outputs = []
  for ex in tqdm.tqdm(examples):
    cur_prompt = (context_prompt + ".")[:-1]
    cur_prompt = cur_prompt.replace("[QUERY]", ex.query)
    qtype_output = model.generate(cur_prompt)
    outputs.append({"query": ex.query, "query_type": qtype_output})

  jsonl_utils.write(_OUTPUT_PATH.value, outputs)


if __name__ == "__main__":
  app.run(main)
