r"""Filter queries based on well-formedness.

Example usage:

INPUT_PATH=data/all_data_latest.jsonl
OUTPUT_PATH=data/all_data_latest_filtered.jsonl
python3.10 main/filter_queries.py \
--input_path=${INPUT_PATH} \
--output_path=${OUTPUT_PATH}
"""

from absl import app
from absl import flags

import collections
import models
import random
import sys
sys.path.append('contexteval/common')
import example_utils
import tsv_utils


random.seed(423)

_INPUT_PATH = flags.DEFINE_string(
    "input_path", "", "Path to the input file."
)
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path", "", "Path to the output file."
)

def sample_examples(examples, num_ex_per_dataset):
  context_prompt = "\n".join(tsv_utils.read_txt("prompts/well_formed_query_check_prompt.txt"))
  model = models.GPT4()
  examples_by_dataset = collections.defaultdict(list)
  for ex in examples:
    examples_by_dataset[ex.source].append(ex)

  sampled_examples = []
  for dataset, exs in examples_by_dataset.items():
    if dataset not in num_ex_per_dataset:
      continue
    examples_to_sample = min(num_ex_per_dataset[dataset], len(exs))
    print(f"Sampling {examples_to_sample} examples from {dataset}")
    print("============================================")
    random.shuffle(exs)
    samples_from_dataset = 0
    for ex in exs:
      cur_prompt = (context_prompt + ".")[:-1]
      cur_prompt = cur_prompt.replace("[QUERY]", ex.query)
      output = model.generate(cur_prompt)
      if "Yes" in output:
        sampled_examples.append(ex)
        samples_from_dataset += 1
        print(f"Sampled {samples_from_dataset} examples from {dataset}")
        if samples_from_dataset == examples_to_sample:
          break

  return sampled_examples


def main(unused_argv) -> None:
  num_ex_per_dataset = {"lmsys/chatbot_arena_conversations": 2500,
                        "tatsu-lab/alpaca_eval": 800,
                        "lmsys/mt_bench_human_judgments": 100,
                        "cmalaviya/expertqa": 300,
                        "fangyuan/kiwi": 100}
  
  examples = example_utils.read_examples(_INPUT_PATH.value)  
  sampled_examples = sample_examples(examples, num_ex_per_dataset)
  example_utils.write_examples(_OUTPUT_PATH.value, sampled_examples)


if __name__ == "__main__":
  app.run(main)
