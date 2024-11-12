r"""Filter queries based on contextual attribute.

Example usage:
CONTEXT_NAME="Level of Detail"
python3.10 main/filter_queries_by_context.py \
    --query_path="data/all_data_latest.jsonl" \
    --context_name=${CONTEXT_NAME}
"""

from absl import app
from absl import flags

import models
import sys
sys.path.append('contexteval/common')
import example_utils
import jsonl_utils
import tsv_utils
import tqdm
import ast


_QUERY_PATH = flags.DEFINE_string(
    "query_path", "data/all_data_latest.jsonl", "Path to the query file."
)
_CONTEXT_NAME = flags.DEFINE_string(
    "context_name", "", "Name of the contextual attribute."
)

def extract_dictionary(input_string):
  try:
    search_str = "{"
    end_str = "}"
    dict_start = input_string.find(search_str)
    dict_end = input_string.find(end_str) + 1
    dict_str = input_string[dict_start:dict_end]
    dictionary = ast.literal_eval(dict_str)
    return dictionary
  except Exception as e:
    print(input_string)
    print(f"Error occurred: {e}")
    return None


def main(unused_argv) -> None:
  contextual_attributes = tsv_utils.read_txt("data/contextual_attributes_qas.txt")
  context_to_question = {}
  for line in contextual_attributes:
    if line[0] == "*":
      cur_context = line[1:].strip()
    else:
      context_to_question[cur_context] = line.strip()

  examples = example_utils.read_examples(_QUERY_PATH.value)
  filter_prompt = "\n".join(tsv_utils.read_txt("prompts/filter_queries_for_context_prompt.txt"))
  model = models.GPT4()
  for context, question in context_to_question.items():
    if context != _CONTEXT_NAME.value:
      continue
    outputs = []
    for ex in tqdm.tqdm(examples):
      cur_prompt = (filter_prompt + ".")[:-1]
      cur_prompt = cur_prompt.replace("[QUERY]", ex.query)
      cur_prompt = cur_prompt.replace("[QUESTION]", question)
      dict_out = None
      count = 0
      while dict_out is None:
        output = model.generate(cur_prompt)
        dict_out = extract_dictionary(output)
        count += 1
        if count == 3:
          break
      outputs.append({"query": ex.query, "question": question, "context": context, "output": output})

    jsonl_utils.write(f"data/context_analysis/{context.replace('/', '_').replace(' ', '_')}_filter_outputs.jsonl", outputs)


if __name__ == "__main__":
  app.run(main)
