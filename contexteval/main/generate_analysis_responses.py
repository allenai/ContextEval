r"""Generate responses for analysis queries.

Example usage:

CONTEXT_NAME="Political Context"
python3.10 main/generate_analysis_responses.py \
--context_name="${CONTEXT_NAME}"
"""

from absl import app
from absl import flags

import ast
import os
import random
import re
import sys
sys.path.append('contexteval/common')
import jsonl_utils
import tsv_utils
import models
import tqdm

_CONTEXT_NAME = flags.DEFINE_string(
    "context_name", None, "Name of the contextual attribute."
)
_MODEL_NAME = flags.DEFINE_string(
    "model_name", "gpt-4", "Model name."
)
_DEFAULT = flags.DEFINE_bool(
    "default", True, "Generate default response.",
)

random.seed(4)
MAX_QUERIES = 1000


def extract_dictionary(input_string):
  try:
    search_str = "{"
    end_str = "}"
    dict_start = input_string.find(search_str)
    dict_end = input_string.find(end_str) + 1
    dict_str = input_string[dict_start:dict_end]
    dictionary = ast.literal_eval(dict_str)
    return dictionary
  except Exception:
    return None


def get_qa(input_str):
  question_match = re.search(r'Q: (.*?) A:', input_str)
  question = question_match.group(1) if question_match else "Question not found"
  list_match = re.search(r'A: (\[.*\])', input_str)
  answer_list_str = list_match.group(1)      
  answer_list = ast.literal_eval(answer_list_str)
  return question, answer_list


def main(unused_argv) -> None:
  print(f"Generating responses for {_CONTEXT_NAME.value}.")
  w_ctx_response_prompt = "\n".join(tsv_utils.read_txt("prompts/response_w_context_prompt.txt"))
  wo_ctx_response_prompt = "\n".join(tsv_utils.read_txt("prompts/response_wo_context_prompt.txt"))
  
  examples_by_context = {}
  qa_by_context = {}
  for file in os.listdir("data/context_analysis/"):
    if file.endswith("_filter_outputs.jsonl"):
      data = jsonl_utils.read(f"data/context_analysis/{file}")
      filtered_data = []
      for ex in data:
        preds = extract_dictionary(ex["output"])
        if preds is None:
          continue
        if preds["1"] == "Yes" and preds["2"] == "Yes" and preds["3"] == "Yes":
          filtered_data.append(ex)
      examples_by_context[filtered_data[0]["context"]] = filtered_data
      qa_by_context[filtered_data[0]["context"]] = get_qa(filtered_data[0]["question"])
      print(f"Loaded {len(filtered_data)} examples for {filtered_data[0]['context']}.")

  OUTPUT_PATH = f"data/context_analysis/{_CONTEXT_NAME.value.replace('/', '_').replace(' ', '_')}_responses_{_MODEL_NAME.value}.jsonl"
  if not _DEFAULT.value:
    OUTPUT_PATH = OUTPUT_PATH.replace(".jsonl", "_adapted.jsonl")

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


  sampled_examples = random.sample(examples_by_context[_CONTEXT_NAME.value], min(MAX_QUERIES, len(examples_by_context[_CONTEXT_NAME.value])))
  outputs = []
  for ex in tqdm.tqdm(sampled_examples):
    if _DEFAULT.value:
      cur_prompt = (wo_ctx_response_prompt + ".")[:-1]
      cur_prompt = cur_prompt.replace("[QUERY]", ex["query"])
      response = model.generate(input_text=cur_prompt, max_len=2048)
      outputs.append({"query": ex["query"], "response": response})
      jsonl_utils.write(OUTPUT_PATH, [outputs[-1]], append=True, verbose=False)
    else:
      question, answer_list = qa_by_context[ex["context"]]
      for answer in answer_list:
        cur_prompt = (w_ctx_response_prompt + ".")[:-1]
        cur_prompt = cur_prompt.replace("[QUERY]", ex["query"])
        context = question + " " + answer
        cur_prompt = cur_prompt.replace("[CONTEXT]", context)
        context_response = model.generate(input_text=cur_prompt, max_len=2048)
        outputs.append({"query": ex["query"], "context": context, "response": context_response})
        jsonl_utils.write(OUTPUT_PATH, [outputs[-1]], append=True, verbose=False)


if __name__ == "__main__":
  app.run(main)
