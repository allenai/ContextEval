r"""Generate evaluation judgements for a pair of responses to queries.

Example usage:

CANDIDATE_ONE_PATH=data/sampled_responses_w_contexts_gpt4.jsonl
CANDIDATE_TWO_PATH=data/sampled_responses_w_contexts_gemini-1.5-pro.jsonl
OUTPUT_PATH=data/sampled_evals_w_contexts_gpt4_vs_gemini-1.5-pro.jsonl
EVAL_MODEL=claude-3.5-sonnet
W_CONTEXT=True
python3.10 main/generate_evals.py \
--candidate_one_path=${CANDIDATE_ONE_PATH} \
--candidate_two_path=${CANDIDATE_TWO_PATH} \
--output_path=${OUTPUT_PATH} \
--eval_model=${EVAL_MODEL} \
--w_context=${W_CONTEXT}
"""

import ast
import random
import sys

import models
from absl import app, flags

sys.path.append('contexteval/common')
import ast
import random
import sys

import jsonl_utils
import tqdm
import tsv_utils
from absl import app, flags

sys.path.append('contexteval/common')

import jsonl_utils
import models
import tsv_utils

random.seed(23)

_CANDIDATE_ONE_PATH = flags.DEFINE_string(
    "candidate_one_path", "", "Path to candidate one's output."
)
_CANDIDATE_TWO_PATH = flags.DEFINE_string(
    "candidate_two_path", "", "Path to candidate two's output."
)
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path", "", "Path to the output file."
)
_EVAL_MODEL = flags.DEFINE_string(
    "eval_model", "", "Model to use for evaluation."
)
_W_CONTEXT = flags.DEFINE_boolean(
    "w_context", True, "Whether to provide queries with context."
)



def extract_dictionary(input_string):
  try:
    input_string = input_string.replace("*", "").strip()
    search_str = "output: "
    end_str = "}"
    dict_start = input_string.find(search_str) + len(search_str)
    dict_end = input_string.find(end_str) + 1
    dict_str = input_string[dict_start:dict_end]
    dictionary = ast.literal_eval(dict_str)
    if dictionary["judgement"] not in ["Response 1", "Response 2", "Tie"]:
      print(input_string)
      return None
    return dictionary
  except Exception as e:
    print(input_string)
    print(f"Error occurred: {e}")
    return None


def main(unused_argv) -> None:
  if _W_CONTEXT.value:
    eval_prompt = "\n".join(tsv_utils.read_txt("prompts/eval_w_context_prompt.txt"))
    candidate_one_outputs_w_context = jsonl_utils.read(_CANDIDATE_ONE_PATH.value.replace("wo_contexts", "w_contexts"))
  else:
    eval_prompt = "\n".join(tsv_utils.read_txt("prompts/eval_wo_context_prompt.txt"))
  candidate_one_outputs = jsonl_utils.read(_CANDIDATE_ONE_PATH.value)
  candidate_two_outputs = jsonl_utils.read(_CANDIDATE_TWO_PATH.value)

  if "gpt" in _EVAL_MODEL.value:
    model = models.GPT4(model_name=_EVAL_MODEL.value)
  elif "gemini" in _EVAL_MODEL.value:
    model = models.Gemini(model_name=_EVAL_MODEL.value)
  elif "claude" in _EVAL_MODEL.value:
    model = models.Claude(model_name=_EVAL_MODEL.value)
  elif _EVAL_MODEL.value == "llama-3.1":
    model = models.TogetherAI()
  elif "jamba" in _EVAL_MODEL.value:
    model = models.Jamba(model_name=_EVAL_MODEL.value)
  else:
    model = models.TogetherAI(model_name=_EVAL_MODEL.value)

  outputs = []
  for idx in tqdm.tqdm(range(len(candidate_one_outputs))):
    candidate_one_output = candidate_one_outputs[idx]["response"]
    candidate_two_output = candidate_two_outputs[idx]["response"]
    
    cur_prompt = (eval_prompt + ".")[:-1]
    cur_prompt = cur_prompt.replace("[QUERY]", candidate_one_outputs[idx]["query"])
    rand_choice = random.randint(1, 2)
    if rand_choice == 1:
      cur_prompt = cur_prompt.replace(
          "[RESPONSE 1]", candidate_one_output
      )
      cur_prompt = cur_prompt.replace(
          "[RESPONSE 2]", candidate_two_output
      )
    elif rand_choice == 2:
      cur_prompt = cur_prompt.replace(
          "[RESPONSE 1]", candidate_two_output
      )
      cur_prompt = cur_prompt.replace(
          "[RESPONSE 2]", candidate_one_output
      )

    if _W_CONTEXT.value:
      cur_prompt = cur_prompt.replace("[CONTEXT]", str(candidate_one_outputs_w_context[idx]["context"]))

    dict_out = None
    count = 0
    while dict_out is None:
      eval_judgement = model.generate(cur_prompt)
      dict_out = extract_dictionary(eval_judgement)
      count += 1
      if count == 3:
        break

    outputs.append({"query": candidate_one_outputs[idx]["query"],
                    "candidate_one_response": candidate_one_output, "candidate_two_response": candidate_two_output,
                    "rand_choice": rand_choice, "eval_judgement": eval_judgement})
  
    if _W_CONTEXT.value:
      outputs[-1]["context"] = candidate_one_outputs_w_context[idx]["context"]

  jsonl_utils.write(_OUTPUT_PATH.value, outputs)


if __name__ == "__main__":
  app.run(main)
