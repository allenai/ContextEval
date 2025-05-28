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
from typing import Any, Dict, List, Optional, Tuple

import tqdm
from absl import app, flags

sys.path.append("contexteval/common")

import jsonl_utils
import models
import tsv_utils

# Set random seed at module level
random.seed(23)

_CANDIDATE_ONE_PATH = flags.DEFINE_string(
    "candidate_one_path", "", "Path to candidate one's output."
)
_CANDIDATE_TWO_PATH = flags.DEFINE_string(
    "candidate_two_path", "", "Path to candidate two's output."
)
_OUTPUT_PATH = flags.DEFINE_string("output_path", "", "Path to the output file.")
_EVAL_MODEL = flags.DEFINE_string("eval_model", "", "Model to use for evaluation.")
_W_CONTEXT = flags.DEFINE_boolean("w_context", True, "Whether to provide queries with context.")


def extract_dictionary(input_string: str) -> Optional[Dict[str, str]]:
    """Extract dictionary from model output string.

    Args:
        input_string: String containing model output

    Returns:
        Dictionary containing judgement or None if invalid
    """
    try:
        input_string = input_string.replace("*", "").strip()
        search_str = "output: "
        end_str = "}"
        dict_start = input_string.find(search_str) + len(search_str)
        dict_end = input_string.find(end_str) + 1
        dict_str = input_string[dict_start:dict_end]
        dictionary = ast.literal_eval(dict_str)
        if dictionary["judgement"] not in ["Response 1", "Response 2", "Tie"]:
            print(f"Invalid judgement: {input_string}")
            return None
        return dictionary
    except Exception:
        print(f"Error parsing dictionary from: {input_string}")
        return None


def get_model(model_name: str) -> Any:
    """Initialize and return appropriate model.

    Args:
        model_name: Name of model to initialize

    Returns:
        Initialized model instance
    """
    if "gpt" in model_name:
        return models.GPT4(model_name=model_name)
    elif "gemini" in model_name:
        return models.Gemini(model_name=model_name)
    elif "claude" in model_name:
        return models.Claude(model_name=model_name)
    elif model_name == "llama-3.1":
        return models.TogetherAI()
    elif "jamba" in model_name:
        return models.Jamba(model_name=model_name)
    else:
        return models.TogetherAI(model_name=model_name)


def get_eval_judgement(
    prompt: str, model: Any, max_retries: int = 3
) -> Tuple[str, Optional[Dict[str, str]]]:
    """Get evaluation judgement from model with retries.

    Args:
        prompt: Prompt to send to model
        model: Model instance to use
        max_retries: Maximum number of retry attempts

    Returns:
        Tuple of (raw judgement string, parsed dictionary or None)
    """
    for attempt in range(max_retries):
        try:
            eval_judgement = model.generate(prompt)
            dict_out = extract_dictionary(eval_judgement)
            if dict_out is not None:
                return eval_judgement, dict_out
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to get evaluation judgement after {max_retries} attempts: {e}")
            continue
    return "", None


def main(unused_argv) -> None:
    # Load appropriate prompt
    if _W_CONTEXT.value:
        eval_prompt = "\n".join(tsv_utils.read_txt("prompts/eval_w_context_prompt.txt"))
        candidate_one_outputs_w_context = jsonl_utils.read(
            _CANDIDATE_ONE_PATH.value.replace("wo_contexts", "w_contexts")
        )
    else:
        eval_prompt = "\n".join(tsv_utils.read_txt("prompts/eval_wo_context_prompt.txt"))

    # Load candidate outputs
    candidate_one_outputs = jsonl_utils.read(_CANDIDATE_ONE_PATH.value)
    candidate_two_outputs = jsonl_utils.read(_CANDIDATE_TWO_PATH.value)

    # Initialize model
    model = get_model(_EVAL_MODEL.value)

    outputs = []
    for idx in tqdm.tqdm(range(len(candidate_one_outputs))):
        candidate_one_output = candidate_one_outputs[idx]["response"]
        candidate_two_output = candidate_two_outputs[idx]["response"]

        # Prepare prompt
        cur_prompt = (eval_prompt + ".")[:-1]
        cur_prompt = cur_prompt.replace("[QUERY]", candidate_one_outputs[idx]["query"])
        rand_choice = random.randint(1, 2)
        if rand_choice == 1:
            cur_prompt = cur_prompt.replace("[RESPONSE 1]", candidate_one_output)
            cur_prompt = cur_prompt.replace("[RESPONSE 2]", candidate_two_output)
        else:
            cur_prompt = cur_prompt.replace("[RESPONSE 1]", candidate_two_output)
            cur_prompt = cur_prompt.replace("[RESPONSE 2]", candidate_one_output)

        if _W_CONTEXT.value:
            cur_prompt = cur_prompt.replace(
                "[CONTEXT]", str(candidate_one_outputs_w_context[idx]["context"])
            )

        # Get evaluation judgement
        eval_judgement, _ = get_eval_judgement(cur_prompt, model)

        # Prepare output
        output = {
            "query": candidate_one_outputs[idx]["query"],
            "candidate_one_response": candidate_one_output,
            "candidate_two_response": candidate_two_output,
            "rand_choice": rand_choice,
            "eval_judgement": eval_judgement,
        }

        if _W_CONTEXT.value:
            output["context"] = candidate_one_outputs_w_context[idx]["context"]

        outputs.append(output)

    jsonl_utils.write(_OUTPUT_PATH.value, outputs)


if __name__ == "__main__":
    app.run(main)
