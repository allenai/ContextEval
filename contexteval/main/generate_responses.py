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

import random
import sys
from typing import Any, Dict, List, Optional

from absl import app, flags

sys.path.append("contexteval/common")
import example_utils
import jsonl_utils
import models
import tqdm
import tsv_utils

# Set random seed at module level
random.seed(423)

_INPUT_PATH = flags.DEFINE_string("input_path", "", "Path to the input file.")
_OUTPUT_PATH = flags.DEFINE_string("output_path", "", "Path to the output file.")
_MODEL_NAME = flags.DEFINE_string("model_name", "", "Model name.")
_W_CONTEXT = flags.DEFINE_boolean("w_context", True, "Whether to provide queries with context.")


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
    elif "jamba" in model_name:
        return models.Jamba(model_name=model_name)
    else:
        return models.TogetherAI(model_name=model_name)


def prepare_prompt(prompt_template: str, query: str, context: Optional[str] = None) -> str:
    """Prepare prompt by replacing placeholders.

    Args:
        prompt_template: Template string with placeholders
        query: Query to insert
        context: Optional context to insert

    Returns:
        Prepared prompt string
    """
    prompt = (prompt_template + ".")[:-1]
    prompt = prompt.replace("[QUERY]", query)
    if context is not None:
        prompt = prompt.replace("[CONTEXT]", context)
    return prompt


def main(unused_argv) -> None:
    # Load appropriate prompt
    if _W_CONTEXT.value:
        response_prompt = "\n".join(tsv_utils.read_txt("prompts/response_w_context_prompt.txt"))
    else:
        response_prompt = "\n".join(tsv_utils.read_txt("prompts/response_wo_context_prompt.txt"))

    # Load examples
    examples = example_utils.read_examples(_INPUT_PATH.value)

    # Initialize model
    model = get_model(_MODEL_NAME.value)

    outputs = []
    for ex in tqdm.tqdm(examples):
        try:
            # Prepare prompt
            cur_prompt = prepare_prompt(
                response_prompt, ex.query, ex.sampled_context if _W_CONTEXT.value else None
            )

            # Generate response
            response = model.generate(input_text=cur_prompt, max_len=2048)

            # Prepare output
            output = {"query": ex.query, "response": response}
            if _W_CONTEXT.value:
                output.update(
                    {
                        "context": ex.sampled_context,
                        "all_contexts": ex.contexts,
                    }
                )

            outputs.append(output)
            jsonl_utils.write(_OUTPUT_PATH.value, [output], append=True)

        except Exception as e:
            print(f"Error processing example: {e}")
            continue


if __name__ == "__main__":
    app.run(main)
