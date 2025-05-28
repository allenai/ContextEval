r"""Generate contexts for a set of queries.

Example usage:

INPUT_PATH=data/all_data_latest_filtered.jsonl
OUTPUT_PATH=data/sampled_qa_contexts_gpt-4.jsonl
python3.10 main/generate_contexts.py \
--input_path=${INPUT_PATH} \
--output_path=${OUTPUT_PATH} \
--model_name=gpt-4
"""

import random
import sys
from typing import Any, List

import tqdm
from absl import app, flags

sys.path.append("contexteval/common")
import example_utils
import models
import tsv_utils

# Set random seed at module level
random.seed(42)

_MODEL_NAME = flags.DEFINE_string("model_name", "", "Model name.")
_INPUT_PATH = flags.DEFINE_string("input_path", "", "Path to the input file.")
_OUTPUT_PATH = flags.DEFINE_string("output_path", "", "Path to the output file.")


def get_model(model_name: str) -> Any:
    """Initialize and return appropriate model.

    Args:
        model_name: Name of model to initialize

    Returns:
        Initialized model instance

    Raises:
        ValueError: If model name is not supported
    """
    if model_name == "gpt-4":
        return models.GPT4()
    elif model_name == "gemini-1.5-pro":
        return models.Gemini()
    elif model_name == "claude-3.5-sonnet":
        return models.Claude(model_name="claude-3.5-sonnet")
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def prepare_prompt(prompt_template: str, query: str) -> str:
    """Prepare prompt by replacing query placeholder.

    Args:
        prompt_template: Template string with [QUERY] placeholder
        query: Query to insert

    Returns:
        Prepared prompt string
    """
    prompt = (prompt_template + ".")[:-1]
    return prompt.replace("[QUERY]", query)


def main(unused_argv) -> None:
    # Load prompt template
    context_prompt = "\n".join(tsv_utils.read_txt("prompts/context_qa_prompt.txt"))

    # Load examples
    examples = example_utils.read_examples(_INPUT_PATH.value)

    # Initialize model
    model = get_model(_MODEL_NAME.value)

    # Process examples
    outputs: List[example_utils.Example] = []
    for ex in tqdm.tqdm(examples):
        try:
            # Prepare prompt
            cur_prompt = prepare_prompt(context_prompt, ex.query)

            # Generate context
            context_output = model.generate(cur_prompt)

            # Update example and add to outputs
            ex.contexts = context_output
            outputs.append(ex)

            # Write example
            example_utils.write_examples(_OUTPUT_PATH.value, [ex], append=True)

        except Exception as e:
            print(f"Error processing example: {e}")
            continue


if __name__ == "__main__":
    app.run(main)
