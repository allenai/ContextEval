r"""Generate contexts for a set of queries.

Example usage:

INPUT_PATH=data/sampled_qa_contexts_with_validation.jsonl
OUTPUT_PATH=data/sampled_qa_contexts_individual.jsonl
python3.10 main/generate_single_context.py \
--input_path=${INPUT_PATH} \
--output_path=${OUTPUT_PATH}
"""

import random
import sys
from typing import Any, List

from absl import app, flags

sys.path.append("contexteval/common")
import example_utils
import models
import tqdm
import tsv_utils

# Set random seed at module level
random.seed(423)

_INPUT_PATH = flags.DEFINE_string("input_path", "", "Path to the input file.")
_OUTPUT_PATH = flags.DEFINE_string("output_path", "", "Path to the output file.")


def get_model() -> Any:
    """Initialize and return GPT-4 model.

    Returns:
        Initialized GPT-4 model instance
    """
    return models.GPT4()


def prepare_prompt(prompt_template: str, query: str, contexts: str) -> str:
    """Prepare prompt by replacing placeholders.

    Args:
        prompt_template: Template string with placeholders
        query: Query to insert
        contexts: Contexts to insert

    Returns:
        Prepared prompt string
    """
    prompt = (prompt_template + ".")[:-1]
    prompt = prompt.replace("[QUERY]", query)
    prompt = prompt.replace("[CONTEXT]", contexts)
    return prompt


def main(unused_argv) -> None:
    # Load prompt template
    context_prompt = "\n".join(tsv_utils.read_txt("prompts/single_context_prompt.txt"))

    # Load examples
    examples = example_utils.read_examples(_INPUT_PATH.value)

    # Initialize model
    model = get_model()

    # Process examples
    outputs: List[example_utils.Example] = []
    for ex in tqdm.tqdm(examples):
        try:
            # Prepare prompt
            cur_prompt = prepare_prompt(context_prompt, ex.query, ex.contexts)

            # Generate sampled context
            sampled_context = model.generate(input_text=cur_prompt, max_len=2048)

            # Update example and add to outputs
            ex.sampled_context = sampled_context
            outputs.append(ex)

            # Write example
            example_utils.write_examples(_OUTPUT_PATH.value, [ex], append=True)

        except Exception as e:
            print(f"Error processing example: {e}")
            continue


if __name__ == "__main__":
    app.run(main)
