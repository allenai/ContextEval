r"""Generate types for a set of queries.

Example usage:

INPUT_PATH=data/sampled_qa_contexts_gpt-4.jsonl
OUTPUT_PATH=data/all_queries_qtypes.jsonl
python3.10 main/generate_query_types.py \
--input_path=${INPUT_PATH} \
--output_path=${OUTPUT_PATH}
"""

import random
import sys
from typing import Any

import example_utils
import jsonl_utils
import models
import tqdm
import tsv_utils
from absl import app, flags

sys.path.append("contexteval/common")

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
    context_prompt = "\n".join(tsv_utils.read_txt("prompts/query_type_prompt.txt"))

    # Load examples
    examples = example_utils.read_examples(_INPUT_PATH.value)

    # Initialize model
    model = get_model()

    # Process examples
    outputs = []
    for ex in tqdm.tqdm(examples):
        try:
            # Prepare prompt
            cur_prompt = prepare_prompt(context_prompt, ex.query)

            # Generate query type
            qtype_output = model.generate(cur_prompt)

            # Add to outputs
            outputs.append({"query": ex.query, "query_type": qtype_output})

        except Exception as e:
            print(f"Error processing example: {e}")
            continue

    # Write outputs
    jsonl_utils.write(_OUTPUT_PATH.value, outputs)


if __name__ == "__main__":
    app.run(main)
