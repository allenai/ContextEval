r"""Filter queries based on well-formedness.

Example usage:

INPUT_PATH=data/all_data_latest.jsonl
OUTPUT_PATH=data/all_data_latest_filtered.jsonl
python3.10 main/filter_queries.py \
--input_path=${INPUT_PATH} \
--output_path=${OUTPUT_PATH}
"""

import collections
import random
import sys
from typing import Any, Dict, List

import models
from absl import app, flags

sys.path.append("contexteval/common")
import example_utils
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


def sample_examples(
    examples: List[example_utils.Example], num_ex_per_dataset: Dict[str, int]
) -> List[example_utils.Example]:
    """Sample examples from each dataset based on well-formedness.

    Args:
        examples: List of examples to sample from
        num_ex_per_dataset: Dictionary mapping dataset names to number of examples to sample

    Returns:
        List of sampled examples
    """
    # Load prompt template
    context_prompt = "\n".join(tsv_utils.read_txt("prompts/well_formed_query_check_prompt.txt"))

    # Initialize model
    model = get_model()

    # Group examples by dataset
    examples_by_dataset = collections.defaultdict(list)
    for ex in examples:
        examples_by_dataset[ex.source].append(ex)

    # Sample examples
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
            try:
                # Prepare prompt
                cur_prompt = prepare_prompt(context_prompt, ex.query)

                # Check if query is well-formed
                output = model.generate(cur_prompt)
                if "Yes" in output:
                    sampled_examples.append(ex)
                    samples_from_dataset += 1
                    print(f"Sampled {samples_from_dataset} examples from {dataset}")
                    if samples_from_dataset == examples_to_sample:
                        break

            except Exception as e:
                print(f"Error processing example: {e}")
                continue

    return sampled_examples


def main(unused_argv) -> None:
    # Define number of examples to sample per dataset
    num_ex_per_dataset = {
        "lmsys/chatbot_arena_conversations": 2500,
        "tatsu-lab/alpaca_eval": 800,
        "lmsys/mt_bench_human_judgments": 100,
        "cmalaviya/expertqa": 300,
        "fangyuan/kiwi": 100,
    }

    # Load examples
    examples = example_utils.read_examples(_INPUT_PATH.value)

    # Sample examples
    sampled_examples = sample_examples(examples, num_ex_per_dataset)

    # Write examples
    example_utils.write_examples(_OUTPUT_PATH.value, sampled_examples)


if __name__ == "__main__":
    app.run(main)
