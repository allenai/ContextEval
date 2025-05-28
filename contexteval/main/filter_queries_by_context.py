r"""Filter queries based on contextual attribute.

Example usage:
CONTEXT_NAME="Level of Detail"
python3.10 main/filter_queries_by_context.py \
    --query_path="data/all_data_latest.jsonl" \
    --context_name=${CONTEXT_NAME}
"""

import sys
from typing import Any, Dict, List, Optional, Tuple

import models
from absl import app, flags

sys.path.append("contexteval/common")
import ast

import example_utils
import jsonl_utils
import tqdm
import tsv_utils

_QUERY_PATH = flags.DEFINE_string(
    "query_path", "data/all_data_latest.jsonl", "Path to the query file."
)
_CONTEXT_NAME = flags.DEFINE_string("context_name", "", "Name of the contextual attribute.")


def get_model() -> Any:
    """Initialize and return GPT-4 model.

    Returns:
        Initialized GPT-4 model instance
    """
    return models.GPT4()


def extract_dictionary(input_string: str) -> Optional[Dict[str, Any]]:
    """Extract dictionary from model output string.

    Args:
        input_string: String containing model output

    Returns:
        Dictionary or None if parsing fails
    """
    try:
        search_str = "{"
        end_str = "}"
        dict_start = input_string.find(search_str)
        dict_end = input_string.find(end_str) + 1
        dict_str = input_string[dict_start:dict_end]
        dictionary = ast.literal_eval(dict_str)
        return dictionary
    except Exception:
        print(f"Error parsing dictionary from: {input_string}")
        return None


def prepare_prompt(prompt_template: str, query: str, question: str) -> str:
    """Prepare prompt by replacing placeholders.

    Args:
        prompt_template: Template string with placeholders
        query: Query to insert
        question: Question to insert

    Returns:
        Prepared prompt string
    """
    prompt = (prompt_template + ".")[:-1]
    prompt = prompt.replace("[QUERY]", query)
    prompt = prompt.replace("[QUESTION]", question)
    return prompt


def get_output_path(context: str) -> str:
    """Get output file path for context.

    Args:
        context: Context name

    Returns:
        Path to output file
    """
    return (
        f"data/context_analysis/{context.replace('/', '_').replace(' ', '_')}_filter_outputs.jsonl"
    )


def get_eval_judgement(
    prompt: str, model: Any, max_retries: int = 3
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Get evaluation judgement from model with retries.

    Args:
        prompt: Prompt to send to model
        model: Model instance to use
        max_retries: Maximum number of retry attempts

    Returns:
        Tuple of (raw output string, parsed dictionary or None)
    """
    for attempt in range(max_retries):
        try:
            output = model.generate(prompt)
            dict_out = extract_dictionary(output)
            if dict_out is not None:
                return output, dict_out
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to get evaluation judgement after {max_retries} attempts: {e}")
            continue
    return "", None


def load_context_questions() -> Dict[str, str]:
    """Load context to question mapping from file.

    Returns:
        Dictionary mapping context names to questions
    """
    contextual_attributes = tsv_utils.read_txt("data/contextual_attributes_qas.txt")
    context_to_question = {}
    for line in contextual_attributes:
        if line[0] == "*":
            cur_context = line[1:].strip()
        else:
            context_to_question[cur_context] = line.strip()
    return context_to_question


def main(unused_argv) -> None:
    # Load context questions
    context_to_question = load_context_questions()

    # Load examples
    examples = example_utils.read_examples(_QUERY_PATH.value)

    # Load prompt template
    filter_prompt = "\n".join(tsv_utils.read_txt("prompts/filter_queries_for_context_prompt.txt"))

    # Initialize model
    model = get_model()

    # Process each context
    for context, question in context_to_question.items():
        if context != _CONTEXT_NAME.value:
            continue

        outputs = []
        for ex in tqdm.tqdm(examples):
            try:
                # Prepare prompt
                cur_prompt = prepare_prompt(filter_prompt, ex.query, question)

                # Get evaluation judgement
                output, _ = get_eval_judgement(cur_prompt, model)

                # Add to outputs
                outputs.append(
                    {"query": ex.query, "question": question, "context": context, "output": output}
                )

            except Exception as e:
                print(f"Error processing example: {e}")
                continue

        # Write outputs
        output_path = get_output_path(context)
        jsonl_utils.write(output_path, outputs)


if __name__ == "__main__":
    app.run(main)
