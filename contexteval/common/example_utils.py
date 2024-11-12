"""Utilities for reading and writing examples."""

import dataclasses
from typing import Optional, Sequence

import jsonl_utils


@dataclasses.dataclass(frozen=False)
class Example:
  """Represents an example for a task."""

  completed: Optional[bool] = False
  # Input query
  query: str = None
  # Query types
  query_types: str = None
  # Name of the models from which the output was sampled
  model_names: Sequence[str] = None
  # Model responses to the input query
  model_responses: Sequence[str] = None
  # Whether the query needs to be supplemented with context
  need_for_context: Optional[bool] = False
  # Contexts generated through model
  contexts: Optional[str] = None
  # Annotator ID of the person who annotated this example
  annotator_id: Optional[str] = None
  # Source of the input query
  source: Optional[str] = None
  # Model from which context was sampled
  context_model_source: Optional[int] = None
  # Sampled QA pairs from context (single answer for each question)
  sampled_context: Optional[str] = None

def read_examples(filepath):
  examples_json = jsonl_utils.read(filepath)
  examples = [Example(**ex) for ex in examples_json]
  return examples


def write_examples(filepath, examples, append=False):
  examples_json = [dataclasses.asdict(ex) for ex in examples]
  jsonl_utils.write(filepath, examples_json, append=append)
