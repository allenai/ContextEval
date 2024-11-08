r"""Sample data for annotation.

Example usage:

OUTPUT_PATH=data/all_data_latest.jsonl
python3.10 main/sample_data.py \
--output_path=${OUTPUT_PATH}
"""

from absl import app
from absl import flags
from datasets import load_dataset

import collections
import sys
sys.path.append('contextual_eval/common')
import example_utils


_OUTPUT_PATH = flags.DEFINE_string(
    "output_path", "", "Path to the output file."
)


def load_chatbot_arena_data():
  chatbot_arena_data = load_dataset("lmsys/chatbot_arena_conversations")
  examples = []
  seen_queries = set()
  for i, ex in enumerate(chatbot_arena_data["train"]):
    # Sample only single-turn conversations for now
    if ex["turn"] != 1:
      continue
    query = ex["conversation_a"][0]["content"].strip()
    if not query or query in seen_queries:
      continue
    seen_queries.add(query)
    examples.append(
        example_utils.Example(
            query=query,
            model_names=[ex["model_a"], ex["model_b"]],
            model_responses=[ex["conversation_a"][1]["content"], ex["conversation_b"][1]["content"]],
            annotator_id="chatbot_arena_conversations_" + str(i),
            source="lmsys/chatbot_arena_conversations"
        )
    )
  return examples


def load_mtbench_data():
  mt_bench_data = load_dataset("lmsys/mt_bench_human_judgments")
  examples = []
  seen_queries = set()
  for i, ex in enumerate(mt_bench_data["human"]):
    if ex["turn"] != 1:
      continue
    query = ex["conversation_a"][0]["content"].strip()
    if not query or query in seen_queries:
      continue
    seen_queries.add(query)
    examples.append(
        example_utils.Example(
            query=query,
            model_names=[ex["model_a"], ex["model_b"]],
            model_responses=[ex["conversation_a"][1]["content"], ex["conversation_b"][1]["content"]],
            annotator_id="mt_bench_human_judgments_" + str(i),
            source="lmsys/mt_bench_human_judgments"
        )
    )
  return examples


def load_kiwi_data():
  kiwi_data = load_dataset("fangyuan/kiwi")
  examples = []
  seen_queries = set()
  for i, ex in enumerate(kiwi_data["train"]):
    query = ex["original_question"].strip()
    if not query or query in seen_queries:
      continue
    seen_queries.add(query)
    examples.append(
        example_utils.Example(
            query=query,
            model_names=[ex["model_name"]],
            model_responses=[ex["initial_answer"]],
            annotator_id="kiwi_" + str(i),
            source="fangyuan/kiwi"
        )
    )
  return examples


def load_expertqa_data():
  expertqa_data = load_dataset("cmalaviya/expertqa", "lfqa_domain")
  examples = []
  count = 0
  seen_queries = set()
  for split in ["train", "validation", "test"]:
    for ex in expertqa_data[split]:
      query = ex["question"].strip()
      if not query or query in seen_queries:
        continue
      seen_queries.add(query)
      examples.append(
          example_utils.Example(
              query=query,
              model_names=["model"],
              model_responses=[ex["answer"]],
              annotator_id="expertqa_" + str(count),
              source="cmalaviya/expertqa"
          )
      )
      count += 1
  return examples


def load_alpaca_data():
  alpaca_data = load_dataset("tatsu-lab/alpaca_eval", trust_remote_code=True)
  examples = []
  count = 0
  seen_queries = set()
  for split in ["eval"]:
    for ex in alpaca_data[split]:
      query = ex["instruction"].strip()
      if not query or query in seen_queries:
        continue
      seen_queries.add(query)
      examples.append(
          example_utils.Example(
              query=query,
              model_names=[ex["generator"]],
              model_responses=[ex["output"]],
              annotator_id="alpaca_" + str(count),
              source="tatsu-lab/alpaca_eval"
          )
      )
      count += 1
  return examples


def main(unused_argv) -> None:
  data = load_chatbot_arena_data()
  data += load_mtbench_data()
  data += load_kiwi_data()
  data += load_expertqa_data()
  data += load_alpaca_data()
  data = [ex for ex in data if len(ex.query.split()) > 3 and len(ex.query.split()) < 60]
  print("Number of examples:", len(data))
  print(collections.Counter([ex.source for ex in data]))
  example_utils.write_examples(_OUTPUT_PATH.value, data)


if __name__ == "__main__":
  app.run(main)
