r"""Computes the agreement and win-rates between human annotators based on their evaluation judgements.

Example usage:

EVAL_JUDGMENTS=data/human_judgements/gpt-4_vs_gemini-1.5-flash-exp-0827.jsonl
python3.10 main/compute_agreement.py \
--eval_outputs=${EVAL_OUTPUTS}
"""

from absl import app
from absl import flags

import collections
import json
import numpy as np
import sys
sys.path.append('contexteval/common')
from scipy.stats import sem, t, ttest_ind, ttest_rel


_EVAL_JUDGMENTS = flags.DEFINE_string(
    "eval_judgments", "", "Path to the files containing the evaluation judgements of humans."
)

def main(unused_argv) -> None:
  
  # Read human judgements.

  with open(_EVAL_JUDGMENTS.value) as f:
    human_judgements = json.load(f)

  model_one, model_two = _EVAL_JUDGMENTS.value.split("/")[-1].replace(".jsonl", "").split("_vs_")

  repeated_data = collections.defaultdict(list)
  for ex in human_judgements:
    repeated_data[ex["example_id"].rsplit("_rep", 1)[0].strip()].append(ex)

  preferences_by_setting = collections.defaultdict(list)
  instance_wise_agreements = collections.defaultdict(list)
  non_tie_instance_wise_agreements = collections.defaultdict(list)
  constraint_diff_instance_wise_agreements = collections.defaultdict(list)
  constraint_instance_wise_agreements = collections.defaultdict(list)
  avg_constraint_satisfied_diff = []
  
  filtered_ex_ids = set()
  for ex_id, exs in repeated_data.items():    
    model_preferences = []
    for ex in exs:
        if ex["overall_preference"] == "Response 1":
            model_preferences.append("Candidate 1" if ex["model_1"] == model_one else "Candidate 2")
        elif ex["overall_preference"] == "Response 2":
            model_preferences.append("Candidate 1" if ex["model_2"] == model_one else "Candidate 2")
        else:
            model_preferences.append("Tie")
          
    counter = collections.Counter(model_preferences)
    most_common_judgement, most_common_judgement_count = counter.most_common(1)[0]
    if most_common_judgement_count / len(model_preferences) > 0.5:
      preferences_by_setting[exs[0]["setting"]].append(most_common_judgement)
    instance_wise_agreements[exs[0]["setting"]].append(most_common_judgement_count / len(model_preferences))
    if most_common_judgement != "Tie":
        non_tie_instance_wise_agreements[exs[0]["setting"]].append(most_common_judgement_count / len(model_preferences))
    if "gen_w_ctx_eval_w_ctx" in exs[0]["setting"] or "gen_wo_ctx_eval_w_ctx" in exs[0]["setting"]:
        constraints_satisfied_diff = []
        for ex in exs:
            satisfied_1 = [qa["satisfied_1"] for qa in ex["follow_up_qas"]]
            satisfied_2 = [qa["satisfied_2"] for qa in ex["follow_up_qas"]]
            if None in satisfied_1 or None in satisfied_2:
              continue
            num_satisfied_1 = sum(satisfied_1)
            num_satisfied_2 = sum(satisfied_2)
            constraints_satisfied_diff.append(num_satisfied_1 - num_satisfied_2)
        for i in range(len(exs[0]["follow_up_qas"])):
            sat_1_ratings = [ex["follow_up_qas"][i]["satisfied_1"] for ex in exs]
            sat_2_ratings = [ex["follow_up_qas"][i]["satisfied_2"] for ex in exs]
            counter = collections.Counter(sat_1_ratings)
            most_common_judgement_1, most_common_judgement_count_1 = counter.most_common(1)[0]
            constraint_instance_wise_agreements[exs[0]["setting"]].append(most_common_judgement_count_1 / len(exs))
            counter = collections.Counter(sat_2_ratings)
            most_common_judgement_2, most_common_judgement_count_2 = counter.most_common(1)[0]
            constraint_instance_wise_agreements[exs[0]["setting"]].append(most_common_judgement_count_2 / len(exs))

        avg_constraint_satisfied_diff.append(np.mean(np.abs(constraints_satisfied_diff)))
        if np.abs(np.mean(constraints_satisfied_diff)) >= 1.0:
            constraint_diff_instance_wise_agreements[exs[0]["setting"]].append(most_common_judgement_count / len(model_preferences))
    
    filtered_ex_ids.add(ex_id.split("_")[0])
    print(ex_id, most_common_judgement, most_common_judgement_count / len(model_preferences))

  filtered_ex_ids = sorted(list(filtered_ex_ids), reverse=True)

  print("\nAverage constraint satisfied difference:", np.mean(avg_constraint_satisfied_diff), np.std(avg_constraint_satisfied_diff), len(avg_constraint_satisfied_diff))
  print("\nHuman-Majority Win Rates:")
  for k, v in preferences_by_setting.items():
      print(k, "(", len(v), ")")
      counter = collections.Counter(v)
      total = sum(counter.values())
      percentages = {key: (value / total) * 100 for key, value in counter.items()}
      print(percentages)

  print("\nHuman-Human Agreement Rates:")
  for k, v in instance_wise_agreements.items():
      print(k, len(v))
      print(sum(v) / len(v))
      print(t.interval(confidence=0.95, df=len(v)-1, loc=np.mean(v), scale=sem(v)))

  print(ttest_rel(instance_wise_agreements["gen_wo_ctx_eval_wo_ctx"], instance_wise_agreements["gen_wo_ctx_eval_w_ctx"]))
  print(ttest_rel(instance_wise_agreements["gen_wo_ctx_eval_wo_ctx"], instance_wise_agreements["gen_w_ctx_eval_w_ctx"]))

  print("\nHuman-Human Non-Tie Agreement Rates:")
  for k, v in non_tie_instance_wise_agreements.items():
      print(k, len(v))
      print(sum(v) / len(v))

  print(ttest_ind(non_tie_instance_wise_agreements["gen_wo_ctx_eval_wo_ctx"], non_tie_instance_wise_agreements["gen_wo_ctx_eval_w_ctx"]))
  print(ttest_ind(non_tie_instance_wise_agreements["gen_wo_ctx_eval_wo_ctx"], non_tie_instance_wise_agreements["gen_w_ctx_eval_w_ctx"]))

  print("\nHuman-Human Constraint Difference Agreement Rates:")
  for k, v in constraint_diff_instance_wise_agreements.items():
      print(k, len(v))
      print(sum(v) / len(v))

  print("\nHuman-Human Constraint Agreement Rates:")
  for k, v in constraint_instance_wise_agreements.items():
      print(k, len(v))
      print(sum(v) / len(v))


if __name__ == "__main__":
  app.run(main)
