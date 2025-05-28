r"""Computes the agreement and win-rates between human annotators based on their evaluation judgements.

Example usage:

EVAL_JUDGMENTS=data/human_judgements/gpt-4_vs_gemini-1.5-flash-exp-0827.jsonl
python3.10 main/compute_agreement.py \
--eval_outputs=${EVAL_OUTPUTS}
"""

import collections
import json
import sys
from typing import Any, Dict, List, Set

import numpy as np
from absl import app, flags

sys.path.append("contexteval/common")
from collections import defaultdict

from scipy.stats import sem, t, ttest_ind, ttest_rel

_EVAL_JUDGMENTS = flags.DEFINE_string(
    "eval_judgments", "", "Path to the files containing the evaluation judgements of humans."
)


def main(unused_argv) -> None:
    # Read human judgements.
    with open(_EVAL_JUDGMENTS.value) as f:
        human_judgements = json.load(f)

    model_one, model_two = _EVAL_JUDGMENTS.value.split("/")[-1].replace(".jsonl", "").split("_vs_")

    repeated_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ex in human_judgements:
        repeated_data[ex["example_id"].rsplit("_rep", 1)[0].strip()].append(ex)

    preferences_by_setting: Dict[str, List[str]] = defaultdict(list)
    instance_wise_agreements: Dict[str, List[float]] = defaultdict(list)
    non_tie_instance_wise_agreements: Dict[str, List[float]] = defaultdict(list)
    constraint_diff_instance_wise_agreements: Dict[str, List[float]] = defaultdict(list)
    constraint_instance_wise_agreements: Dict[str, List[float]] = defaultdict(list)
    avg_constraint_satisfied_diff: List[float] = []

    filtered_ex_ids: Set[str] = set()
    for ex_id, exs in repeated_data.items():
        model_preferences: List[str] = []
        for ex in exs:
            if ex["overall_preference"] == "Response 1":
                model_preferences.append(
                    "Candidate 1" if ex["model_1"] == model_one else "Candidate 2"
                )
            elif ex["overall_preference"] == "Response 2":
                model_preferences.append(
                    "Candidate 1" if ex["model_2"] == model_one else "Candidate 2"
                )
            else:
                model_preferences.append("Tie")

        counter = collections.Counter(model_preferences)
        most_common_judgement, most_common_judgement_count = counter.most_common(1)[0]
        if most_common_judgement_count / len(model_preferences) > 0.5:
            preferences_by_setting[exs[0]["setting"]].append(most_common_judgement)
        instance_wise_agreements[exs[0]["setting"]].append(
            most_common_judgement_count / len(model_preferences)
        )
        if most_common_judgement != "Tie":
            non_tie_instance_wise_agreements[exs[0]["setting"]].append(
                most_common_judgement_count / len(model_preferences)
            )
        if (
            "gen_w_ctx_eval_w_ctx" in exs[0]["setting"]
            or "gen_wo_ctx_eval_w_ctx" in exs[0]["setting"]
        ):
            constraints_satisfied_diff: List[float] = []
            for ex in exs:
                satisfied_1: List[bool] = [
                    bool(qa["satisfied_1"])
                    for qa in ex["follow_up_qas"]
                    if qa["satisfied_1"] is not None
                ]
                satisfied_2: List[bool] = [
                    bool(qa["satisfied_2"])
                    for qa in ex["follow_up_qas"]
                    if qa["satisfied_2"] is not None
                ]
                if not satisfied_1 or not satisfied_2:
                    continue
                num_satisfied_1 = sum(satisfied_1)
                num_satisfied_2 = sum(satisfied_2)
                constraints_satisfied_diff.append(float(num_satisfied_1 - num_satisfied_2))
            for i in range(len(exs[0]["follow_up_qas"])):
                sat_1_ratings: List[bool] = [
                    bool(ex["follow_up_qas"][i]["satisfied_1"])
                    for ex in exs
                    if ex["follow_up_qas"][i]["satisfied_1"] is not None
                ]
                sat_2_ratings: List[bool] = [
                    bool(ex["follow_up_qas"][i]["satisfied_2"])
                    for ex in exs
                    if ex["follow_up_qas"][i]["satisfied_2"] is not None
                ]
                if not sat_1_ratings or not sat_2_ratings:
                    continue
                counter_1 = collections.Counter(str(x) for x in sat_1_ratings)
                most_common_judgement_1, most_common_judgement_count_1 = counter_1.most_common(1)[0]
                float_agreement_1 = float(most_common_judgement_count_1) / float(len(exs))
                constraint_instance_wise_agreements[exs[0]["setting"]].append(float_agreement_1)
                counter_2 = collections.Counter(str(x) for x in sat_2_ratings)
                most_common_judgement_2, most_common_judgement_count_2 = counter_2.most_common(1)[0]
                float_agreement_2 = float(most_common_judgement_count_2) / float(len(exs))
                constraint_instance_wise_agreements[exs[0]["setting"]].append(float_agreement_2)

            if constraints_satisfied_diff:
                avg_constraint_satisfied_diff.append(np.mean(np.abs(constraints_satisfied_diff)))
                if np.abs(np.mean(constraints_satisfied_diff)) >= 1.0:
                    constraint_diff_instance_wise_agreements[exs[0]["setting"]].append(
                        most_common_judgement_count / len(model_preferences)
                    )

        filtered_ex_ids.add(ex_id.split("_")[0])
        print(ex_id, most_common_judgement, most_common_judgement_count / len(model_preferences))

    filtered_ex_ids = set(sorted(list(filtered_ex_ids), reverse=True))

    print(
        "\nAverage constraint satisfied difference:",
        np.mean(avg_constraint_satisfied_diff),
        np.std(avg_constraint_satisfied_diff),
        len(avg_constraint_satisfied_diff),
    )
    print("\nHuman-Majority Win Rates:")
    for k, v in preferences_by_setting.items():
        print(k, "(", len(v), ")")
        counter = collections.Counter(v)
        total = sum(counter.values())
        percentages = {key: (value / total) * 100 for key, value in counter.items()}
        print(percentages)

    print("\nHuman-Human Agreement Rates:")
    for k, v_agree in instance_wise_agreements.items():
        v_agree_float: List[float] = [float(x) for x in v_agree]
        print(k, len(v_agree_float))
        print(sum(v_agree_float) / len(v_agree_float) if v_agree_float else 0.0)
        print(
            t.interval(
                confidence=0.95,
                df=len(v_agree_float) - 1,
                loc=np.mean(v_agree_float),
                scale=sem(v_agree_float),
            )
        )

    print(
        ttest_rel(
            instance_wise_agreements["gen_wo_ctx_eval_wo_ctx"],
            instance_wise_agreements["gen_wo_ctx_eval_w_ctx"],
        )
    )
    print(
        ttest_rel(
            instance_wise_agreements["gen_wo_ctx_eval_wo_ctx"],
            instance_wise_agreements["gen_w_ctx_eval_w_ctx"],
        )
    )

    print("\nHuman-Human Non-Tie Agreement Rates:")
    for k, v_nontie in non_tie_instance_wise_agreements.items():
        v_nontie_float: List[float] = [float(x) for x in v_nontie]
        print(k, len(v_nontie_float))
        print(sum(v_nontie_float) / len(v_nontie_float) if v_nontie_float else 0.0)

    print(
        ttest_ind(
            non_tie_instance_wise_agreements["gen_wo_ctx_eval_wo_ctx"],
            non_tie_instance_wise_agreements["gen_wo_ctx_eval_w_ctx"],
        )
    )
    print(
        ttest_ind(
            non_tie_instance_wise_agreements["gen_wo_ctx_eval_wo_ctx"],
            non_tie_instance_wise_agreements["gen_w_ctx_eval_w_ctx"],
        )
    )

    print("\nHuman-Human Constraint Difference Agreement Rates:")
    for k, v_diff in constraint_diff_instance_wise_agreements.items():
        v_diff_float: List[float] = [float(x) for x in v_diff]
        print(k, len(v_diff_float))
        print(sum(v_diff_float) / len(v_diff_float) if v_diff_float else 0.0)

    print("\nHuman-Human Constraint Agreement Rates:")
    for k, v_cons in constraint_instance_wise_agreements.items():
        v_cons_float: List[float] = [float(x) for x in v_cons]
        print(k, len(v_cons_float))
        print(sum(v_cons_float) / len(v_cons_float) if v_cons_float else 0.0)


if __name__ == "__main__":
    app.run(main)
