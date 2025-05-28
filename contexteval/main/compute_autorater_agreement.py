r"""Computes the agreement and and win-rates between models based on their evaluation judgements.

Example usage:

EVAL_OUTPUTS=data/sampled_evals_wo_contexts_EVAL_MODEL-judge_gpt-4_vs_gemini-1.5-pro.jsonl
python3.10 main/compute_agreement.py \
--eval_outputs=${EVAL_OUTPUTS}
"""

import ast
import collections
import sys

import numpy as np
from absl import app, flags

sys.path.append("contexteval/common")
import jsonl_utils
from scipy.stats import ttest_ind, ttest_rel
from statsmodels.stats import inter_rater as irr

_EVAL_OUTPUTS = flags.DEFINE_string(
    "eval_outputs", "", "Path to the files containing the evaluation judgements of multiple models."
)

TIE_WORD = "Tie"


def pairwise_agreement(judgements1, judgements2):
    """Calculate the agreement between two sets of judgements."""
    agree_count = sum(1 for x, y in zip(judgements1, judgements2) if x == y)
    return agree_count / len(judgements1)


def extract_dictionary(input_string):
    try:
        input_string = input_string.replace("*", "").strip()
        search_str = "output: "
        end_str = "}"
        dict_start = input_string.find(search_str) + len(search_str)
        dict_end = input_string.find(end_str) + 1
        dict_str = input_string[dict_start:dict_end]
        dictionary = ast.literal_eval(dict_str)
        return dictionary
    except Exception as e:
        # print(input_string)
        print(f"Error occurred: {e}")
        return None


def get_eval_judgements(eval_models, filename_template):
    eval_judgements = {}
    invalid: dict[str, int] = collections.defaultdict(int)
    for model in eval_models:
        filepath = filename_template.replace("EVAL_MODEL", model)
        print(filepath)
        eval_data = jsonl_utils.read(filepath)
        eval_outputs = []
        for ex in eval_data:
            dict_out = extract_dictionary(ex["eval_judgement"])
            if dict_out is None or "judgement" not in dict_out:
                if "**Response 1**" in ex["eval_judgement"] or "Response 1" in ex["eval_judgement"]:
                    dict_out = {"judgement": "Response 1"}
                elif (
                    "**Response 2**" in ex["eval_judgement"] or "Response 2" in ex["eval_judgement"]
                ):
                    dict_out = {"judgement": "Response 2"}
            if dict_out is None or "judgement" not in dict_out:
                invalid[model] += 1
                eval_outputs.append("Invalid output")
                continue
            if ex["rand_choice"] == 1:
                if "Response 1" in dict_out["judgement"]:
                    eval_outputs.append("Candidate 1")
                elif "Response 2" in dict_out["judgement"]:
                    eval_outputs.append("Candidate 2")
                elif TIE_WORD in dict_out["judgement"]:
                    eval_outputs.append(TIE_WORD)
                else:
                    print(dict_out["judgement"])
                    invalid[model] += 1
                    eval_outputs.append("Invalid output")

            elif ex["rand_choice"] == 2:
                if "Response 1" in dict_out["judgement"]:
                    eval_outputs.append("Candidate 2")
                elif "Response 2" in dict_out["judgement"]:
                    eval_outputs.append("Candidate 1")
                elif TIE_WORD in dict_out["judgement"]:
                    eval_outputs.append(TIE_WORD)
                else:
                    invalid[model] += 1
                    eval_outputs.append("Invalid output")

        eval_judgements[model] = eval_outputs
    print("Invalid output counts: ", invalid)
    return eval_judgements


def calculate_length_win_rate(filename_template, eval_models):
    # Compute win rates based on the length of the responses.
    filepath = filename_template.replace("EVAL_MODEL", eval_models[0])
    eval_data = jsonl_utils.read(filepath)
    length_win_rates = []
    for ex in eval_data:
        candidate_one_length = len(ex["candidate_one_response"].split())
        candidate_two_length = len(ex["candidate_two_response"].split())
        length_win_rates.append(candidate_one_length > candidate_two_length)
    print("Candidate 1 win rate by length: ", sum(length_win_rates) / len(length_win_rates))


def get_agreement(eval_judgements, eval_models, selected_idxs=None):
    # Compute the agreement between models.
    instance_wise_agreements = []
    non_tie_instance_wise_agreements = []
    majority_judgements = []
    if not selected_idxs:
        selected_idxs = range(len(eval_judgements[eval_models[0]]))
    tot_examples = len(selected_idxs)
    fleiss_table = np.zeros((len(selected_idxs), 3))
    for i, idx in enumerate(selected_idxs):
        model_judgements = [
            eval_judgements[model][idx]
            if eval_judgements[model][idx] != "Invalid output"
            else TIE_WORD
            for model in eval_models
        ]
        counter = collections.Counter(model_judgements)
        most_common_judgement, most_common_judgement_count = counter.most_common(1)[0]
        instance_wise_agreements.append(most_common_judgement_count / len(model_judgements))
        if most_common_judgement != TIE_WORD:
            non_tie_instance_wise_agreements.append(
                most_common_judgement_count / len(model_judgements)
            )
        if most_common_judgement_count / len(model_judgements) > 0.5:
            majority_judgements.append(most_common_judgement)
        fleiss_table[i] = [
            model_judgements.count("Candidate 1"),
            model_judgements.count("Candidate 2"),
            model_judgements.count(TIE_WORD),
        ]

    print("Total examples: ", tot_examples)
    print("Agreement between models: ", sum(instance_wise_agreements) / tot_examples)
    print(
        "Non-tie agreement between models: ",
        sum(non_tie_instance_wise_agreements) / len(non_tie_instance_wise_agreements),
    )
    print(
        "Candidate 1 win rate: ",
        sum([1 for x in majority_judgements if x == "Candidate 1"]) / len(majority_judgements),
    )
    print(
        "Candidate 2 win rate: ",
        sum([1 for x in majority_judgements if x == "Candidate 2"]) / len(majority_judgements),
    )
    print(
        "Tie rate: ",
        sum([1 for x in majority_judgements if x == TIE_WORD]) / len(majority_judgements),
    )
    print("Fleiss Kappa: ", irr.fleiss_kappa(fleiss_table, method="rand"))
    return instance_wise_agreements, non_tie_instance_wise_agreements


def main(unused_argv) -> None:
    eval_models = []

    if "claude-3.5-sonnet_vs_meta-llama_Meta-Llama-3.1-405B-Instruct-Turbo" in _EVAL_OUTPUTS.value:
        eval_models = [
            "gpt-4",
            "gemini-1.5-pro",
            "google_gemma-2-27b-it",
            "jamba-1.5-large",
            "Qwen_Qwen2-72B-Instruct",
        ]
    elif "google_gemma-2-27b-it_vs_jamba-1.5-large" in _EVAL_OUTPUTS.value:
        eval_models = [
            "claude-3.5-sonnet",
            "meta-llama_Meta-Llama-3.1-405B-Instruct-Turbo",
            "gpt-4",
            "gemini-1.5-pro",
            "Qwen_Qwen2-72B-Instruct",
        ]
    elif "gpt-4_vs_gemini-1.5-flash-exp-0827" in _EVAL_OUTPUTS.value:
        eval_models = [
            "claude-3.5-sonnet",
            "meta-llama_Meta-Llama-3.1-405B-Instruct-Turbo",
            "google_gemma-2-27b-it",
            "jamba-1.5-large",
            "Qwen_Qwen2-72B-Instruct",
        ]

    selected_idxs = None

    wo_ctx_eval_judgements = get_eval_judgements(eval_models, _EVAL_OUTPUTS.value)
    print("\n\nAgreement between models for gen_wo_ctx_eval_wo_ctx:")
    wo_ctx_instance_wise_agreements, wo_ctx_non_tie_instance_wise_agreements = get_agreement(
        wo_ctx_eval_judgements, eval_models, selected_idxs
    )

    EVAL_OUTPUTS = _EVAL_OUTPUTS.value.replace("wo_contexts", "gen_wo_contexts_eval_w_contexts")
    gen_wo_ctx_eval_w_ctx_eval_judgements = get_eval_judgements(eval_models, EVAL_OUTPUTS)
    print("\n\nAgreement between models for gen_wo_ctx_eval_w_ctx:")
    (
        gen_wo_ctx_eval_w_ctx_instance_wise_agreements,
        gen_wo_ctx_eval_w_ctx_non_tie_instance_wise_agreements,
    ) = get_agreement(gen_wo_ctx_eval_w_ctx_eval_judgements, eval_models, selected_idxs)

    EVAL_OUTPUTS = _EVAL_OUTPUTS.value.replace("wo_contexts", "w_contexts")
    w_ctx_eval_judgements = get_eval_judgements(eval_models, EVAL_OUTPUTS)
    print("\n\nAgreement between models for gen_w_ctx_eval_w_ctx:")
    w_ctx_instance_wise_agreements, w_ctx_non_tie_instance_wise_agreements = get_agreement(
        w_ctx_eval_judgements, eval_models, selected_idxs
    )

    print("\nLength win rates (w/o ctx)")
    calculate_length_win_rate(_EVAL_OUTPUTS.value, eval_models)
    EVAL_OUTPUTS = _EVAL_OUTPUTS.value.replace("wo_contexts", "w_contexts")
    print("\nLength win rates (w ctx)")
    calculate_length_win_rate(EVAL_OUTPUTS, eval_models)

    sig = [True] * 6
    print("\n\nT-test statistics:")

    ttest_res = ttest_rel(
        wo_ctx_instance_wise_agreements, gen_wo_ctx_eval_w_ctx_instance_wise_agreements
    )
    print("T-test between gen_wo_ctx_eval_wo_ctx and gen_wo_ctx_eval_w_ctx (with ties):", ttest_res)
    if ttest_res.pvalue > 0.05:
        sig[0] = False

    ttest_res = ttest_ind(
        wo_ctx_non_tie_instance_wise_agreements,
        gen_wo_ctx_eval_w_ctx_non_tie_instance_wise_agreements,
    )
    print(
        "T-test between gen_wo_ctx_eval_wo_ctx and gen_wo_ctx_eval_w_ctx (without ties):", ttest_res
    )
    if ttest_res.pvalue > 0.05:
        sig[1] = False

    ttest_res = ttest_rel(wo_ctx_instance_wise_agreements, w_ctx_instance_wise_agreements)
    print("T-test between gen_wo_ctx_eval_wo_ctx and gen_w_ctx_eval_w_ctx (with ties):", ttest_res)
    if ttest_res.pvalue > 0.05:
        sig[2] = False

    ttest_res = ttest_ind(
        wo_ctx_non_tie_instance_wise_agreements, w_ctx_non_tie_instance_wise_agreements
    )
    print(
        "T-test between gen_wo_ctx_eval_wo_ctx and gen_w_ctx_eval_w_ctx (without ties):", ttest_res
    )
    if ttest_res.pvalue > 0.05:
        sig[3] = False

    ttest_res = ttest_rel(
        gen_wo_ctx_eval_w_ctx_instance_wise_agreements, w_ctx_instance_wise_agreements
    )
    print("T-test between gen_wo_ctx_eval_w_ctx and gen_w_ctx_eval_w_ctx (with ties):", ttest_res)
    if ttest_res.pvalue > 0.05:
        sig[4] = False

    ttest_res = ttest_ind(
        gen_wo_ctx_eval_w_ctx_non_tie_instance_wise_agreements,
        w_ctx_non_tie_instance_wise_agreements,
    )
    print(
        "T-test between gen_wo_ctx_eval_w_ctx and gen_w_ctx_eval_w_ctx (without ties):", ttest_res
    )
    if ttest_res.pvalue > 0.05:
        sig[5] = False

    if sum(sig) == 6:
        print(
            "\u2713\u2713\u2713\u2713\u2713\u2713 All differences are significant!!! \u2713\u2713\u2713\u2713\u2713\u2713"
        )
    else:
        print(f"*******************{sum(sig)} differences are significant!!!*******************")


if __name__ == "__main__":
    app.run(main)
