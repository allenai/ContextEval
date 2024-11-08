#!/bin/bash

export OPENAI_ORGANIZATION=INSERT_ORG_NAME
export OPENAI_API_KEY=INSERT_API_KEY
export GOOGLE_API_KEY=INSERT_API_KEY
export ANTHROPIC_API_KEY=INSERT_API_KEY
export TOGETHER_API_KEY=INSERT_API_KEY

EVAL_MODEL=$1
CANDIDATE_ONE=claude-3.5-sonnet
CANDIDATE_TWO=meta-llama_Meta-Llama-3.1-405B-Instruct-Turbo

# Setting CtxGen-CtxEval
CANDIDATE_ONE_PATH=data/sampled_responses_w_contexts_${CANDIDATE_ONE}.jsonl
CANDIDATE_TWO_PATH=data/sampled_responses_w_contexts_${CANDIDATE_TWO}.jsonl
OUTPUT_PATH=data/sampled_evals_w_contexts_"${EVAL_MODEL/\//_}"-judge_${CANDIDATE_ONE}_vs_${CANDIDATE_TWO}.jsonl
W_CONTEXT=True
python3.10 main/generate_pairwise_evals.py \
--candidate_one_path=${CANDIDATE_ONE_PATH} \
--candidate_two_path=${CANDIDATE_TWO_PATH} \
--output_path=${OUTPUT_PATH} \
--eval_model=${EVAL_MODEL} \
--w_context=${W_CONTEXT}

# Setting NoCtxGen-NoCtxEval
CANDIDATE_ONE_PATH=data/sampled_responses_wo_contexts_${CANDIDATE_ONE}.jsonl
CANDIDATE_TWO_PATH=data/sampled_responses_wo_contexts_${CANDIDATE_TWO}.jsonl
OUTPUT_PATH=data/sampled_evals_wo_contexts_"${EVAL_MODEL/\//_}"-judge_${CANDIDATE_ONE}_vs_${CANDIDATE_TWO}.jsonl
W_CONTEXT=False
python3.10 main/generate_pairwise_evals.py \
--candidate_one_path=${CANDIDATE_ONE_PATH} \
--candidate_two_path=${CANDIDATE_TWO_PATH} \
--output_path=${OUTPUT_PATH} \
--eval_model=${EVAL_MODEL} \
--w_context=${W_CONTEXT}

# Setting NoCtxGen-CtxEval
CANDIDATE_ONE_PATH=data/sampled_responses_wo_contexts_${CANDIDATE_ONE}.jsonl
CANDIDATE_TWO_PATH=data/sampled_responses_wo_contexts_${CANDIDATE_TWO}.jsonl
OUTPUT_PATH=data/sampled_evals_gen_wo_contexts_eval_w_contexts_"${EVAL_MODEL/\//_}"-judge_${CANDIDATE_ONE}_vs_${CANDIDATE_TWO}.jsonl
W_CONTEXT=True
python3.10 main/generate_pairwise_evals.py \
--candidate_one_path=${CANDIDATE_ONE_PATH} \
--candidate_two_path=${CANDIDATE_TWO_PATH} \
--output_path=${OUTPUT_PATH} \
--eval_model=${EVAL_MODEL} \
--w_context=${W_CONTEXT}
