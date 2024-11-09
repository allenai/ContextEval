#!/bin/bash

export OPENAI_ORGANIZATION=INSERT_ORG_NAME
export OPENAI_API_KEY=INSERT_API_KEY
export GOOGLE_API_KEY=INSERT_API_KEY
export ANTHROPIC_API_KEY=INSERT_API_KEY
export TOGETHER_API_KEY=INSERT_API_KEY

MODEL_NAME=$1

INPUT_PATH=data/sampled_qa_contexts_individual.jsonl
OUTPUT_PATH=data/sampled_responses_w_contexts_"${MODEL_NAME/\//_}".jsonl
python3.10 main/generate_responses.py \
--input_path=${INPUT_PATH} \
--output_path=${OUTPUT_PATH} \
--model_name=${MODEL_NAME} \
--w_context=True

INPUT_PATH=data/sampled_qa_contexts_individual.jsonl
OUTPUT_PATH=data/sampled_responses_wo_contexts_"${MODEL_NAME/\//_}".jsonl
python3.10 main/generate_responses.py \
--input_path=${INPUT_PATH} \
--output_path=${OUTPUT_PATH} \
--model_name=${MODEL_NAME} \
--w_context=False
