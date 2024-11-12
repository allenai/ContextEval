#!/bin/bash

export OPENAI_ORGANIZATION=INSERT_ORG_NAME
export OPENAI_API_KEY=INSERT_API_KEY
export GOOGLE_API_KEY=INSERT_API_KEY
export ANTHROPIC_API_KEY=INSERT_API_KEY
export TOGETHER_API_KEY=INSERT_API_KEY

INPUT_PATH=data/all_data_latest_filtered.jsonl
MODEL_NAME=$1
OUTPUT_PATH=data/sampled_qa_contexts_${MODEL_NAME}.jsonl
python3.10 main/generate_contexts.py \
--input_path=${INPUT_PATH} \
--output_path=${OUTPUT_PATH} \
--model_name=${MODEL_NAME}

INPUT_PATH=data/sampled_qa_contexts_gpt-4.jsonl
OUTPUT_PATH=data/sampled_qa_contexts_with_validation.jsonl
python3.10 main/generate_context_validation.py \
--input_path=${INPUT_PATH} \
--output_path=${OUTPUT_PATH}

INPUT_PATH=data/sampled_qa_contexts_with_validation.jsonl
OUTPUT_PATH=data/sampled_qa_contexts_individual.jsonl
python3.10 main/generate_single_context.py \
--input_path=${INPUT_PATH} \
--output_path=${OUTPUT_PATH}