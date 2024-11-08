#!/bin/bash

export OPENAI_ORGANIZATION=INSERT_ORG_NAME
export OPENAI_API_KEY=INSERT_API_KEY
export GOOGLE_API_KEY=INSERT_API_KEY
export ANTHROPIC_API_KEY=INSERT_API_KEY
export TOGETHER_API_KEY=INSERT_API_KEY

CONTEXT_NAMES=("Level of Detail" "User Expertise" "Length" "Format of response" "Style" "Intended Audience" "Geographical / Regional Context" "Cultural Context" "Age Group" "Economic Context" "Political Context" "Gender")
CONTEXT_INDEX=$1

echo "Filtering queries for context: ${CONTEXT_NAMES[$CONTEXT_INDEX]}"
python3 main/filter_queries_by_context.py --context_name "${CONTEXT_NAMES[$CONTEXT_INDEX]}"

MODEL_NAME="gpt-4"
echo "Generating responses for context: ${CONTEXT_NAMES[$CONTEXT_INDEX]}"
python3 main/generate_analysis_responses.py --context_name "${CONTEXT_NAMES[$CONTEXT_INDEX]}"

echo "Generating evals for context: ${CONTEXT_NAMES[$CONTEXT_INDEX]}"
python3 main/generate_analysis_evals.py --context_name "${CONTEXT_NAMES[$CONTEXT_INDEX]}"
