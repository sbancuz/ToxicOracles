#!/bin/bash
# set -xe

# Declare files array
models=(
  "llama3"
  "mistral"
  "vicuna"
  "vicunaUC"
)

# Change to AutoDAN repo
cd submodules/AutoDAN
# Loop over the tested models
for model in "${models[@]}"; do
  LOGLEVEL=DEBUG conda run -n toxic_oracles_autodan python autodan_hga_eval.py --model "${model}" --data "./data/EvoTox/questions_reduced100.csv" --num_steps 50
  LOGLEVEL=DEBUG conda run -n toxic_oracles_autodan python get_responses.py --model "${model}" --path "./results/autodan_hga/"
done
