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
  # conda run -n toxic_oracles_autodan python autodan_ga_eval.py --model "${model}" --data "./data/EvoTox/questions_reduced100.csv" --save_suffix "${model}" --num_steps 50
  # conda run -n toxic_oracles_autodan LOG_LEVEL=DEBUG python get_responses.py --model "${model}" --path "./results/autodan_ga/${model}_0_${model}.json"
  conda run -n toxic_oracles_autodan python autodan_hga_eval.py --model "${model}" --data "./data/EvoTox/questions_reduced100.csv" --num_steps 50
  conda run -n toxic_oracles_autodan LOG_LEVEL=DEBUG python get_responses.py --model "${model}" --path "./results/autodan_hga/${model}_0_normal.json"
done
