#!/bin/bash
# set -xe

# Declare data arrays
model_ids=(
  "llama3"
  "mistral"
  "vicuna"
  "vicunaUC"
)

# Loop over the models
for model_id in "${model_ids[@]}"; do
  python tools/autodan_output_converter.py --autodan_out_file_path "submodules/AutoDAN/results/autodan_hga/${model_id}_0_normal.json" --autodan_responses_file_path "submodules/AutoDAN/results/autodan_hga/${model_id}_0_normal_responses.json" --evotox_out_file_path "results/externalBaselines/${model_id}/autodan.json" --model_id "${model_id}"
done
