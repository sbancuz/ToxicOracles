#!/bin/bash
# set -xe

# Declare files array
input_files=(
  "submodules/AutoDAN/results/autodan_ga/llama_0_llama.json"
  "submodules/AutoDAN/results/autodan_ga/mistral_0_mistral.json"
  "submodules/AutoDAN/results/autodan_ga/vicuna_0_vicuna.json",
  "submodules/AutoDAN/results/autodan_ga/vicunaUC_0_vicunaUC.json"
)

output_files=(
  "results/externalBaselines/llama3/autodan.json"
  "results/externalBaselines/mistral/autodan.json"
  "results/externalBaselines/vicuna/autodan.json"
  "results/externalBaselines/vicunaUC/autodan.json"
)

model_ids=(
  "llama3"
  "mistral"
  "vicuna"
  "vicunaUC"
)

# Loop over the input-output file pairs
for i in "${!input_files[@]}"; do
  input_file="${input_files[${i}]}"
  output_file="${output_files[${i}]}"
  model_id="${model_ids[${i}]}"

  python tools/autodan_output_converter.py --input_path "${input_file}" --output_path "${output_file}" --model_id "${model_id}"
done
