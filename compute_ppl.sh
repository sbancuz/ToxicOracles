#!/bin/bash
set -xe

# HF token
hf_token="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"
# Declare models array
models=(
  "EleutherAI/pythia-12b"
  "EleutherAI/pythia-12b"
  "EleutherAI/pythia-12b"
)
# Declare directories array (list of lists)
directories=(
  "results/finalTests/mistral_mistral results/finalTests/mistral_vicuna results/finalTests/mistral_vicunaUC"
  "results/finalTests/llama3_llama3 results/finalTests/llama3_vicuna results/finalTests/llama3_vicunaUC"
  "results/finalTests/vicuna_mistral results/finalTests/vicuna_vicuna results/finalTests/vicuna_vicunaUC"
)
# Declare files array
files=(
  #"baseline.json"
  "max.json"
  "max_fs.json"
  "max_fs_glit.json"
  "max_mem_5_fs_glit.json"
)

# Loop over the models
for i in "${!models[@]}"; do
  model="${models[$i]}"
  # Get corresponding directories for the current model
  IFS=' ' read -r -a model_dirs <<< "${directories[$i]}"
  # Loop over the directories for the current model
  for dir in "${model_dirs[@]}"; do
    # Loop over the files
    for file in "${files[@]}"; do
      # Run PPL computation
      python ./perplexity.py --model "${model}" --token "${hf_token}" --data_path "${dir}/${file}" --batch_size 16
    done
  done
done
