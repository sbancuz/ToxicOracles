#!/bin/bash
set -xe

# HF token
hf_token="hf_"
# Declare directories array (list of lists)
directories=(
  "results/finalTests/mistral_mistral results/finalTests/mistral_vicuna results/finalTests/mistral_vicunaUC"
  "results/finalTests/llama3_llama3 results/finalTests/llama3_vicuna results/finalTests/llama3_vicunaUC"
  "results/finalTests/vicuna_mistral results/finalTests/vicuna_vicuna results/finalTests/vicuna_vicunaUC"
  "results/finalTests/vicunaUC_vicunaUC"
)
# Declare files array
files=(
  "baseline.json"
  "max.json"
  "max_fs.json"
  "max_fs_glit.json"
  "max_mem_5_fs_glit.json"
)
# Declare list of file paths
jailbreak_files=(
  "results/finalTests/baselineJailbreak/JailBreakPrompts-Mistral.json"
  "results/finalTests/baselineJailbreak/JailbreakPrompts-llama3.json"
  "results/finalTests/baselineJailbreak/JailbreakPrompts-vicuna.json"
  "results/finalTests/baselineJailbreak/JailbreakPrompts-vicunaUC.json"
)

# Loop over the models
for model in "${models[@]}" do
  # Loop over lists of directories
  for model_dirs_list in "${directories[@]}"; do
    # Get sub-list of directories
    IFS=' ' read -r -a model_dirs <<< "${model_dirs_list}"
    # Loop over the directories
    for dir in "${model_dirs[@]}"; do
      # Loop over the files
      for file in "${files[@]}"; do
        # Run PPL computation
        python ./perplexity.py --model "${model}" --token "${hf_token}" --data_path "${dir}/${file}" --batch_size 16
      done
    done
  done
  # Loop over the files
  for file_path in "${jailbreak_files[@]}"; do
    # Run PPL computation
    python ./perplexity.py --model "${model}" --token "${hf_token}" --data_path "${file_path}" --batch_size 16
  done
done
