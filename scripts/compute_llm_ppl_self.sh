#!/bin/bash
set -xe

# HF token
hf_token="hf_"
# Declare models array
models=(
  "Open-Orca/Mistral-7B-OpenOrca"
  "meta-llama/Meta-Llama-3-8B-Instruct"
  "lmsys/vicuna-13b-v1.5-16k"
  "cognitivecomputations/Wizard-Vicuna-13B-Uncensored"
)
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
for i in "${!models[@]}"; do
  model="${models[${i}]}"
  # Get corresponding directories for the current model
  IFS=' ' read -r -a model_dirs <<< "${directories[${i}]}"
  # Loop over the directories for the current model
  for dir in "${model_dirs[@]}"; do
    # Loop over the files
    for file in "${files[@]}"; do
      # Run PPL computation
      python ./perplexity.py --model "${model}" --token "${hf_token}" --data_path "${dir}/${file}" --batch_size 16
    done
  done
  # Select file
  file_path="${jailbreak_files[${i}]}"
  # Run PPL computation
  python ./perplexity.py --model "${model}" --token "${hf_token}" --data_path "${file_path}" --batch_size 16
done
