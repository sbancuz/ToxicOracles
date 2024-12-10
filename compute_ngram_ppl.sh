#!/bin/bash
set -xe

# Declare n-grams orders array
orders=(
  5
)
# Declare corpora for n-grams array
corpora=(
  "wikitext2"
  "book_corpus"
)
# Declare directories array (list of lists)
directories=(
  "results/finalTests/mistral_mistral results/finalTests/mistral_vicuna results/finalTests/mistral_vicunaUC"
  "results/finalTests/llama3_llama3 results/finalTests/llama3_vicuna results/finalTests/llama3_vicunaUC"
  "results/finalTests/vicuna_mistral results/finalTests/vicuna_vicuna results/finalTests/vicuna_vicunaUC"
)
# Declare files array
files=(
  "baseline.json"
  "max.json"
  "max_fs.json"
  "max_fs_glit.json"
  "max_mem_5_fs_glit.json"
)

# Loop over the n-gram orders
for order in "${orders[@]}"; do
  # Loop over training corpora
  for corpus in "${corpora[@]}"; do
    # Loop over lists of directories
    for model_dirs_list in "${directories[@]}"; do
      # Get sub-list of directories
      IFS=' ' read -r -a model_dirs <<< "${directories[$i]}"
      # Loop over the directories
      for dir in "${model_dirs[@]}"; do
        # Loop over the files
        for file in "${files[@]}"; do
          # Run PPL computation
          python ./ngram_perplexity.py --order "${order}" --corpus "${corpus}" --data_path "${dir}/${file}"
        done
      done
    done
  done
done
