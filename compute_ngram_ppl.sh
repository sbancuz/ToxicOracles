#!/bin/bash
set -xe

# Declare n-grams orders array
orders=(
  3
  4
  5
)
# Declare corpora for n-grams array
corpora=(
  "wikitext2"
  "book_corpus_sentences"
  "book_corpus"
)
# Declare directories array (list of lists)
directories=(
  "results/finalTests/mistral_mistral results/finalTests/mistral_vicuna results/finalTests/mistral_vicunaUC"
  "results/finalTests/llama3_llama3 results/finalTests/llama3_vicuna results/finalTests/llama3_vicunaUC"
  "results/finalTests/vicuna_mistral results/finalTests/vicuna_vicuna results/finalTests/vicuna_vicunaUC"
  "results/finalTests/baselineJailbreak/JailBreakPrompts-Mistral.json results/finalTests/baselineJailbreak/JailbreakPrompts-llama3.json results/finalTests/baselineJailbreak/JailbreakPrompts-vicuna.json results/finalTests/baselineJailbreak/JailbreakPrompts-vicunaUC.json"
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
      IFS=' ' read -r -a model_dirs <<< "${model_dirs_list[$i]}"
      # Loop over the directories
      for dir in "${model_dirs[@]}"; do
        # Loop over the files
        for file in "${files[@]}"; do
          # Run PPL computation
          python ./ngram_perplexity.py --model "ngrams/${order}-gram.${corpus}.arpa" --data_path "${dir}/${file}"
        done
      done
    done
  done
done
