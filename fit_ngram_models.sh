#!/bin/bash
set -xe

# Declare corpora for n-grams array
corpora=(
  "wikitext2"
  "book_corpus_sentences"
  "book_corpus"
)
# Declare n-grams orders array
orders=(
  3
  4
  5
)

# Create output directory
mkdir -p "ngrams"
# Loop over training corpora
for corpus in "${corpora[@]}"; do
  # Loop over the n-gram orders
  for order in "${orders[@]}"; do
    # Create a temporary file for this iteration
    tmpfile=$(mktemp)
    # Ensure cleanup for this specific file in case of errors
    # trap 'rm -f "${tmpfile}"' RETURN
    # Generate data set in correct format
    python ./tools/ngram_data_generator.py --corpus "${corpus}" > ${tmpfile}
    # Fit language model
    ./kenlm/build/bin/lmplz -o ${order} < ${tmpfile} > "ngrams/${order}-gram.${corpus}.arpa"
    # Clean up the temporary file immediately (optional since we are using using trap)
    rm -f "${tmpfile}"
    # Remove RETURN trap if not needed for subsequent commands
    # trap - RETURN
  done
done
