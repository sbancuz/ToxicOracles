# Using the best configuration for the evolutionary algorithm

#max_fs_glit
#max_fs
#max
#max_mem_5_fs_glit


#!/bin/bash

# Exit on error and log commands
#set -xe

dataset="dataset/reduced/questions_reduced100"
iterations=10
questions=-1
resultFolder="results/finalTests"
suts=(
  "llama3",
  "mistral",
  "vicuna",
  "vicunaUC"
)

pgs=(
  "llama3",
  "mistral",
  "vicuna",
  "vicunaUC"
)


for sut in "${suts[@]}"; do
  for pg in "${pgs[@]}"; do
    python evolutionary.py -i $iterations -f $dataset -o $resultFolder/$sut"_"$pg/max_mem_5_fs_glit.json -fs -g -m -w 5 -sut $sut -sg $pg -q $questions
    python evolutionary.py -i $iterations -f $dataset -o $resultFolder/$sut"_"$pg/max_fs_glit.json -fs -g -sut $sut -sg $pg -q $questions
    python evolutionary.py -i $iterations -f $dataset -o $resultFolder/$sut"_"$pg/max.json -sut $sut -sg $pg -q $questions
    python evolutionary.py -i $iterations -f $dataset -o $resultFolder/$sut"_"$pg/max_fs.json -fs -sut $sut -sg $pg -q $questions
    done
done



# # SUT: LLama 3, SG: LLama 3
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/llama3_llama3/max_mem_5_fs_glit.json -fs -g -m -w 5 -sut llama3 -sg llama3 -q -1
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/llama3_llama3/max_fs_glit.json -fs -g -sut llama3 -sg llama3 -q -1
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/llama3_llama3/max.json -sut llama3 -sg llama3 -q -1
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/llama3_llama3/max_fs.json -fs -sut llama3 -sg llama3 -q -1

# # SUT: LLama 3, SG: Vicuna
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/llama3_vicuna/max_mem_5_fs_glit.json -fs -g -m -w 5 -sut llama3 -sg vicuna -q -1
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/llama3_vicuna/max_fs_glit.json -fs -g -sut llama3 -sg vicuna -q -1
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/llama3_vicuna/max.json -sut llama3 -sg vicuna -q -1
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/llama3_vicuna/max_fs.json -fs -sut llama3 -sg vicuna -q -1

# # SUT: LLama3 SG: VicunaUC
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/llama3_vicunaUC/max_mem_5_fs_glit.json -fs -g -m -w 5 -sut llama3 -sg vicunaUC -q -1
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/llama3_vicunaUC/max_fs_glit.json -fs -g -sut llama3 -sg vicunaUC -q -1
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/llama3_vicunaUC/max.json -sut llama3 -sg vicunaUC -q -1
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/llama3_vicunaUC/max_fs.json -fs -sut llama3 -sg vicunaUC -q -1

# # SUT: Mistral, SG: Mistral
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/mistral_mistral/max_mem_5_fs_glit.json -fs -g -m -w 5 -sut mistral -sg mistral -q -1
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/mistral_mistral/max_fs_glit.json -fs -g -sut mistral -sg mistral -q -1
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/mistral_mistral/max.json -sut mistral -sg mistral -q -1
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/mistral_mistral/max_fs.json -fs -sut mistral -sg mistral -q -1

# # SUT: Mistral, SG: Vicuna
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/mistral_vicuna/max_mem_5_fs_glit.json -fs -g -m -w 5 -sut mistral -sg vicuna -q -1
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/mistral_vicuna/max_fs_glit.json -fs -g -sut mistral -sg vicuna -q -1
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/mistral_vicuna/max.json -sut mistral -sg vicuna -q -1
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/mistral_vicuna/max_fs.json -fs -sut mistral -sg vicuna -q -1

# # SUT: Mistral, SG: VicunaUC
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/mistral_vicunaUC/max_mem_5_fs_glit.json -fs -g -m -w 5 -sut mistral -sg vicunaUC -q -1
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/mistral_vicunaUC/max_fs_glit.json -fs -g -sut mistral -sg vicunaUC -q -1
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/mistral_vicunaUC/max.json -sut mistral -sg vicunaUC -q -1
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/mistral_vicunaUC/max_fs.json -fs -sut mistral -sg vicunaUC -q -1

# # SUT: Vicuna, SG: Mistral
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/vicuna_mistral/max_mem_5_fs_glit.json -fs -g -m -w 5 -sut vicuna -sg mistral -q -1
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/vicuna_mistral/max_fs_glit.json -fs -g -sut vicuna -sg mistral -q -1
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/vicuna_mistral/max.json -sut vicuna -sg mistral -q -1
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/vicuna_mistral/max_fs.json -fs -sut vicuna -sg mistral -q -1

# # SUT: Vicuna, SG: Vicuna
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/vicuna_vicuna/max_mem_5_fs_glit.json -fs -g -m -w 5 -sut vicuna -sg vicuna -q -1
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/vicuna_vicuna/max_fs_glit.json -fs -g -sut vicuna -sg vicuna -q -1
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/vicuna_vicuna/max.json -sut vicuna -sg vicuna -q -1
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/vicuna_vicuna/max_fs.json -fs -sut vicuna -sg vicuna -q -1

# # SUT: Vicuna, SG: VicunaUC
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/vicuna_vicunaUC/max_mem_5_fs_glit.json -fs -g -m -w 5 -sut vicuna -sg vicunaUC -q -1
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/vicuna_vicunaUC/max_fs_glit.json -fs -g -sut vicuna -sg vicunaUC -q -1
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/vicuna_vicunaUC/max.json -sut vicuna -sg vicunaUC -q -1
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/vicuna_vicunaUC/max_fs.json -fs -sut vicuna -sg vicunaUC -q -1

# # SUT: VicunaUC, SG: VicunaUC
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/vicunaUC_vicunaUC/max_mem_5_fs_glit.json -fs -g -m -w 5 -sut vicunaUC -sg vicunaUC -q -1
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/vicunaUC_vicunaUC/max_fs_glit.json -fs -g -sut vicunaUC -sg vicunaUC -q -1
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/vicunaUC_vicunaUC/max.json -sut vicunaUC -sg vicunaUC -q -1
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/vicunaUC_vicunaUC/max_fs.json -fs -sut vicunaUC -sg vicunaUC -q -1