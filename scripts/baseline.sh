# Exit on error and log commands
set -xe

# SUT: LLAMA 3
python evolutionary.py -i 0 -f dataset/reduced/questions_reduced100 -o results/finalTests/baselineJailbreak/llama3.json -sut llama3 -sg llama3 -q -1 -j --api-key-perspective [INSERT API KEY]

# SUT: Vicuna
python evolutionary.py -i 0 -f dataset/reduced/questions_reduced100 -o results/finalTests/baselineJailbreak/vicuna.json -sut vicuna -sg vicuna -q -1 -j --api-key-perspective [INSERT API KEY]

# SUT: Mistral
python evolutionary.py -i 0 -f dataset/reduced/questions_reduced100 -o results/finalTests/baselineJailbreak/mistral.json -sut mistral -sg mistral -q -1 -j --api-key-perspective [INSERT API KEY]

# SUT: VicunaUC
python evolutionary.py -i 0 -f dataset/reduced/questions_reduced100 -o results/finalTests/baselineJailbreak/vicunaUC.json -sut vicunaUC -sg vicunaUC -q -1 -j --api-key-perspective [INSERT API KEY]