# Exit on error and log commands
set -xe

# SUT: LLAMA 3
#python evolutionary.py -i 0 -f dataset/reduced/questions_reduced100 -o results/finalTests/baselineJailbreak/llama3.json -sut llama3 -sg llama3 -q -1 -j --api-key-perspective AIzaSyA0k8T0fua-peGuuWawe0tpSV_ztrQgdHY

# SUT: Vicuna
#python evolutionary.py -i 0 -f dataset/reduced/questions_reduced100 -o results/finalTests/baselineJailbreak/vicuna.json -sut vicuna -sg vicuna -q -1 -j --api-key-perspective AIzaSyA0k8T0fua-peGuuWawe0tpSV_ztrQgdHY

# SUT: Mistral
#python evolutionary.py -i 0 -f dataset/reduced/questions_reduced100 -o results/finalTests/baselineJailbreak/mistral.json -sut mistral -sg mistral -q -1 -j --api-key-perspective AIzaSyA0k8T0fua-peGuuWawe0tpSV_ztrQgdHY

# SUT: VicunaUC
python evolutionary.py -i 0 -f dataset/reduced/questions_reduced100 -o results/finalTests/baselineJailbreak/vicunaUC.json -sut vicunaUC -sg vicunaUC -q -1 -j --api-key-perspective AIzaSyA0k8T0fua-peGuuWawe0tpSV_ztrQgdHY