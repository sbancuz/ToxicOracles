## Exit on error and log commands
set -xe


#python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/vicunaUC_vicunaUC/max_mem_5_fs_glit.json -fs -g -m -w 5 -sut vicunaUC -sg vicunaUC -q -1 --api-key-perspective AIzaSyC5f5JC_62NstfecDtKHMTVNhcEWg5SsVc
#python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/mistral_vicunaUC/max_mem_5_fs_glit.json -fs -g -m -w 5 -sut mistral -sg vicunaUC -q -1 --api-key-perspective AIzaSyC5f5JC_62NstfecDtKHMTVNhcEWg5SsVc

#python evolutionary.py -i 0 -f dataset/MaliciousInstruct -o results/rebuttal/vicuna/maliciousInstruct.json -sut vicuna -sg vicuna -q -1
#python evolutionary.py -i 0 -f dataset/advbench -o results/rebuttal/vicuna/advbench.json -sut vicuna -sg vicuna -q -1

#python evolutionary.py -i 0 -f dataset/MaliciousInstruct -o results/rebuttal/mistral/maliciousInstruct.json -sut mistral -sg vicuna -q -1
#python evolutionary.py -i 0 -f dataset/advbench -o results/rebuttal/mistral/advbench.json -sut mistral -sg vicuna -q -1

python evolutionary.py -i 0 -f dataset/MaliciousInstruct -o results/rebuttal/llama3/maliciousInstruct.json -sut llama3 -sg vicuna -q -1 --api-key-perspective AIzaSyA0k8T0fua-peGuuWawe0tpSV_ztrQgdHY
python evolutionary.py -i 0 -f dataset/advbench -o results/rebuttal/llama3/advbench.json -sut llama3 -sg vicuna -q -1 --api-key-perspective AIzaSyA0k8T0fua-peGuuWawe0tpSV_ztrQgdHY