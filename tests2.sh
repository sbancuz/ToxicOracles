## Exit on error and log commands
set -xe


#python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/vicunaUC_vicunaUC/max_mem_5_fs_glit.json -fs -g -m -w 5 -sut vicunaUC -sg vicunaUC -q -1 --api-key-perspective AIzaSyC5f5JC_62NstfecDtKHMTVNhcEWg5SsVc
python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/llama3_vicunaUC/max_mem_5_fs_glit.json -fs -g -m -w 5 -sut llama3 -sg vicunaUC -q -1 --api-key-perspective AIzaSyC5f5JC_62NstfecDtKHMTVNhcEWg5SsVc