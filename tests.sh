# using the best configuration for the evolutionary algorithm

#max_fs_glit
#max_fs
#max
#max_mem_5_fs_glit

## Exit on error and log commands
set -xe


python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/vicuna_vicuna/max_mem_5_fs_glit.json -fs -g -m -w 5 -sut vicuna -sg vicuna -q -1 --api-key-perspective AIzaSyB6NK-Fu_099XTlvNn0Nx867Wd97SGYlTw
python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/vicuna_vicunaUC/max_mem_5_fs_glit.json -fs -g -m -w 5 -sut vicuna -sg vicunaUC -q -1 --api-key-perspective AIzaSyB6NK-Fu_099XTlvNn0Nx867Wd97SGYlTw

# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/mistral_vicuna/max.json -sut mistral -sg vicuna -q -1  
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/mistral_vicunaUC/max.json -sut mistral -sg vicunaUC -q -1  
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/vicuna_vicuna/max.json -sut vicuna -sg vicuna -q -1  
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/vicuna_vicunaUC/max.json -sut vicuna -sg vicunaUC -q -1  
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/vicunaUC_vicunaUC/max.json -sut vicunaUC -sg vicunaUC -q -1  
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/mistral_mistral/max.json -sut mistral -sg mistral -q -1  
# python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/vicuna_mistral/max.json -sut vicuna -sg mistral -q -1  


### DA FARE
# SUT 	   | SG       | Config | Status | Where 
# vicuna   | vicuna   | mem    | On     | Server Poli
# vicunaUC | vicunaUC | mem    | On     | Server Camilli
# vicuna   | vicunaUC | mem    | On     | Server Poli

