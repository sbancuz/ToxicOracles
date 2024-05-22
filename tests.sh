# using the best configuration for the evolutionary algorithm

#max_fs_glit
#max_fs
#max
#max_mem_5_fs_glit

# with mistral as sut and generator
python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/mistral_mistral/max_fs_glit.json -fs -g -sut mistral -sg mistral
python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/mistral_mistral/max_fs.json -fs -sut mistral -sg mistral
python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/mistral_mistral/max.json -sut mistral -sg mistral
python evolutionary.py -i 10 -f dataset/reduced/questions_reduced100 -o results/finalTests/mistral_mistral/max_mem_5_fs_glit.json -fs -g -m -w 5 -sut mistral -sg mistral