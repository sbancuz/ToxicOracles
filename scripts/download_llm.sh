# Prepare directories
mkdir -p ${PWD}/resources
mkdir -p ${PWD}/resources/models
mkdir -p ${PWD}/resources/models/pre_trained
# Download LLM models
# wget https://huggingface.co/TheBloke/Nous-Capybara-34B-GGUF/resolve/main/nous-capybara-34b.Q4_K_M.gguf -P ${PWD}/resources/models/pre_trained > ${PWD}/resources/models/pre_trained/capybara_34b_q4_download.out &
wget https://huggingface.co/TheBloke/vicuna-13B-v1.5-16K-GGUF/resolve/main/vicuna-13b-v1.5-16k.Q5_K_M.gguf -P ${PWD}/resources/models/pre_trained > ${PWD}/resources/models/pre_trained/vicuna_13b_16k_q5_download.out &
wget https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/resolve/main/mistral-7b-openorca.Q5_K_M.gguf -P ${PWD}/resources/models/pre_trained > ${PWD}/resources/models/pre_trained/mistral_7b_q5_download.out &
wget https://huggingface.co/JohanAR/Wizard-Vicuna-13B-Uncensored-SuperHOT-8K-GGUF/resolve/main/wizard-vicuna-13b-uncensored-superhot-8k.q5_K_M.gguf -P ${PWD}/resources/models/pre_trained > ${PWD}/resources/models/pre_trained/vicuna_uncensored_13b_8k_q5_download.out &
wget https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf -P ${PWD}/resources/models/pre_trained > ${PWD}/resources/models/pre_trained/llama3_8b_q5_download.out &