#! /bin/bash
# Function to display script usage
show_help() {
    echo "Usage: ${0} [-m|--model {capybara|mistral|vicuna|llama|gemma}] [-p|--port] [-h|--help]"
    echo "Options:"
    echo "  -m, --model: LLM name"
    echo "  -p, --port : Service running port for forwarding (defaults to 8000)"
    echo "  -h, --help : Show this help message"
    exit 1
}
# Script options
## Parse command-line options
while getopts ":m:p:h" opt; do
    case ${opt} in
        m|--model)
            case ${OPTARG} in
              capybara|mistral|vicuna|vicuna_uncensored|llama|gemma)
                  llm="${OPTARG}"
                  ;;
              *)
                  echo "Invalid LLM name: ${OPTARG}"
                  show_help
                  ;;
            esac
            ;;
        p|--port)
            port="${OPTARG:-8000}"
            ;;
        h|--help)
            show_help
            ;;
        \?)
            echo "Invalid option: -${OPTARG}"
            show_help
            ;;
        :)
            echo "Option -${OPTARG} requires an argument."
            show_help
            ;;
    esac
done
## Check if required options are provided
if [ -z "${llm}" ]; then
    echo "Error: LLM name is required is required."
    show_help
fi
# Add port forwarding rule
sudo ufw allow ${port}
# Generate access key
current_datetime=$(date "+%Y_%m_%d_%H_%M_%S")
access_token=$(echo -n "${current_datetime}" | sha256sum | awk '{print $1}')
echo "API token: ${access_token}" >> ${HOME}/llm_server_${current_datetime}_${llm}
# Run the LLM server using docker
# docker rm llm_server
case ${llm} in
    capybara)
        docker \
          run \
          -v ${PWD}/resources/models/pre_trained:/app/llm/resources/models/pre_trained \
          --gpus all \
          -p ${port}:8000 \
          --name llm_server_capybara \
          -dit llm_server \
          python3 -m llama_cpp.server --api_key ${access_token} --model ./resources/models/pre_trained/nous-capybara-34b.Q4_K_M.gguf --n_gpu_layers -1 --chat_format vicuna --n_ctx 8192
        ;;
    mistral)
        docker \
          run \
          -v ${PWD}/resources/models/pre_trained:/app/llm/resources/models/pre_trained \
          --gpus all \
          -p ${port}:8000 \
          --name llm_server_mistral \
          -dit llm_server \
          python3 -m llama_cpp.server --api_key ${access_token} --model ./resources/models/pre_trained/mistral-7b-openorca.Q5_K_M.gguf --n_gpu_layers -1 --chat_format chatml --n_ctx 8192
        ;;
    vicuna)
        docker \
          run \
          -v ${PWD}/resources/models/pre_trained:/app/llm/resources/models/pre_trained \
          --gpus all \
          -p ${port}:8000 \
          --name llm_server_vicuna \
          -dit llm_server \
          python3 -m llama_cpp.server --api_key ${access_token} --model ./resources/models/pre_trained/vicuna-13b-v1.5-16k.Q5_K_M.gguf --n_gpu_layers -1 --chat_format vicuna --n_ctx 8192 --rope_freq_scale 0.25
        ;;
    vicuna_uncensored)
        docker \
          run \
          -v ${PWD}/resources/models/pre_trained:/app/llm/resources/models/pre_trained \
          --gpus all \
          -p ${port}:8000 \
          --name llm_server_vicuna_uncensored \
          -dit llm_server \
          python3 -m llama_cpp.server --api_key ${access_token} --model ./resources/models/pre_trained/wizard-vicuna-13b-uncensored-superhot-8k.q5_K_M.gguf --n_gpu_layers 16 --chat_format vicuna --n_ctx 8192 --rope_freq_scale 0.5
        ;;
    llama)
        docker \
          run \
          -v ${PWD}/resources/models/pre_trained:/app/llm/resources/models/pre_trained \
          --gpus all \
          -p ${port}:8000 \
          --name llm_server_llama \
          -dit llm_server \
          python3 -m llama_cpp.server --api_key ${access_token} --model ./resources/models/pre_trained/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf --n_gpu_layers -1 --chat_format llama-3 --n_ctx 8192
        ;;
    gemma)
        docker \
          run \
          -v ${PWD}/resources/models/pre_trained:/app/llm/resources/models/pre_trained \
          --gpus all \
          -p ${port}:8000 \
          --name llm_server_gemma \
          -dit llm_server \
          python3 -m llama_cpp.server --api_key ${access_token} --model ./resources/models/pre_trained/gemma-7b-it.Q5_K_M.gguf --n_gpu_layers -1 --chat_format gemma --n_ctx 8192
        ;;
    *)
        echo "Invalid LLM name: ${OPTARG}"
        show_help
        ;;
esac
