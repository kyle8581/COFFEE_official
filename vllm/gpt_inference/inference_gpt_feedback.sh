#!/bin/bash
function start_server(){
    echo "Starting server..."
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES_SINGLE python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH_SINGLE \
    --tensor-parallel-size $TENSOR_SIZE_SINGLE \
    --seed 42 \
    --port $PORT_SINGLE > ${SERVER_LOG_FILE} 2>&1 &


    # sh ${RUN_SERVER} ${CUDA_DEVICES_SINGLE} ${MODEL_PATH_SINGLE} ${TENSOR_SIZE_SINGLE} ${PORT_SINGLE} > ${SERVER_LOG_FILE} 2>&1 &
    SERVER_PID=$!
    echo "Server PID: ${SERVER_PID}"

    echo "Waiting for server to start..."
    while ! grep "Uvicorn running on" ${SERVER_LOG_FILE}; do
        sleep 10
    done
}

function start_run_script_single() {
    echo "Starting the run script..."
    python vllm/gpt_inference/inference_gpt.py \
    --model_name ${MODEL_PATH_SINGLE} \
    --port ${PORT_SINGLE} \
    --gpt_feedback_result ${GPT_FEEDBACK_RESULT} \
    --prompt_key ${PROMPT} \
    --split test \
    --save_dir ${SAVE_DIR} 
    # \
    # > "${RUN_SCRIPT_LOG_FILE}" &

    RUN_SCRIPT_PID=$!
    # echo "Run script PID: ${RUN_SCRIPT_PID}"

    # echo "Waiting for run script to complete..."
    # while ! grep "Done!" ${RUN_SCRIPT_LOG_FILE}; do
    #     sleep 10
    # done
}





function handle_sigint() {
    echo -e "\nYou've stopped the main script. The run script process will be terminated."
    kill ${RUN_SCRIPT_PID} || true
    
    kill ${SERVER_PID}
    echo "Server process terminated."


    exit 0
}


# ----------------- Main Execution -----------------
trap handle_sigint SIGINT

MODEL_LAST_NAME=$(basename $2)
SAVE_DIR="vllm/results/${MODEL_LAST_NAME}-gpt-feedback"
LOG_DIR="vllm/logs"
SERVER_LOG_FILE="${LOG_DIR}/${MODEL_LAST_NAME}-gptfeedback_server.log"
touch ${SERVER_LOG_FILE}
# RUN_SCRIPT_LOG_FILE="${LOG_DIR}/${MODEL_LAST_NAME}_run_script.log"
if [ "$#" -ne 7 ]; then
    echo "Error: Incorrect number of arguments for single-server version."
    exit 1
fi

CUDA_DEVICES_SINGLE=$1
MODEL_PATH_SINGLE=$2
PORT_SINGLE=$3
TENSOR_SIZE_SINGLE=$4
PROMPT=$5
ENV_NAME=$6
GPT_FEEDBACK_RESULT=$7
start_server
start_run_script_single

echo "Saved in: ${SAVE_DIR}"

kill ${SERVER_PID}
