# set -ex

# DATASET_NAME='math'
# CUT_COEFF=0.05
# SMALL_COEFF=0.5
# BIG_COEFF=1.5
# BALANCE_COEFF=0.1
# NUM_SAMPLES=100
# R=1.0

# Read parameters from environment variables, use default values if not set
DATASET_NAME=${DATASET_NAME:-'math'}
CUT_COEFF=${CUT_COEFF:-0.05}
SMALL_COEFF=${SMALL_COEFF:-0.5}
BIG_COEFF=${BIG_COEFF:-1.5}
BALANCE_COEFF=${BALANCE_COEFF:-0.6}
R=${R:-0.1}
NUM_SAMPLES=${NUM_SAMPLES:-100}

# Print current experiment configuration
echo "=== Experiment Configuration ==="
echo "DATASET_NAME: $DATASET_NAME"
echo "CUT_COEFF: $CUT_COEFF"
echo "SMALL_COEFF: $SMALL_COEFF"
echo "BIG_COEFF: $BIG_COEFF"
echo "BALANCE_COEFF: $BALANCE_COEFF"
echo "R: $R"
echo "NUM_SAMPLES: $NUM_SAMPLES"
echo "==============================="

# ================================================================

MODEL_SERIES='deepseek'
MODEL_TYPE='base'
MODEL_NAME='/tmp/llm_file_save/deepseek/deepseek-math-7b-instruct'
python3 -u run_tests.py \
    --dataset_name ${DATASET_NAME} \
    --model_series ${MODEL_SERIES} \
    --model_name ${MODEL_NAME} \
    --model_type ${MODEL_TYPE} \
    --r ${R} \
    --cut_coeff ${CUT_COEFF} \
    --small_coeff ${SMALL_COEFF} \
    --big_coeff ${BIG_COEFF} \
    --balance_coeff ${BALANCE_COEFF} \
    --num_samples ${NUM_SAMPLES}

MODEL_SERIES='deepseek'
MODEL_TYPE='rl'
MODEL_NAME='/tmp/llm_file_save/deepseek/deepseek-math-7b-rl'
python3 -u run_tests.py \
    --dataset_name ${DATASET_NAME} \
    --model_series ${MODEL_SERIES} \
    --model_name ${MODEL_NAME} \
    --model_type ${MODEL_TYPE} \
    --r ${R} \
    --cut_coeff ${CUT_COEFF} \
    --small_coeff ${SMALL_COEFF} \
    --big_coeff ${BIG_COEFF} \
    --balance_coeff ${BALANCE_COEFF} \
    --num_samples ${NUM_SAMPLES}

# ================================================================

MODEL_SERIES='mistral'
MODEL_TYPE='base'
MODEL_NAME='/tmp/llm_file_save/mistral/mistral-7b-sft'
python3 -u run_tests.py \
    --dataset_name ${DATASET_NAME} \
    --model_series ${MODEL_SERIES} \
    --model_name ${MODEL_NAME} \
    --model_type ${MODEL_TYPE} \
    --r ${R} \
    --cut_coeff ${CUT_COEFF} \
    --small_coeff ${SMALL_COEFF} \
    --big_coeff ${BIG_COEFF} \
    --balance_coeff ${BALANCE_COEFF} \
    --num_samples ${NUM_SAMPLES}

MODEL_SERIES='mistral'
MODEL_TYPE='rl'
MODEL_NAME='/tmp/llm_file_save/mistral/math-shepherd-mistral-7b-rl'
python3 -u run_tests.py \
    --dataset_name ${DATASET_NAME} \
    --model_series ${MODEL_SERIES} \
    --model_name ${MODEL_NAME} \
    --model_type ${MODEL_TYPE} \
    --r ${R} \
    --cut_coeff ${CUT_COEFF} \
    --small_coeff ${SMALL_COEFF} \
    --big_coeff ${BIG_COEFF} \
    --balance_coeff ${BALANCE_COEFF} \
    --num_samples ${NUM_SAMPLES}


# ================================================================

MODEL_SERIES='qwen'
MODEL_TYPE='base'
MODEL_NAME='/tmp/llm_file_save/qwen/Qwen2.5-7B-SFT'
python3 -u run_tests.py \
    --dataset_name ${DATASET_NAME} \
    --model_series ${MODEL_SERIES} \
    --model_name ${MODEL_NAME} \
    --model_type ${MODEL_TYPE} \
    --r ${R} \
    --cut_coeff ${CUT_COEFF} \
    --small_coeff ${SMALL_COEFF} \
    --big_coeff ${BIG_COEFF} \
    --balance_coeff ${BALANCE_COEFF} \
    --num_samples ${NUM_SAMPLES}

MODEL_SERIES='qwen'
MODEL_TYPE='rl'
MODEL_NAME='/tmp/llm_file_save/qwen/Qwen2.5-7B-DPO'
python3 -u run_tests.py \
    --dataset_name ${DATASET_NAME} \
    --model_series ${MODEL_SERIES} \
    --model_name ${MODEL_NAME} \
    --model_type ${MODEL_TYPE} \
    --r ${R} \
    --cut_coeff ${CUT_COEFF} \
    --small_coeff ${SMALL_COEFF} \
    --big_coeff ${BIG_COEFF} \
    --balance_coeff ${BALANCE_COEFF} \
    --num_samples ${NUM_SAMPLES}

# ================================================================

# # ////////////////////////
# CUT_COEFF=0.04
# # ////////////////////////

MODEL_SERIES='nvidia-qwen'
MODEL_TYPE='base'
MODEL_NAME='/tmp/llm_file_save/nvidia-qwen/DeepSeek-R1-Distill-Qwen-7B'
python3 -u run_tests.py \
    --dataset_name ${DATASET_NAME} \
    --model_series ${MODEL_SERIES} \
    --model_name ${MODEL_NAME} \
    --model_type ${MODEL_TYPE} \
    --r ${R} \
    --cut_coeff ${CUT_COEFF} \
    --small_coeff ${SMALL_COEFF} \
    --big_coeff ${BIG_COEFF} \
    --balance_coeff ${BALANCE_COEFF} \
    --num_samples ${NUM_SAMPLES}

MODEL_SERIES='nvidia-qwen'
MODEL_TYPE='rl'
MODEL_NAME='/tmp/llm_file_save/nvidia-qwen/AceReason-Nemotron-7B'
python3 -u run_tests.py \
    --dataset_name ${DATASET_NAME} \
    --model_series ${MODEL_SERIES} \
    --model_name ${MODEL_NAME} \
    --model_type ${MODEL_TYPE} \
    --r ${R} \
    --cut_coeff ${CUT_COEFF} \
    --small_coeff ${SMALL_COEFF} \
    --big_coeff ${BIG_COEFF} \
    --balance_coeff ${BALANCE_COEFF} \
    --num_samples ${NUM_SAMPLES}

# ================================================================