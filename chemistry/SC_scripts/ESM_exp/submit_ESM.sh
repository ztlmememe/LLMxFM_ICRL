

export HF_HOME=/home/chanalvin/cache
cd /home/chanalvin/LLMxFM
conda activate llmxfm

MAX_RUNTIME=47200

run_script_with_timeout() {
    local CMD=$1
    local RETRY_LIMIT=1 
    local ATTEMPT=0

    while [ $ATTEMPT -lt $RETRY_LIMIT ]; do
        echo "Attempt $(($ATTEMPT + 1)) for command: $CMD"
        timeout $MAX_RUNTIME bash -c "$CMD"
        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
            echo "Command succeeded: $CMD"
            return 0
        elif [ $EXIT_CODE -eq 124 ]; then
            echo "Command timed out. Retrying..."
        else
            echo "Command failed with exit code $EXIT_CODE. Retrying..."
        fi

        ATTEMPT=$(($ATTEMPT + 1))
    done

    echo "Command failed after $RETRY_LIMIT attempts: $CMD"
    return 1
}

# SEEDS=("18451246")
# SEEDS=("6666" "8888" "1111" "2222" "3333" "4444" "5555" "10086" "11037" "1806" "8520" "512" "51423" "3000" "10001" "9999" "888" "999" "101010" "202020" "303030" "404040" "505050" "606060")

# SEEDS POOL
SEEDS=("7777" "8888" "101010" "5555" "606060")
NTEST=1000

TASK_NAME="Stability"

for SEED in "${SEEDS[@]}"; do
    echo "Running for seed: $SEED"


    run_script_with_timeout "python -m chemistry.SC_scripts.ESM_run_pca_icl_baseline --experiment_type icl --experiment_name baseline --seed $SEED --task_name $TASK_NAME --num_tests $NTEST"
    
    run_script_with_timeout "python -m chemistry.SC_scripts.ESM_run_trainer_icrl_v2 --experiment_type rep_icl_w_o_norm --experiment_name icrl_v2 --seed $SEED --task_name $TASK_NAME --num_tests $NTEST"

    run_script_with_timeout "python -m chemistry.SC_scripts.ESM_run_trainer_icrl_v2 --experiment_type rep_w_o_norm --experiment_name icrl_v2 --seed $SEED --task_name $TASK_NAME --num_tests $NTEST"


    run_script_with_timeout "python -m chemistry.SC_scripts.ESM_run_trainer_icrl_v2 --experiment_type rep_w_o_norm --experiment_name icrl_v2 \
                            --n_representation_shots 50 --seed $SEED --task_name $TASK_NAME --num_tests $NTEST"

done

