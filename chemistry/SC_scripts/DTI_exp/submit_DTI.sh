
# conda activate llmxfm
export HF_HOME=/home/chanalvin/cache
cd /home/chanalvin/LLMxFM
conda activate llmxfm


MAX_RUNTIME=2000

run_script_with_timeout() {
    local CMD=$1
    local RETRY_LIMIT=3
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


SEEDS=("2021" "7777" "8888" "1111" "2222" "3333" "5555" "101010" "202020" "606060")
MODES=("max_affinity")
TASKS_NAMES=("BindingDB_IC50")
INSTRUCATION_PROMPT_NAMES=("DTI_prompt_mol_pro_rep_icl_KIBA")

for MODE in "${MODES[@]}"; do

    for TASKS_NAME in "${TASKS_NAMES[@]}"; do
        echo "Running for task: $TASKS_NAME"

        for INSTRUCATION_PROMPT_NAME in "${INSTRUCATION_PROMPT_NAMES[@]}"; do

            for SEED in "${SEEDS[@]}"; do
                echo "Running for seed: $SEED"

                run_script_with_timeout "python -m chemistry.SC_scripts.DTI_run_trainer_pca_OT_icl --experiment_type rep_icl --experiment_name pca_ot --seeds $SEED --mode $MODE --task_name $TASKS_NAME --task_name $TASKS_NAME --instruction_prompt_name $INSTRUCATION_PROMPT_NAME"
                run_script_with_timeout "python -m chemistry.SC_scripts.DTI_run_trainer_pca_OT_icl --experiment_type rep --experiment_name pca_ot --seeds $SEED --mode $MODE --task_name $TASKS_NAME --instruction_prompt_name $INSTRUCATION_PROMPT_NAME"

                run_script_with_timeout "python -m chemistry.SC_scripts.DTI_run_trainer_embed_OT_icl --experiment_type rep_icl --experiment_name embed_ot --seeds $SEED --mode $MODE --task_name $TASKS_NAME --instruction_prompt_name $INSTRUCATION_PROMPT_NAME"
                run_script_with_timeout "python -m chemistry.SC_scripts.DTI_run_trainer_embed_OT_icl --experiment_type rep --experiment_name embed_ot --seeds $SEED --mode $MODE --task_name $TASKS_NAME --instruction_prompt_name $INSTRUCATION_PROMPT_NAME"

                run_script_with_timeout "python -m chemistry.SC_scripts.DTI_run_trainer_icrl --experiment_type rep_icl_w_o_norm --experiment_name icrl --seeds $SEED --mode $MODE --task_name $TASKS_NAME --instruction_prompt_name $INSTRUCATION_PROMPT_NAME"
                run_script_with_timeout "python -m chemistry.SC_scripts.DTI_run_trainer_icrl --experiment_type rep_w_o_norm --experiment_name icrl --seeds $SEED --mode $MODE --task_name $TASKS_NAME --instruction_prompt_name $INSTRUCATION_PROMPT_NAME"

                run_script_with_timeout "python -m chemistry.SC_scripts.DTI_run_trainer_hash --experiment_type rep_icl --experiment_name hash --seeds $SEED --mode $MODE --task_name $TASKS_NAME --instruction_prompt_name $INSTRUCATION_PROMPT_NAME"
                run_script_with_timeout "python -m chemistry.SC_scripts.DTI_run_trainer_hash --experiment_type rep --experiment_name hash --seeds $SEED --mode $MODE --task_name $TASKS_NAME --instruction_prompt_name $INSTRUCATION_PROMPT_NAME"


                run_script_with_timeout "python -m chemistry.SC_scripts.DTI_run_pca_icl_baseline --experiment_type icl --experiment_name baseline --seeds $SEED --mode $MODE --task_name $TASKS_NAME --instruction_prompt_name $INSTRUCATION_PROMPT_NAME"
                run_script_with_timeout "python -m chemistry.SC_scripts.DTI_run_pca_icl_baseline --experiment_type pca_icl --experiment_name baseline --seeds $SEED --mode $MODE --task_name $TASKS_NAME --instruction_prompt_name $INSTRUCATION_PROMPT_NAME"
                run_script_with_timeout "python -m chemistry.SC_scripts.DTI_run_pca_icl_baseline --experiment_type pca --experiment_name baseline --seeds $SEED --mode $MODE --task_name $TASKS_NAME --instruction_prompt_name $INSTRUCATION_PROMPT_NAME"

            done
        done
    done
done





