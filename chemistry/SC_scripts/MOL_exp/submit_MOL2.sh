
export HF_HOME=/mnt/hdd1/cache
cd /mnt/ssd/ztl/LLMxFM
conda activate llmxfm

SEEDS=("2021" "7777" "8888" "1111" "2222" "3333" "5555" "101010" "202020" "606060")
SHOTS=(20)


TSAK_NAME="Lipophilicity_AstraZeneca"
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"



for SHOT in "${SHOTS[@]}"; do

    echo "Running for shot: $SHOT"
        for SEED in "${SEEDS[@]}"; do
        echo "Running for seed: $SEED"

        CUDA_VISIBLE_DEVICES=3 python -m chemistry.SC_scripts.MOL_run_trainer_pca_OT_icl --experiment_type rep --model_name $MODEL_NAME \
        --num_tests 1000 --n_representation_shots $SHOT --experiment_name pca_ot_abl --seed $SEED --task_name $TSAK_NAME

        CUDA_VISIBLE_DEVICES=3 python -m chemistry.SC_scripts.MOL_run_trainer_pca_OT_icl --experiment_type rep_w_o_norm --model_name $MODEL_NAME \
        --num_tests 1000 --n_representation_shots $SHOT --experiment_name pca_ot_abl --seed $SEED --task_name $TSAK_NAME

        CUDA_VISIBLE_DEVICES=3 python -m chemistry.SC_scripts.MOL_run_trainer_pca_OT_icl --experiment_type rep_icl --model_name $MODEL_NAME \
        --num_tests 1000 --n_representation_shots $SHOT --experiment_name pca_ot_abl --seed $SEED --task_name $TSAK_NAME


    done
 
done
