
#!/bin/bash

export HF_HOME=/mnt/hdd1/cache
cd /mnt/ssd/ztl/LLMxFM


SEEDS=("2021" "7777" "8888" "1111" "2222" "3333" "5555" "101010" "202020" "606060")


TASK_NAME="Stability"
MODEL_NAME="/mnt/hdd1/cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
PROJ="/mnt/ssd/ztl/LLMxFM/chemistry/logs/trained_projectors/Fluorescence_Stability/LLM_align_random_layer_last_layer_cls_Nonlinear_lr_0.001_wd_0.01_batch_64_epochs200_Fluorescence_Stability/20250607-211316/Fluorescence_Stability_best_projector.pth"
ESM_CACHE_PATH="/mnt/ssd/ztl/LLMxFM/chemistry/datasets/aaseq_to_rep_store_cls.pkl"
TYPE="MLP_Nonlinear"
# MLP_Linear MLP_Nonlinear
ARCH="8192-8192"

# "Stability", "Fluorescence"

for SEED in "${SEEDS[@]}"; do
    echo "Running for seed: $SEED"

    CUDA_VISIBLE_DEVICES=1 python -m chemistry.SC_scripts.ESM_run_trainer_contrastive_icl \
    --experiment_type rep_icl --experiment_name contrastive_nolinear_8192_esm_last_layer --seed $SEED --task_name $TASK_NAME --num_tests 1000 \
    --model_name $MODEL_NAME --esm_cache_path $ESM_CACHE_PATH \
    --all_arch $ARCH \
    --proj_path $PROJ \
    --projector_type $TYPE \


    CUDA_VISIBLE_DEVICES=1 python -m chemistry.SC_scripts.ESM_run_trainer_contrastive_icl \
    --experiment_type rep --experiment_name contrastive_nolinear_8192_esm_last_layer --seed $SEED --task_name $TASK_NAME --num_tests 1000 \
    --model_name $MODEL_NAME --esm_cache_path $ESM_CACHE_PATH \
    --all_arch $ARCH \
    --proj_path $PROJ \
    --projector_type $TYPE \

done
