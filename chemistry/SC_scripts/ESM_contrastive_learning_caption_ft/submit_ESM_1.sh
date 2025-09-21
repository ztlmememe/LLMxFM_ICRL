
#!/bin/bash

export HF_HOME=/mnt/hdd1/cache
cd /mnt/ssd/ztl/LLMxFM


SEEDS=("2021" "7777" "8888" "1111" "2222" "3333" "5555" "101010" "202020" "606060")


TASK_NAME="Fluorescence"
MODEL_NAME="/mnt/hdd1/cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

# "Stability", "Fluorescence"

for SEED in "${SEEDS[@]}"; do
    echo "Running for seed: $SEED"

    CUDA_VISIBLE_DEVICES=0 python -m chemistry.SC_scripts.ESM_run_trainer_pca_OT_icl_no_batch \
    --experiment_type rep --experiment_name no_batch_pca_ot_64_prot_layer_5 --seed $SEED --task_name $TASK_NAME --num_tests 1000 \
    --model_name $MODEL_NAME --esm_cache_path /mnt/ssd/ztl/LLMxFM/chemistry/datasets/prot_bert_aaseq_to_rep_store_layer_5.pkl \
    --esm_model_name "Rostlab/prot_bert"

    CUDA_VISIBLE_DEVICES=0 python -m chemistry.SC_scripts.ESM_run_trainer_pca_OT_icl_no_batch \
    --experiment_type rep_icl --experiment_name no_batch_pca_ot_64_prot_layer_5 --seed $SEED --task_name $TASK_NAME --num_tests 1000 \
    --model_name $MODEL_NAME --esm_cache_path /mnt/ssd/ztl/LLMxFM/chemistry/datasets/prot_bert_aaseq_to_rep_store_layer_5.pkl \
    --esm_model_name "Rostlab/prot_bert"


    
done
