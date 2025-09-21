
export HF_HOME=/mnt/hdd1/cache
cd /mnt/ssd/ztl/LLMxFM
conda activate llmxfm

# fix layer
CUDA_VISIBLE_DEVICES=2 python -m chemistry.SC_scripts.ESM_trainer_contrastive_multiple \
    --task_names Stability \
    --llm_model_name "/mnt/hdd1/cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" \
    --esm_feature_dim 640 \
    --projector_type "MLP_Linear" \
    --projector_arch "8192-8192" \
    --llm_target_layer_k -1 \
    --learning_rate 1e-3 \
    --batch_size 64 \
    --num_epochs 200 \
    --validation_frequency 3 \
    --save_every_n_epochs 100 \
    --wandb_project "ESM_single_task_contrastive" \
    --base_run_name "LLM_align_random_layer_esm_layer_6" \
    --output_dir "/mnt/ssd/ztl/LLMxFM/chemistry/logs/trained_projectors" \
    --esm_cache_path /mnt/ssd/ztl/LLMxFM/chemistry/datasets/aaseq_to_rep_store_layer_6.pkl \
    --use_bfloat16 \
    --gradient_accumulation_steps 4

CUDA_VISIBLE_DEVICES=2 python -m chemistry.SC_scripts.ESM_trainer_contrastive_multiple \
    --task_names Stability \
    --llm_model_name "/mnt/hdd1/cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" \
    --esm_feature_dim 640 \
    --projector_type "MLP_Nonlinear" \
    --projector_arch "8192-8192" \
    --llm_target_layer_k -1 \
    --learning_rate 1e-3 \
    --batch_size 64 \
    --num_epochs 200 \
    --validation_frequency 3 \
    --save_every_n_epochs 100 \
    --wandb_project "ESM_single_task_contrastive" \
    --base_run_name "LLM_align_random_layer_esm_layer_6" \
    --output_dir "/mnt/ssd/ztl/LLMxFM/chemistry/logs/trained_projectors" \
    --esm_cache_path /mnt/ssd/ztl/LLMxFM/chemistry/datasets/aaseq_to_rep_store_layer_6.pkl \
    --use_bfloat16 \
    --gradient_accumulation_steps 4