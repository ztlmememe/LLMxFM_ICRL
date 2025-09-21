
export HF_HOME=/mnt/hdd1/cache
cd /mnt/ssd/ztl/LLMxFM

CUDA_VISIBLE_DEVICES=2 python -m chemistry.SC_scripts.MOL_trainer_caption_contrastive \
    --batch_size 16 \
    --num_epochs 5 \
    --unimol_feature_dim 512 \
    --projector_type "MLP_Nonlinear" \
    --projector_arch "4096-4096" \
    --llm_model_name "/mnt/hdd1/cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" \
    --learning_rate 1e-3 \
    --weight_decay 0.01 \
    --use_bfloat16 \
    --skip_unimol_errors \
    --gradient_accumulation_steps 8 \
    --wandb_project "caption_ft" \
    --base_run_name "contrastive" \
    --llm_target_layer_k -1 \
    --contrastive_loss_weight 1 \
    --contrastive_temperature_tau 0.07 \
    --output_dir "/mnt/ssd/ztl/LLMxFM/chemistry/logs/trained_projectors"


# CUDA_VISIBLE_DEVICES=2 python -m chemistry.SC_scripts.MOL_trainer_caption \
#     --batch_size 16 \
#     --num_epochs 5 \
#     --unimol_feature_dim 512 \
#     --projector_type "MLP_Linear" \
#     --projector_arch "8196-8196" \
#     --llm_model_name "/mnt/hdd1/cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" \
#     --learning_rate 1e-3 \
#     --weight_decay 0.01 \
#     --use_bfloat16 \
#     --skip_unimol_errors \
#     --gradient_accumulation_steps 8 \
#     --wandb_project "caption_ft" \
#     --base_run_name "standard" \
#     --output_dir "/mnt/ssd/ztl/LLMxFM/chemistry/logs/trained_projectors"




# CUDA_VISIBLE_DEVICES=2 python -m chemistry.SC_scripts.MOL_trainer_caption \
#     --batch_size 16 \
#     --num_epochs 5 \
#     --unimol_feature_dim 512 \
#     --projector_type "Vicl_Linear" \
#     --projector_arch "4096-4096" \
#     --llm_model_name "/mnt/hdd1/cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" \
#     --learning_rate 1e-3 \
#     --weight_decay 0.01 \
#     --use_bfloat16 \
#     --skip_unimol_errors \
#     --gradient_accumulation_steps 8 \
#     --wandb_project "caption_ft" \
#     --base_run_name "standard" \
#     --output_dir "/mnt/ssd/ztl/LLMxFM/chemistry/logs/trained_projectors"

