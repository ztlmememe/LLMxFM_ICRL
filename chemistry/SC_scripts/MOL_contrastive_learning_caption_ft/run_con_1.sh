
export HF_HOME=/mnt/hdd1/cache
cd /mnt/ssd/ztl/LLMxFM
conda activate llmxfm

# fix layer
CUDA_VISIBLE_DEVICES=3 python -m chemistry.SC_scripts.MOL_trainer_contrastive_multiple_front \
    --task_names ESOL Caco2_wang LD50_Zhu Solubility_AqSolDB Lipophilicity_AstraZeneca\
    --llm_model_name "/mnt/hdd1/cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" \
    --unimol_feature_dim 512 \
    --projector_type "MLP_Linear" \
    --projector_arch "8196-8196" \
    --llm_target_layer_k 3 \
    --learning_rate 1e-4 \
    --batch_size 64 \
    --num_epochs 200 \
    --validation_frequency 3 \
    --save_every_n_epochs 50 \
    --wandb_project "multi_task_contrastive" \
    --base_run_name "LLM_align_fix_3_hid_layer" \
    --output_dir "/mnt/ssd/ztl/LLMxFM/chemistry/logs/trained_projectors" \
    --use_bfloat16 \
    --gradient_accumulation_steps 2

CUDA_VISIBLE_DEVICES=3 python -m chemistry.SC_scripts.MOL_trainer_contrastive_multiple_front \
    --task_names ESOL Caco2_wang LD50_Zhu Solubility_AqSolDB Lipophilicity_AstraZeneca\
    --llm_model_name "/mnt/hdd1/cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" \
    --unimol_feature_dim 512 \
    --projector_type "MLP_Nonlinear" \
    --projector_arch "5120-5120" \
    --llm_target_layer_k 3 \
    --learning_rate 1e-4 \
    --batch_size 64 \
    --num_epochs 200 \
    --validation_frequency 3 \
    --save_every_n_epochs 50 \
    --wandb_project "multi_task_contrastive" \
    --base_run_name "LLM_align_fix_3_hid_layer" \
    --output_dir "/mnt/ssd/ztl/LLMxFM/chemistry/logs/trained_projectors" \
    --use_bfloat16 \
    --gradient_accumulation_steps 2



# python -m chemistry.SC_scripts.MOL_trainer_contrastive_multiple2 \
#     --task_names ESOL Caco2_wang LD50_Zhu Solubility_AqSolDB Lipophilicity_AstraZeneca\
#     --llm_model_name "/mnt/hdd1/cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" \
#     --unimol_feature_dim 512 \
#     --projector_type "MLP_Linear" \
#     --projector_arch "64-64" \
#     --learning_rate 1e-4 \
#     --batch_size 32 \
#     --num_epochs 150 \
#     --validation_frequency 3 \
#     --save_every_n_epochs 50 \
#     --wandb_project "multi_task_contrastive" \
#     --base_run_name "LLM_align" \
#     --output_dir "/mnt/ssd/ztl/LLMxFM/chemistry/logs/trained_projectors" \
#     --use_bfloat16 \
#     --gradient_accumulation_steps 2


# python -m chemistry.SC_scripts.MOL_trainer_contrastive_multiple2 \
#     --task_names ESOL Caco2_wang LD50_Zhu Solubility_AqSolDB Lipophilicity_AstraZeneca\
#     --llm_model_name "/mnt/hdd1/cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" \
#     --unimol_feature_dim 512 \
#     --learning_rate 1e-4 \
#     --batch_size 64 \
#     --num_epochs 150 \
#     --validation_frequency 3 \
#     --save_every_n_epochs 50 \
#     --wandb_project "multi_task_contrastive" \
#     --base_run_name "LLM_align" \
#     --output_dir "/mnt/ssd/ztl/LLMxFM/chemistry/logs/trained_projectors" \
#     --use_bfloat16 \
#     --projector_type "MLP_Nonlinear" \
#     --gradient_accumulation_steps 2

# # random
# CUDA_VISIBLE_DEVICES=3 python -m chemistry.SC_scripts.MOL_trainer_contrastive_multiple3 \
#     --task_names ESOL Caco2_wang LD50_Zhu Solubility_AqSolDB Lipophilicity_AstraZeneca\
#     --llm_model_name "/mnt/hdd1/cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" \
#     --unimol_feature_dim 512 \
#     --projector_type "MLP_Linear" \
#     --projector_arch "8196-8196" \
#     --learning_rate 1e-4 \
#     --batch_size 64 \
#     --num_epochs 200 \
#     --validation_frequency 3 \
#     --save_every_n_epochs 50 \
#     --wandb_project "multi_task_contrastive" \
#     --base_run_name "LLM_align_random_hid_layer" \
#     --output_dir "/mnt/ssd/ztl/LLMxFM/chemistry/logs/trained_projectors" \
#     --use_bfloat16 \
#     --gradient_accumulation_steps 2

# # random
# CUDA_VISIBLE_DEVICES=3 python -m chemistry.SC_scripts.MOL_trainer_contrastive_multiple3 \
#     --task_names ESOL Caco2_wang LD50_Zhu Solubility_AqSolDB Lipophilicity_AstraZeneca\
#     --llm_model_name "/mnt/hdd1/cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" \
#     --unimol_feature_dim 512 \
#     --projector_type "MLP_Nonlinear" \
#     --projector_arch "5120-5120" \
#     --learning_rate 1e-4 \
#     --batch_size 64 \
#     --num_epochs 200 \
#     --validation_frequency 3 \
#     --save_every_n_epochs 50 \
#     --wandb_project "multi_task_contrastive" \
#     --base_run_name "LLM_align_random_hid_layer" \
#     --output_dir "/mnt/ssd/ztl/LLMxFM/chemistry/logs/trained_projectors" \
#     --use_bfloat16 \
#     --gradient_accumulation_steps 2

# python -m chemistry.SC_scripts.MOL_trainer_contrastive_multiple2 \
#     --task_names ESOL Caco2_wang LD50_Zhu Solubility_AqSolDB Lipophilicity_AstraZeneca\
#     --llm_model_name "/mnt/hdd1/cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" \
#     --unimol_feature_dim 512 \
#     --projector_type "MLP_Linear" \
#     --projector_arch "512-512" \
#     --learning_rate 1e-4 \
#     --batch_size 64 \
#     --num_epochs 150 \
#     --validation_frequency 3 \
#     --save_every_n_epochs 50 \
#     --wandb_project "multi_task_contrastive" \
#     --base_run_name "LLM_align" \
#     --output_dir "/mnt/ssd/ztl/LLMxFM/chemistry/logs/trained_projectors" \
#     --use_bfloat16 \
#     --gradient_accumulation_steps 2

# python -m chemistry.SC_scripts.MOL_trainer_contrastive_multiple2 \
#     --task_names ESOL Caco2_wang LD50_Zhu Solubility_AqSolDB Lipophilicity_AstraZeneca\
#     --llm_model_name "/mnt/hdd1/cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" \
#     --unimol_feature_dim 512 \
#     --projector_type "MLP_Linear" \
#     --projector_arch "2048-2048" \
#     --learning_rate 1e-4 \
#     --batch_size 64 \
#     --num_epochs 150 \
#     --validation_frequency 3 \
#     --save_every_n_epochs 50 \
#     --wandb_project "multi_task_contrastive" \
#     --base_run_name "LLM_align" \
#     --output_dir "/mnt/ssd/ztl/LLMxFM/chemistry/logs/trained_projectors" \
#     --use_bfloat16 \
#     --gradient_accumulation_steps 2


# python -m chemistry.SC_scripts.MOL_trainer_contrastive_multiple2 \
#     --task_names ESOL Caco2_wang LD50_Zhu Solubility_AqSolDB Lipophilicity_AstraZeneca\
#     --llm_model_name "/mnt/hdd1/cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" \
#     --unimol_feature_dim 512 \
#     --projector_type "MLP_Linear" \
#     --projector_arch "4096-4096" \
#     --learning_rate 1e-4 \
#     --batch_size 64 \
#     --num_epochs 150 \
#     --validation_frequency 3 \
#     --save_every_n_epochs 50 \
#     --wandb_project "multi_task_contrastive" \
#     --base_run_name "LLM_align" \
#     --output_dir "/mnt/ssd/ztl/LLMxFM/chemistry/logs/trained_projectors" \
#     --use_bfloat16 \
#     --gradient_accumulation_steps 2


# python -m chemistry.SC_scripts.MOL_trainer_contrastive_multiple3 \
#     --task_names ESOL Caco2_wang LD50_Zhu Solubility_AqSolDB Lipophilicity_AstraZeneca\
#     --llm_model_name "/mnt/hdd1/cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" \
#     --unimol_feature_dim 512 \
#     --projector_type "MLP_Linear" \
#     --projector_arch "4096-4096" \
#     --learning_rate 1e-4 \
#     --batch_size 64 \
#     --num_epochs 200 \
#     --validation_frequency 3 \
#     --save_every_n_epochs 50 \
#     --wandb_project "multi_task_contrastive" \
#     --base_run_name "LLM_align_random_hid_layer" \
#     --output_dir "/mnt/ssd/ztl/LLMxFM/chemistry/logs/trained_projectors" \
#     --use_bfloat16 \
#     --gradient_accumulation_steps 2



# python -m chemistry.SC_scripts.MOL_trainer_contrastive_multiple2 \
#     --task_names ESOL Caco2_wang LD50_Zhu Solubility_AqSolDB Lipophilicity_AstraZeneca\
#     --llm_model_name "/mnt/hdd1/cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" \
#     --unimol_feature_dim 512 \
#     --projector_type "MLP_Linear" \
#     --projector_arch "5120-5120" \
#     --learning_rate 1e-4 \
#     --batch_size 64 \
#     --num_epochs 150 \
#     --validation_frequency 3 \
#     --save_every_n_epochs 50 \
#     --wandb_project "multi_task_contrastive" \
#     --base_run_name "LLM_align" \
#     --output_dir "/mnt/ssd/ztl/LLMxFM/chemistry/logs/trained_projectors" \
#     --use_bfloat16 \
#     --gradient_accumulation_steps 2


# python -m chemistry.SC_scripts.MOL_trainer_contrastive_multiple3 \
#     --task_names ESOL Caco2_wang LD50_Zhu Solubility_AqSolDB Lipophilicity_AstraZeneca\
#     --llm_model_name "/mnt/hdd1/cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" \
#     --unimol_feature_dim 512 \
#     --projector_type "MLP_Linear" \
#     --projector_arch "5120-5120" \
#     --learning_rate 1e-4 \
#     --batch_size 64 \
#     --num_epochs 200 \
#     --validation_frequency 3 \
#     --save_every_n_epochs 50 \
#     --wandb_project "multi_task_contrastive" \
#     --base_run_name "LLM_align_random_hid_layer" \
#     --output_dir "/mnt/ssd/ztl/LLMxFM/chemistry/logs/trained_projectors" \
#     --use_bfloat16 \
#     --gradient_accumulation_steps 2