#!/bin/bash

# Run script for executing the LoRA fine-tuning process
# Set your WANDB API key here or export it before running this script
# export WANDB_API_KEY="your-wandb-api-key"

# Create output directory
OUTPUT_DIR="/mnt/ssd/ztl/LLMxFM/chemistry/logs/lora_output/$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# --model_name "models--meta-llama--Llama-3.2-3B-Instruct" \
# Run the trainer
python -m chemistry.SC_scripts.run_lora_trainer \
  --output_dir $OUTPUT_DIR \
  --experiment_type "random_noise" \
  --num_train_maps 50 \
  --num_test_maps 5 \
  --samples_per_map 1000 \
  --train_seed_range 9999999 99999999 \
  --test_seed_range 1 3000 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --num_epochs 20 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --target_modules "q_proj,v_proj,k_proj" \
  --wandb_project "lora-embedding-training" \
  --wandb_run_name "llama-3.2-3b-embedding-training2"