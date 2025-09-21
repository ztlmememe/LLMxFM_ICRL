import argparse
import datetime
import gc
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["WANDB_MODE"] = "offline" # Uncomment this line to force offline mode
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb
from chemistry.utils.esm_utils import *
from chemistry.utils.models import MLP_Linear, MLP_Nonlinear
from chemistry.utils.utils import *
import transformers
# --- Task Type Dictionary (Updated for Protein Tasks) ---
task_type_dict = {
    "BindingDB_Ki": "DTI",
    "BindingDB_IC50": "DTI",
    "KIBA": "DTI",
    "DAVIS": "DTI",

    "Fluorescence": "PROTEIN",
    "Stability": "PROTEIN",
    "Beta_Lactamase": "PROTEIN",
    "PPI_Affinity": "PROTEIN", # Added PPI_Affinity

    "Caco2_wang":"MOL",
    "ESOL": "MOL",
    "Solubility_AqSolDB": "MOL",
    "Half_Life_Obach": "MOL",
    "Clearance_Hepatocyte_AZ": "MOL",
    "HydrationFreeEnergy_FreeSolv": "MOL",
    "Lipophilicity_AstraZeneca": "MOL",
    "LD50_Zhu": "MOL",
}

# --- Configuration & Argument Parsing (Updated for Protein) ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train a projector for Protein (Amino Acid) representations using contrastive learning on multiple datasets.")
    # Changed task_name to task_names, accepts multiple values, default list for proteins
    parser.add_argument("--task_names", type=str, nargs='+', default=["Stability"],
                        help="List of dataset/task names to use (e.g., Stability Fluorescence).")
    parser.add_argument("--base_data_path", type=str, default="./../datasets/",
                        help="Base path for datasets and representation caches.")
    parser.add_argument("--output_dir", type=str, default="./protein_contrastive_output", # Changed output dir
                        help="Directory to save trained models and logs.")

    parser.add_argument("--llm_model_name", type=str, default="meta-llama/Meta-Llama-3-8B",
                        help="Hugging Face model name for LLM.")
    parser.add_argument("--llm_target_layer_k", type=int, default=-1,
                        help="Target layer in LLM to extract features from (-1 for last).")
    parser.add_argument("--llm_max_length", type=int, default=1024, # Increased for proteins
                        help="Max sequence length for LLM tokenizer.")

    # Added ESM arguments, removed UniMol
    parser.add_argument("--esm_model_name", type=str, default="facebook/esm2_t30_150M_UR50D",
                        help="ESM model name for protein feature extraction.")
    parser.add_argument("--esm_feature_dim", type=int, default=1280, # Default for esm2_t33_650M
                        help="Feature dimension of ESM output.")
    parser.add_argument("--projector_arch", type=str, default="1024-2048",
                        help="Architecture of the MLP projector (e.g., '1280-2048', empty for linear).")
    parser.add_argument("--projector_dropout", type=float, default=0.1,
                        help="Dropout rate for projector MLP.")
    parser.add_argument("--projector_type", type=str, default="MLP_Linear",
                        help="Type of projector architecture (MLP_Nonlinear or MLP_Linear).")

    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer.")
    parser.add_argument("--batch_size", type=int, default=16, # Reduced BS for potentially larger protein models
                        help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--contrastive_temperature_tau", type=float, default=0.07, help="Temperature parameter for InfoNCE loss.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer.")
    # --weight_decay 0.01 \

    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader.")
    parser.add_argument("--use_bfloat16", action='store_true', help="Use bfloat16 for LLM and projector if available.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, # Increased accumulation
                        help="Number of steps to accumulate gradients before an optimizer step.")

    parser.add_argument("--wandb_project", type=str, default="contrastive_protein_projector", # Changed project name
                        help="WANDB project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WANDB entity (username or team).")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WANDB run name (optional, defaults to auto-generated).")
    parser.add_argument("--skip_esm_errors", action='store_true', # Renamed from unimol
                        help="Skip sequences that cause errors in ESM processing instead of crashing.")
    parser.add_argument("--validation_frequency", type=int, default=1,
                        help="Frequency (in epochs) to perform validation.")
    parser.add_argument("--save_every_n_epochs", type=int, default=5,
                        help="Frequency (in epochs) to save the latest model checkpoint.")
    parser.add_argument("--base_run_name", type=str, default="protein_contrastive_projector", # Changed base name
                        help="Base name for WANDB run.")
    # Added placeholder args for protein loaders if needed, or assume they adapt
    parser.add_argument("--num_trains", type=int, default=10000, help="Placeholder for protein loaders.")
    parser.add_argument("--num_tests", type=int, default=2000, help="Placeholder for protein loaders.")

    parser.add_argument("--esm_cache_path", type=str, default=None, help="Path to the ESM cache.")


    return parser.parse_args()

# --- Dataset Loading & Preparation (Modified for multiple Protein tasks) ---
class SequenceDataset(Dataset): # Renamed from SmilesDataset
    def __init__(self, sequence_list):
        self.sequence_list = [s for s in sequence_list if isinstance(s, str) and s.strip()]
    def __len__(self):
        return len(self.sequence_list)
    def __getitem__(self, idx):
        return self.sequence_list[idx]

def get_dataloaders(args):
    all_train_sequences = []
    all_valid_sequences = []

    # Helper to flatten sequence lists
    def flatten_sequences(inputs_raw):
        if inputs_raw is None: return []
        if isinstance(inputs_raw, (pd.Series, pd.DataFrame)): inputs_raw = inputs_raw.to_numpy()
        if isinstance(inputs_raw, np.ndarray): inputs_raw = inputs_raw.flatten().tolist()
        # Handle cases where inputs might be nested (e.g., pairs for PPI) - take first element for now
        processed_sequences = [str(item[0]) if isinstance(item, (list, np.ndarray)) else str(item) for item in inputs_raw]
        return [s for s in processed_sequences if s.strip()]

    print(f"Loading data for tasks: {args.task_names}")
    for task_name in args.task_names:
        print(f"--> Loading task: {task_name}")
        train_inputs_raw, valid_inputs_raw = None, None
        # Placeholders
        train_y, valid_y, test_inputs, test_y = None, None, None, None

        try:
            # Determine which loading function to use based on task name
            # NOTE: These loaders might need adaptation or specific implementations
            # They need to return at least train_inputs and valid_inputs (sequences)
            args.base_data_path = os.path.join(os.path.dirname(__file__), "./../datasets/")
            if task_name == "Stability":
                train_inputs_raw, _, valid_inputs_raw, _, _, _ = load_Stability(-1, -1, args.base_data_path)
            elif task_name == "Fluorescence":
                train_inputs_raw, _, valid_inputs_raw, _, _, _ = load_Fluorescence(-1, -1, args.base_data_path)
            else:
                print(f"    Warning: Unsupported task_name '{task_name}'. Skipping.")
                continue

            task_train_sequences = flatten_sequences(train_inputs_raw)
            task_valid_sequences = flatten_sequences(valid_inputs_raw)

            # Optional: Add illegal sequence check if needed

            if not task_train_sequences:
                 print(f"    Warning: No training sequences found for task '{task_name}'.")
            else:
                print(f"    Loaded {len(task_train_sequences)} train, {len(task_valid_sequences)} valid sequences.")
                all_train_sequences.extend(task_train_sequences)
                all_valid_sequences.extend(task_valid_sequences)

        except Exception as e:
            print(f"    Error loading data for task '{task_name}': {e}. Skipping this task.")
            continue

    # --- Combine and Deduplicate ---
    if not all_train_sequences:
        raise ValueError("No training data loaded successfully for any specified task.")

    print("-" * 30)
    print(f"Total sequences loaded before deduplication: Train={len(all_train_sequences)}, Valid={len(all_valid_sequences)}")

    all_train_sequences = sorted(list(set(all_train_sequences)))
    all_valid_sequences = sorted(list(set(all_valid_sequences)))
    print(f"Total unique sequences after deduplication: Train={len(all_train_sequences)}, Valid={len(all_valid_sequences)}")

    train_sequences_set = set(all_train_sequences)
    original_valid_count = len(all_valid_sequences)
    all_valid_sequences = [s for s in all_valid_sequences if s not in train_sequences_set]
    removed_valid_count = original_valid_count - len(all_valid_sequences)
    if removed_valid_count > 0:
        print(f"Removed {removed_valid_count} validation sequences present in the training set.")
    print(f"Final dataset sizes: Train={len(all_train_sequences)}, Valid={len(all_valid_sequences)}")
    print("-" * 30)

    if not all_train_sequences: raise ValueError("Training set is empty.")
    valid_dataloader = None
    if all_valid_sequences:
        valid_dataset = SequenceDataset(all_valid_sequences)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    else:
        print("Warning: Validation set is empty.")

    train_dataset = SequenceDataset(all_train_sequences)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    return train_dataloader, valid_dataloader


# --- LLM Feature Extraction (Modified to take protein sequences) ---
def get_llm_text_and_projected_features(
    sequence_batch, # Changed from smiles_batch
    projected_features,
    llm,
    tokenizer,
    layer_k_arg,
    device,
    target_dtype,
    max_length=1024, # Increased default
    sampling_strategy="random"
):
    llm.eval()

    # === Tokenize Text (Protein Sequences) ===
    with torch.no_grad():
        inputs = tokenizer(
            sequence_batch, # Use protein sequences here
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_length,
        ).to(device)

        text_outputs = llm(**inputs, output_hidden_states=True)
        text_hidden_states_tuple = text_outputs.hidden_states
        num_transformer_layers = len(text_hidden_states_tuple) - 1

    # === Determine Sampling Layer ===
    if sampling_strategy == "fixed" and (1 <= layer_k_arg <= num_transformer_layers):
        actual_layer_index = layer_k_arg
    elif sampling_strategy == "random":
        third = num_transformer_layers // 3
        segment = random.choice(["front", "middle", "back"])
        if segment == "front":
            actual_layer_index = random.randint(1, max(1, third))
        elif segment == "middle":
            actual_layer_index = random.randint(third + 1, 2 * third)
        else:
            actual_layer_index = random.randint(2 * third + 1, num_transformer_layers)
    elif sampling_strategy == "front":
        actual_layer_index = random.randint(1, 5)
    elif sampling_strategy == "middle":
        actual_layer_index = random.randint(num_transformer_layers // 3 - 3, 2 * num_transformer_layers // 3 + 3)
    elif sampling_strategy == "back":
        actual_layer_index = random.randint(num_transformer_layers - 4, num_transformer_layers)
    else: # Default or fixed if -1
         actual_layer_index = layer_k_arg if layer_k_arg != -1 else num_transformer_layers


    # === Get text hidden ===
    text_hidden_layer = text_hidden_states_tuple[actual_layer_index]
    text_last_token_features = text_hidden_layer[:, -1, :].to(dtype=target_dtype)

    # === Get projected vector LLM output ===
    inputs_embeds = projected_features.unsqueeze(1).to(device)
    if inputs_embeds.dtype != llm.dtype:
        inputs_embeds = inputs_embeds.to(llm.dtype)
    proj_outputs = llm(inputs_embeds=inputs_embeds, output_hidden_states=True)
    proj_hidden_states_tuple = proj_outputs.hidden_states
    proj_hidden_layer = proj_hidden_states_tuple[actual_layer_index]
    proj_token_features = proj_hidden_layer[:, -1, :].to(dtype=target_dtype)

    print(f"Using LLM hidden layer {actual_layer_index}/{num_transformer_layers}.")

    return text_last_token_features, proj_token_features


# --- Model Initialization (Updated for ESM) ---
def get_models(args, device, dtype):
    # ESM Model
    cache_path = args.esm_cache_path if args.esm_cache_path else f"{args.base_data_path}/aaseq_to_rep_store.pkl"

    protein_fm = esm_model(
        esm_model_name=args.esm_model_name,
        avoid_loading_model=False,
        rep_cache_path=cache_path
    )
    if hasattr(protein_fm, 'model') and protein_fm.model is not None and isinstance(protein_fm.model, nn.Module):
        protein_fm.model.to(device).eval()
    elif isinstance(protein_fm, nn.Module):
        protein_fm.to(device).eval()
    else:
        try: protein_fm.to(device)
        except: pass
        try: protein_fm.eval()
        except: pass

    # LLM Tokenizer
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_model_name)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
        if llm_tokenizer.pad_token is None: llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        llm_tokenizer.pad_token_id = llm_tokenizer.convert_tokens_to_ids(llm_tokenizer.pad_token)
    llm_tokenizer.padding_side = "left"

    # LLM Model
    llm_model_kwargs = {}
    if args.use_bfloat16 and torch.cuda.is_bf16_supported(): llm_model_kwargs["torch_dtype"] = torch.bfloat16
    elif dtype == torch.float16 and torch.cuda.is_available(): llm_model_kwargs["torch_dtype"] = torch.float16
    llm_model = AutoModelForCausalLM.from_pretrained(args.llm_model_name, **llm_model_kwargs)
    if "device_map" not in llm_model_kwargs: llm_model = llm_model.to(device)
    llm_model.eval()

    projector_input_dim = args.esm_feature_dim # Use ESM feature dim
    try:
        projector_output_dim = llm_model.config.hidden_size
        print(f"Inferred LLM hidden_size for projector output: {projector_output_dim}")
    except Exception as e:
        projector_output_dim = 4096 # Fallback
        print(f"Warning: Could not infer LLM hidden_size. Using fallback: {projector_output_dim}")

    if args.projector_type == "MLP_Nonlinear":
        projector = MLP_Nonlinear(ninput=projector_input_dim, noutput=projector_output_dim).to(device).to(dtype)
    elif args.projector_type == "MLP_Linear":
        projector = MLP_Linear(ninput=projector_input_dim, noutput=projector_output_dim, layers=args.projector_arch).to(device).to(dtype)

    return protein_fm, llm_tokenizer, llm_model, projector, projector_output_dim


# --- Helper Functions (Updated for ESM) ---
@torch.no_grad()
def get_esm_features(sequence_batch, protein_fm, device, target_dtype, skip_errors=False):
    fm_reps_list = []
    valid_sequences_in_batch = []

    for seq_str in sequence_batch:
        try:
            esm_rep = protein_fm.get_esm_rep_tensor(seq_str, single_input_batch=True)
            if esm_rep is None or esm_rep.nelement() == 0:
                raise ValueError("ESM returned empty tensor.")
            fm_reps_list.append(esm_rep.to(device, dtype=target_dtype))
            valid_sequences_in_batch.append(seq_str)
        except Exception as e:
            if skip_errors:
                print(f"Skipping sequence '{seq_str[:30]}...' due to ESM error: {e}")
                continue
            else:
                raise e

    if not fm_reps_list:
        return None, []

    fm_reps_batch = torch.cat(fm_reps_list, dim=0)
    return fm_reps_batch, valid_sequences_in_batch

# get_llm_text_features can be kept if needed for validation/debugging,
# but get_llm_text_and_projected_features is now the primary one used.

def info_nce_loss(features_a, features_b, temperature):
    features_a = F.normalize(features_a, p=2, dim=1)
    features_b = F.normalize(features_b, p=2, dim=1)
    logits = torch.matmul(features_a, features_b.T) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_a_b = F.cross_entropy(logits, labels)
    loss_b_a = F.cross_entropy(logits.T, labels)
    loss = (loss_a_b + loss_b_a) / 2
    return loss

# --- Main Training Function (Adapted for Protein) ---
def train_projector(args):
    set_random_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.use_bfloat16 and torch.cuda.is_bf16_supported(): dtype = torch.bfloat16
    elif torch.cuda.is_available(): dtype = torch.float16
    else: dtype = torch.float32
    print(f"Using dtype: {dtype}")

    # --- WANDB Initialization ---
    tasks_str = "_".join(sorted(args.task_names))
    if args.projector_type == "MLP_Nonlinear":
        run_name_prefix = f"{args.base_run_name}_Nonlinear_lr_{args.learning_rate}_wd_{args.weight_decay}_batch_{args.batch_size}_epochs{args.num_epochs}_{tasks_str}"
    elif args.projector_type == "MLP_Linear":
        run_name_prefix = f"{args.base_run_name}_Linear_arch_{args.projector_arch}_lr_{args.learning_rate}_wd_{args.weight_decay}_batch_{args.batch_size}_epochs{args.num_epochs}_{tasks_str}"

    run_name = args.wandb_run_name if args.wandb_run_name else f"{run_name_prefix}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, config=args)

    # --- DataLoaders ---
    train_dataloader, valid_dataloader = get_dataloaders(args)
    if train_dataloader is None:
        print("Exiting: No training data loaded.")
        return

    # --- Models ---
    protein_fm, llm_tokenizer, llm_model, projector, projector_output_dim_actual = get_models(args, device, dtype)
    args.projector_output_dim = projector_output_dim_actual
    wandb.config.update({"projector_output_dim_actual": projector_output_dim_actual}, allow_val_change=True)

    optimizer = optim.AdamW(projector.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    num_training_steps = (len(train_dataloader) * args.num_epochs) // args.gradient_accumulation_steps
    num_warmup_steps = int(0.1 * num_training_steps)
    lr_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    best_valid_loss = float('inf')
    args.output_dir = os.path.join(args.output_dir, tasks_str, run_name_prefix, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n--- Starting Training ---")
    for epoch in range(args.num_epochs):
        projector.train()
        total_train_loss = 0
        train_batches = 0
        optimizer.zero_grad()

        progress_bar_train = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")
        for batch_idx, sequence_batch in progress_bar_train:
            if not sequence_batch: continue

            fm_reps_batch, valid_sequences_fm = get_esm_features(sequence_batch, protein_fm, device, dtype, args.skip_esm_errors)
            if fm_reps_batch is None: continue

            proj_param_dtype = next(projector.parameters()).dtype
            projected_features = projector(fm_reps_batch.to(proj_param_dtype))

            target_text_features, projected_features = get_llm_text_and_projected_features(
                sequence_batch=valid_sequences_fm,
                projected_features=projected_features,
                llm=llm_model,
                tokenizer=llm_tokenizer,
                layer_k_arg=args.llm_target_layer_k,
                device=device,
                target_dtype=projected_features.dtype,
                max_length=args.llm_max_length,
                sampling_strategy="random" # Use random during training
            )

            if projected_features.shape[0] != target_text_features.shape[0] or projected_features.shape[0] == 0: continue

            loss = info_nce_loss(projected_features, target_text_features, args.contrastive_temperature_tau)
            loss = loss / args.gradient_accumulation_steps

            loss.backward()
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            current_loss = loss.item() * args.gradient_accumulation_steps
            total_train_loss += current_loss
            train_batches += 1
            progress_bar_train.set_postfix({"loss": f"{current_loss:.4f}"})
            if (batch_idx + 1) % (args.gradient_accumulation_steps * 10) == 0:
                 wandb.log({"train/batch_loss": current_loss})

        avg_train_loss = total_train_loss / train_batches if train_batches > 0 else 0
        wandb.log({"train/epoch_loss": avg_train_loss, "epoch": epoch + 1,"lr": lr_scheduler.get_last_lr()[0]})
        print(f"Epoch [{epoch+1}/{args.num_epochs}] Training Loss: {avg_train_loss:.4f}")

        # --- Validation Step ---
        perform_validation = (valid_dataloader is not None) and \
                             (args.validation_frequency > 0 and ((epoch + 1) % args.validation_frequency == 0 or (epoch + 1) == args.num_epochs))

        if perform_validation:
            projector.eval()
            total_valid_loss = 0
            valid_batches = 0
            progress_bar_valid = tqdm(valid_dataloader, total=len(valid_dataloader), desc=f"Epoch {epoch+1}/{args.num_epochs} [Valid]", leave=False)
            with torch.no_grad():
                for sequence_batch_valid in progress_bar_valid:
                    if not sequence_batch_valid: continue

                    fm_reps_valid, valid_seqs_fm_val = get_esm_features(sequence_batch_valid, protein_fm, device, dtype, args.skip_esm_errors)
                    if fm_reps_valid is None: continue

                    proj_param_dtype = next(projector.parameters()).dtype
                    projected_valid = projector(fm_reps_valid.to(proj_param_dtype))

                    target_text_valid, projected_valid = get_llm_text_and_projected_features(
                            sequence_batch=valid_seqs_fm_val,
                            projected_features=projected_valid,
                            llm=llm_model,
                            tokenizer=llm_tokenizer,
                            layer_k_arg=args.llm_target_layer_k,
                            device=device,
                            target_dtype=proj_param_dtype,
                            max_length=args.llm_max_length,
                            sampling_strategy="fixed" # Use fixed (-1, last layer) for validation consistency
                        )

                    if projected_valid.shape[0] != target_text_valid.shape[0] or projected_valid.shape[0] == 0 : continue

                    valid_loss = info_nce_loss(projected_valid, target_text_valid, args.contrastive_temperature_tau)
                    total_valid_loss += valid_loss.item()
                    valid_batches += 1
                    progress_bar_valid.set_postfix({"loss": f"{valid_loss.item():.4f}"})

            avg_valid_loss = total_valid_loss / valid_batches if valid_batches > 0 else float('inf')
            wandb.log({"valid/epoch_loss": avg_valid_loss, "epoch": epoch + 1})
            print(f"Epoch [{epoch+1}/{args.num_epochs}] Validation Loss: {avg_valid_loss:.4f}")

            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                model_path = os.path.join(args.output_dir, f"{tasks_str}_best_projector.pth")
                torch.save(projector.state_dict(), model_path)
                if os.path.exists(model_path): wandb.save(model_path)
                print(f"Saved best model to {model_path} with loss {best_valid_loss:.4f}")

        # --- Save Latest Model ---
        if args.save_every_n_epochs > 0 and ((epoch + 1) % args.save_every_n_epochs == 0 or (epoch + 1) == args.num_epochs):
            latest_model_path = os.path.join(args.output_dir, f"{tasks_str}_latest_projector_epoch{epoch+1}.pth")
            torch.save(projector.state_dict(), latest_model_path)
            print(f"Saved latest model checkpoint to {latest_model_path}")

        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    final_model_path = os.path.join(args.output_dir, f"{tasks_str}_final_projector.pth")
    torch.save(projector.state_dict(), final_model_path)
    if os.path.exists(final_model_path): wandb.save(final_model_path)
    print(f"Training complete. Final projector saved to {final_model_path}")
    wandb.finish()
    print("--- Training Finished ---")


if __name__ == "__main__":
    args = parse_args()

    print("--- Starting Protein Contrastive Projector Training ---")
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("-" * 30)
    train_projector(args)