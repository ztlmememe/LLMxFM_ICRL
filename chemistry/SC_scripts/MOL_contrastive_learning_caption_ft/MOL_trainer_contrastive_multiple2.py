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
from transformers import AutoTokenizer, AutoModelForCausalLM # Using AutoModelForCausalLM for broader compatibility
import wandb
from chemistry.utils.unimol_utils import *
from chemistry.utils.models import MLP_Linear,MLP_Nonlinear
from chemistry.utils.utils import *

# --- Task Type Dictionary (keep as reference if needed) ---
task_type_dict = {
    "BindingDB_Ki": "DTI",
    "BindingDB_IC50": "DTI",
    "KIBA": "DTI",
    "DAVIS": "DTI",

    "Fluorescence": "ESM",
    "Stability": "ESM",
    "Beta_Lactamase": "ESM",

    "Caco2_wang":"MOL",
    "ESOL": "MOL",
    "Solubility_AqSolDB": "MOL",
    "Half_Life_Obach": "MOL",
    "Clearance_Hepatocyte_AZ": "MOL",
    "HydrationFreeEnergy_FreeSolv": "MOL",
    "Lipophilicity_AstraZeneca": "MOL",
    "LD50_Zhu": "MOL",
}

# --- Configuration & Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train a projector for SMILES representations using contrastive learning on multiple datasets.")
    # Changed task_name to task_names, accepts multiple values, default list
    parser.add_argument("--task_names", type=str, nargs='+', default=["ESOL"],
                        help="List of dataset/task names to use (e.g., ESOL Caco2_wang).")
    parser.add_argument("--base_data_path", type=str, default="./../datasets/", # Adjusted default relative path
                        help="Base path for datasets and representation caches.")
    parser.add_argument("--output_dir", type=str, default="./contrastive_output",
                        help="Directory to save trained models and logs.")

    parser.add_argument("--llm_model_name", type=str, default="meta-llama/Meta-Llama-3-8B", # Changed default LLM
                        help="Hugging Face model name for LLM.")
    parser.add_argument("--llm_target_layer_k", type=int, default=-1,
                        help="Target layer in LLM to extract features from (e.g., 12 for 12th block, -1 for last). 1-indexed if positive.")
    parser.add_argument("--llm_max_length", type=int, default=128,
                        help="Max sequence length for LLM tokenizer.")

    parser.add_argument("--unimol_feature_dim", type=int, default=640, # Changed default to 640
                        help="Feature dimension of UniMol output. Check your UniMol checkpoint (e.g., 512 or 640).")
    parser.add_argument("--projector_arch", type=str, default="1024-2048",
                        help="Architecture of the MLP projector (e.g., '512-1024', empty for linear). Hidden layers separated by '-'.")
    parser.add_argument("--projector_dropout", type=float, default=0.1,
                        help="Dropout rate for projector MLP.")
    parser.add_argument("--projector_type", type=str, default="MLP_Linear",
                        help="Type of projector architecture (MLP_Nonlinear or MLP_Linear).")


    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--contrastive_temperature_tau", type=float, default=0.07, help="Temperature parameter for InfoNCE loss.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer.")


    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader.")
    parser.add_argument("--use_bfloat16", action='store_true', help="Use bfloat16 for LLM and projector if available.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients before an optimizer step.")

    parser.add_argument("--wandb_project", type=str, default="contrastive_smiles_projector", help="WANDB project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WANDB entity (username or team).")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WANDB run name (optional, defaults to auto-generated).")
    parser.add_argument("--skip_unimol_errors", action='store_true', help="Skip SMILES that cause errors in UniMol processing instead of crashing.")
    # Added arguments for validation and saving frequency
    parser.add_argument("--validation_frequency", type=int, default=1,
                        help="Frequency (in epochs) to perform validation. Default: 1 (every epoch). Set <= 0 to validate only at the end.")
    parser.add_argument("--save_every_n_epochs", type=int, default=5,
                        help="Frequency (in epochs) to save the latest model checkpoint. Default: 5.")
    
    parser.add_argument("--base_run_name", type=str, default="contrastive_projector", help="Base name for WANDB run.")

    return parser.parse_args()

# --- Dataset Loading & Preparation (Modified for multiple tasks) ---
class SmilesDataset(Dataset):
    def __init__(self, smiles_list):
        self.smiles_list = [s for s in smiles_list if isinstance(s, str) and s.strip()]
    def __len__(self):
        return len(self.smiles_list)
    def __getitem__(self, idx):
        return self.smiles_list[idx]

def get_dataloaders(args):
    all_train_smiles = []
    all_valid_smiles = []

    # Helper to flatten SMILES lists potentially nested in arrays/lists
    def flatten_smiles(inputs_raw):
        if inputs_raw is None: return []
        if isinstance(inputs_raw, (pd.Series, pd.DataFrame)): inputs_raw = inputs_raw.to_numpy()
        if isinstance(inputs_raw, np.ndarray): inputs_raw = inputs_raw.flatten().tolist()
        processed_smiles = [str(item[0]) if isinstance(item, (list, np.ndarray)) else str(item) for item in inputs_raw]
        return [s for s in processed_smiles if s.strip()]

    print(f"Loading data for tasks: {args.task_names}")
    # Loop through each specified task name
    for task_name in args.task_names:
        print(f"--> Loading task: {task_name}")
        train_inputs_raw, valid_inputs_raw = None, None
        # These are placeholders if labels/test sets are needed by loaders but not used here
        train_y, valid_y, test_inputs, test_y = None, None, None, None

        try:
            # Determine which loading function to use based on task name
            if task_name in ["Caco2_wang", "Solubility_AqSolDB", "Half_Life_Obach",
                             "Clearance_Hepatocyte_AZ", "HydrationFreeEnergy_FreeSolv",
                             "Lipophilicity_AstraZeneca"]:
                data = data_split_loading_ADME(dataset_name=task_name, split_type='random',
                                               seed=args.random_seed, split_fracs=[0.7, 0.1, 0.2])
                train_inputs_raw = data.get("train", {}).get("SMILES")
                valid_inputs_raw = data.get("valid", {}).get("SMILES")
            elif task_name == "ESOL":
                train_inputs_raw, _, valid_inputs_raw, _, _, _ = load_ESOL()
            elif task_name == "LD50_Zhu":
                data = data_split_loading_Tox(dataset_name=task_name, split_type='random',
                                              seed=args.random_seed, split_fracs=[0.7, 0.1, 0.2])
                train_inputs_raw = data.get("train", {}).get("SMILES")
                valid_inputs_raw = data.get("valid", {}).get("SMILES")
            else:
                print(f"    Warning: Unsupported task_name '{task_name}'. Skipping.")
                continue # Skip to next task_name

            # Flatten the loaded SMILES
            task_train_smiles_raw = flatten_smiles(train_inputs_raw)
            task_valid_smiles_raw = flatten_smiles(valid_inputs_raw)

            # remove illegal SMILES
            # /mnt/ssd/ztl/LLMxFM/chemistry/datasets/illegal_smiles.txt
            illegal_smiles = set()
            illegal_smiles_file = os.path.join('/mnt/ssd/ztl/LLMxFM/chemistry/datasets', 'illegal_smiles.txt')

            with open(illegal_smiles_file, "r") as f:
                illegal_smiles = set(line.strip() for line in f)

            task_train_smiles = [s for s in task_train_smiles_raw if s not in illegal_smiles]
            task_valid_smiles = [s for s in task_valid_smiles_raw if s not in illegal_smiles]

            # check the length difference
            if len(task_train_smiles) != len(task_train_smiles_raw):
                print(f"    Warning: {len(task_train_smiles_raw) - len(task_train_smiles)} illegal SMILES removed from training set.")
            if len(task_valid_smiles) != len(task_valid_smiles_raw):
                print(f"    Warning: {len(task_valid_smiles_raw) - len(task_valid_smiles)} illegal SMILES removed from validation set.")



            if not task_train_smiles:
                 print(f"    Warning: No training SMILES found for task '{task_name}'.")
            else:
                print(f"    Loaded {len(task_train_smiles)} train, {len(task_valid_smiles)} valid SMILES.")
                all_train_smiles.extend(task_train_smiles)
                all_valid_smiles.extend(task_valid_smiles)

        except Exception as e:
            print(f"    Error loading data for task '{task_name}': {e}. Skipping this task.")
            continue

    # --- Combine and Deduplicate ---
    if not all_train_smiles:
        raise ValueError("No training data loaded successfully for any specified task. Please check task names and data paths.")

    print("-" * 30)
    print(f"Total SMILES loaded before deduplication: Train={len(all_train_smiles)}, Valid={len(all_valid_smiles)}")

    # Remove duplicates
    all_train_smiles = sorted(list(set(all_train_smiles)))
    all_valid_smiles = sorted(list(set(all_valid_smiles)))
    print(f"Total unique SMILES after deduplication: Train={len(all_train_smiles)}, Valid={len(all_valid_smiles)}")

    # Ensure validation set doesn't overlap with training set
    train_smiles_set = set(all_train_smiles)
    original_valid_count = len(all_valid_smiles)
    all_valid_smiles = [s for s in all_valid_smiles if s not in train_smiles_set]
    removed_valid_count = original_valid_count - len(all_valid_smiles)
    if removed_valid_count > 0:
        print(f"Removed {removed_valid_count} validation SMILES that were present in the training set.")
    print(f"Final dataset sizes: Train={len(all_train_smiles)}, Valid={len(all_valid_smiles)}")
    print("-" * 30)

    if not all_train_smiles:
        raise ValueError("Training set is empty after processing.")
    if not all_valid_smiles:
        print("Warning: Validation set is empty after removing training set overlap.")
        # Optionally create an empty dataloader or handle this case later
        valid_dataloader = None # Set to None if empty
    else:
        valid_dataset = SmilesDataset(all_valid_smiles)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)


    train_dataset = SmilesDataset(all_train_smiles)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)


    return train_dataloader, valid_dataloader


def get_llm_text_and_projected_features(
    smiles_batch,
    projected_features,  # Tensor of shape [batch_size, hidden_dim]
    llm,
    tokenizer,
    layer_k_arg,
    device,
    target_dtype,
    max_length=128
):
    """
    Computes LLM hidden states for both text-based input (SMILES) and projected input (dense vectors).

    Returns:
        text_last_token_features: detached (optional), not requiring grad
        proj_token_features: requires_grad=True (for projector training)
    """

    batch_size = len(smiles_batch)
    hidden_dim = projected_features.shape[-1]

    llm.eval()  # Keep deterministic, but avoid torch.no_grad() for projector side

    # === First: TEXT side — can optionally disable gradient if you want to save memory ===
    with torch.no_grad():
        inputs = tokenizer(
            smiles_batch,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_length,
        ).to(device)

        text_outputs = llm(**inputs, output_hidden_states=True)
        text_hidden_states_tuple = text_outputs.hidden_states

        num_transformer_layers = len(text_hidden_states_tuple) - 1
        actual_layer_index = layer_k_arg if 1 <= layer_k_arg <= num_transformer_layers else -1
        text_hidden_layer = text_hidden_states_tuple[actual_layer_index]
        text_last_token_features = text_hidden_layer[:, -1, :].to(dtype=target_dtype)

    # === Second: PROJECTED VECTOR side — no torch.no_grad(), retain gradient flow ===
    inputs_embeds = projected_features.unsqueeze(1).to(device)  # (B, 1, D)

    # DO NOT USE torch.no_grad() here!
    proj_outputs = llm(inputs_embeds=inputs_embeds, output_hidden_states=True)
    proj_hidden_states_tuple = proj_outputs.hidden_states
    proj_hidden_layer = proj_hidden_states_tuple[actual_layer_index]
    proj_token_features = proj_hidden_layer[:, -1, :].to(dtype=target_dtype)

    # test gradient flow
    print(f"Gradient flow test: {proj_token_features.requires_grad}, {text_last_token_features.requires_grad}")

    return text_last_token_features, proj_token_features

# --- Model Initialization (Using the version with fallback for projector_output_dim) ---
def get_models(args, device, dtype):
    # UniMol
    # Note: Using a single cache path here based on args.base_data_path.
    # If datasets have separate caches, this might need adjustment or disable caching.
    mol_fm = unimol_clf(avoid_loading_model=False, rep_cache_path=f'{args.base_data_path}/smile_to_rep_store.pkl')
    if hasattr(mol_fm, 'model') and mol_fm.model is not None and isinstance(mol_fm.model, nn.Module):
        mol_fm.model.to(device).eval()
    elif isinstance(mol_fm, nn.Module):
        mol_fm.to(device).eval()
    else:
        try: mol_fm.to(device)
        except: pass
        try: mol_fm.eval()
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
    # Consider adding device_map="auto" if facing OOM issues with large LLMs
    # if you use device_map, remove the .to(device) call below
    llm_model = AutoModelForCausalLM.from_pretrained(args.llm_model_name, **llm_model_kwargs)
    if "device_map" not in llm_model_kwargs: llm_model = llm_model.to(device)
    llm_model.eval()

    projector_input_dim = args.unimol_feature_dim
    # Infer or fallback projector_output_dim
    projector_output_dim = None
    try:
        projector_output_dim = llm_model.config.hidden_size
        if projector_output_dim is None: raise AttributeError("llm_model.config.hidden_size is None.")
        print(f"Inferred LLM hidden_size for projector output: {projector_output_dim}")
    except AttributeError as e:
        default_fallback_dim = 4096
        print(f"Warning: Could not infer LLM hidden_size from config. Error: {e}. Using fallback: {default_fallback_dim}")
        projector_output_dim = default_fallback_dim
    except Exception as e_general:
        default_fallback_dim = 4096
        print(f"Warning: Unexpected error inferring hidden_size. Error: {e_general}. Using fallback: {default_fallback_dim}")
        projector_output_dim = default_fallback_dim

    if args.projector_type == "MLP_Nonlinear":

        projector = MLP_Nonlinear(
            ninput=projector_input_dim,
            noutput=projector_output_dim,
        ).to(device).to(dtype) # Ensure projector is also on the correct device and dtype

    elif args.projector_type == "MLP_Linear":

        projector = MLP_Linear(
            ninput=projector_input_dim,
            noutput=projector_output_dim,
            layers=args.projector_arch,
        ).to(device).to(dtype) # Ensure projector is also on the correct device and dtype

    return mol_fm, llm_tokenizer, llm_model, projector, projector_output_dim


# --- Helper Functions (Remain the same, including updated get_llm_text_features) ---
@torch.no_grad() # UniMol feature extraction should not require gradients
def get_unimol_features(smiles_batch, mol_fm, device, target_dtype, skip_errors=False):
    fm_reps_list = []
    valid_smiles_in_batch = []
    
    original_unimol_dtype = None # To store original dtype if we temporarily change it
    if hasattr(mol_fm, 'model') and mol_fm.model is not None and hasattr(mol_fm.model, 'dtype'):
        original_unimol_dtype = mol_fm.model.dtype
        # mol_fm.model = mol_fm.model.to(target_dtype) # This might be too disruptive if UniMol has its own dtype needs
    elif isinstance(mol_fm, nn.Module) and hasattr(mol_fm, 'dtype'):
        original_unimol_dtype = mol_fm.dtype
        # mol_fm = mol_fm.to(target_dtype)
    # print(f"Original UniMol dtype: {original_unimol_dtype}")


    for smiles_str in smiles_batch:
        try:
            # UniMol's get_unimol_rep_tensor might return float32 by default
            unimol_rep = mol_fm.get_unimol_rep_tensor(smiles_str) # Expected: [1, unimol_dim]
            if unimol_rep is None or unimol_rep.nelement() == 0:
                raise ValueError("UniMol returned empty tensor.")
            fm_reps_list.append(unimol_rep.to(device, dtype=target_dtype))
            valid_smiles_in_batch.append(smiles_str)
        except Exception as e:
            if skip_errors:
                print(f"Skipping SMILES '{smiles_str}' due to UniMol error: {e}")
                continue
            else:
                raise e # Re-raise if not skipping

    # Restore original unimol dtype if changed
    # if original_unimol_dtype is not None:
    #     if hasattr(mol_fm, 'model') and mol_fm.model is not None:
    #         mol_fm.model = mol_fm.model.to(original_unimol_dtype)
    #     elif isinstance(mol_fm, nn.Module):
    #         mol_fm = mol_fm.to(original_unimol_dtype)


    if not fm_reps_list:
        return None, []
    
    fm_reps_batch = torch.cat(fm_reps_list, dim=0)

    # print(f"Extracted {len(fm_reps_list)} valid UniMol features from batch of size {len(smiles_batch)}.")
    # print(f"UniMol feature tensor shape: {fm_reps_batch.shape}, dtype: {fm_reps_batch.dtype}")

    # print(f"Valid SMILES number in batch: {len(valid_smiles_in_batch)}")
    return fm_reps_batch, valid_smiles_in_batch

@torch.no_grad()
def get_llm_text_features(smiles_batch, llm, tokenizer, layer_k_arg, device, target_dtype, max_length=128):
    """
    Extracts text features (last token representation) from a specific layer of the LLM.
    Assumes tokenizer.padding_side is 'left'.
    """
    llm.eval() # Ensure model is in eval mode


    # Tokenize the batch. tokenizer.padding_side = "left" should be set on the tokenizer object itself.
    inputs = tokenizer(
        smiles_batch,
        return_tensors="pt",
        padding="longest", # Pad to max_length on the left
        truncation=True,
        max_length=max_length,
    ).to(device)

    outputs = llm(**inputs, output_hidden_states=True)
    
    hidden_states_tuple = outputs.hidden_states # Tuple: (input_embeds, layer1_hidden, ..., final_layer_hidden)
    
    num_transformer_layers = len(hidden_states_tuple) - 1 # Exclude input embeddings layer

    actual_layer_index = -1 # Default to last layer output (final transformer block output)
    if layer_k_arg == -1 or layer_k_arg == num_transformer_layers:
        actual_layer_index = -1 # Accesses the last element of the tuple
    elif layer_k_arg > 0 and layer_k_arg <= num_transformer_layers:
        # hidden_states_tuple[0] is input embeddings
        # hidden_states_tuple[1] is output of 1st transformer block
        # So, for 1-indexed layer_k_arg, the index in tuple is layer_k_arg
        actual_layer_index = layer_k_arg 
    else:
        raise ValueError(
            f"Invalid llm_target_layer_k: {layer_k_arg}. "
            f"LLM has {num_transformer_layers} transformer blocks (available range for layer_k_arg: 1 to {num_transformer_layers}, or -1 for last). "
        )
    
    target_layer_hidden_states = hidden_states_tuple[actual_layer_index] # Shape: (batch_size, seq_len, hidden_dim)
    
    # With left padding, the last *actual* token's representation is at the -1 sequence index.
    # For example, if inputs are:
    # [PAD, PAD, T1, T2, T3]
    # [PAD, T4, T5, T6, T7]
    # hidden_states[:, -1, :] will correctly get the features for T3 and T7 respectively.
    last_token_features = target_layer_hidden_states[:, -1, :] # Shape: (batch_size, hidden_dim)
    
    return last_token_features.to(dtype=target_dtype)


def info_nce_loss(features_a, features_b, temperature):
    # features_a: [batch_size, dim] (e.g., projected_features)
    # features_b: [batch_size, dim] (e.g., text_features, batch_inner corresponding are positive)
    
    # Normalize features to prevent scaling issues and ensure cosine similarity
    features_a = F.normalize(features_a, p=2, dim=1)
    features_b = F.normalize(features_b, p=2, dim=1)

    # Calculate cosine similarity matrix (logits)
    # features_a @ features_b.T
    logits = torch.matmul(features_a, features_b.T) / temperature
    
    # Labels: positive pairs are on the diagonal
    labels = torch.arange(logits.shape[0], device=logits.device)
    
    # Calculate cross-entropy loss for A->B and B->A (symmetric loss)
    loss_a_b = F.cross_entropy(logits, labels)
    loss_b_a = F.cross_entropy(logits.T, labels) # Transpose logits for B->A
    
    loss = (loss_a_b + loss_b_a) / 2
    return loss

# --- Main Training Function (Modified for multi-task naming and validation frequency) ---
def train_projector(args):
    set_random_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.use_bfloat16 and torch.cuda.is_bf16_supported(): dtype = torch.bfloat16
    elif torch.cuda.is_available(): dtype = torch.float16
    else: dtype = torch.float32
    print(f"Using dtype: {dtype}")

    validation_frequency = args.validation_frequency
    save_every_n_epochs = args.save_every_n_epochs

    # --- WANDB Initialization ---
    tasks_str = "_".join(sorted(args.task_names)) # Create a string representation of tasks
    if args.projector_type == "MLP_Nonlinear":
        run_name_prefix = f"{args.base_run_name}_Nonlinear_lr_{args.learning_rate}_batch_{args.batch_size}_epochs{args.num_epochs}_{tasks_str}"
    elif args.projector_type == "MLP_Linear":
        run_name_prefix = f"{args.base_run_name}_Linear_arch_{args.projector_arch}_lr_{args.learning_rate}_batch_{args.batch_size}_epochs{args.num_epochs}_{tasks_str}"
    
    run_name = args.wandb_run_name if args.wandb_run_name else f"{run_name_prefix}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, config=args)

    # --- DataLoaders ---
    train_dataloader, valid_dataloader = get_dataloaders(args)
    if train_dataloader is None: # Check added in get_dataloaders
        print("Exiting: No training data loaded.")
        return

    # Print samples after potential combination and deduplication
    if train_dataloader.dataset: print(f"Sample training SMILES: {train_dataloader.dataset.smiles_list[:5]}")
    if valid_dataloader and valid_dataloader.dataset: print(f"Sample validation SMILES: {valid_dataloader.dataset.smiles_list[:5]}")

    # --- Models ---
    mol_fm, llm_tokenizer, llm_model, projector, projector_output_dim_actual = get_models(args, device, dtype)
    args.projector_output_dim = projector_output_dim_actual # Update args with actual dim
    wandb.config.update({"projector_output_dim_actual": projector_output_dim_actual}, allow_val_change=True)
    print(f"Projector output dimension set to: {projector_output_dim_actual}")
    print(f"Projector architecture: {args.projector_arch if args.projector_arch else 'Linear'}")


    optimizer = optim.AdamW(projector.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_valid_loss = float('inf')

    args.output_dir = os.path.join(args.output_dir, tasks_str)

    # add run_name_prefix
    args.output_dir = os.path.join(args.output_dir, run_name_prefix)

    # add timestamp
    args.output_dir = os.path.join(args.output_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n--- Starting Training ---")
    for epoch in range(args.num_epochs):
        projector.train()
        total_train_loss = 0
        train_batches = 0
        optimizer.zero_grad()

        progress_bar_train = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")
        for batch_idx, smiles_batch in progress_bar_train:
            if not smiles_batch: continue

            # --- Forward Pass ---
            fm_reps_batch, valid_smiles_in_batch_fm = get_unimol_features(smiles_batch, mol_fm, device, dtype, args.skip_unimol_errors)
            if fm_reps_batch is None: continue # Skip if unimol failed for all

            # Determine projector's expected dtype (from its first parameter)
            proj_param_dtype = next(projector.parameters()).dtype
            projected_features = projector(fm_reps_batch.to(proj_param_dtype))

            # target_text_features = get_llm_text_features(valid_smiles_in_batch_fm, llm_model, llm_tokenizer,
            #                                              args.llm_target_layer_k, device, proj_param_dtype, # Match LLM output dtype to projector
            #                                              args.llm_max_length)
            target_text_features, projected_features = get_llm_text_and_projected_features(
                smiles_batch=valid_smiles_in_batch_fm,
                projected_features=projected_features,
                llm=llm_model,
                tokenizer=llm_tokenizer,
                layer_k_arg=args.llm_target_layer_k,
                device=device,
                target_dtype=projected_features.dtype,
                max_length=args.llm_max_length,
            )

            if projected_features.shape[0] != target_text_features.shape[0] or projected_features.shape[0] == 0: continue

            loss = info_nce_loss(projected_features, target_text_features, args.contrastive_temperature_tau)
            loss = loss / args.gradient_accumulation_steps

            # --- Backward Pass & Optimization ---
            loss.backward()
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            # --- Logging ---
            current_loss = loss.item() * args.gradient_accumulation_steps
            total_train_loss += current_loss
            train_batches += 1
            progress_bar_train.set_postfix({"loss": f"{current_loss:.4f}"})
            if (batch_idx + 1) % (args.gradient_accumulation_steps * 20) == 0 : # Log batch loss less often
                 wandb.log({"train/batch_loss": current_loss})

        avg_train_loss = total_train_loss / train_batches if train_batches > 0 else 0
        wandb.log({"train/epoch_loss": avg_train_loss, "epoch": epoch + 1})
        print(f"Epoch [{epoch+1}/{args.num_epochs}] Training Loss: {avg_train_loss:.4f}")

        # --- Validation Step ---
        # Validate based on frequency or if it's the last epoch
        perform_validation = (valid_dataloader is not None) and \
                             (validation_frequency > 0 and ((epoch + 1) % validation_frequency == 0 or (epoch + 1) == args.num_epochs))

        if perform_validation:
            projector.eval()
            total_valid_loss = 0
            valid_batches = 0
            progress_bar_valid = tqdm(valid_dataloader, total=len(valid_dataloader), desc=f"Epoch {epoch+1}/{args.num_epochs} [Valid]", leave=False)
            with torch.no_grad():
                for smiles_batch_valid in progress_bar_valid:
                    if not smiles_batch_valid: continue

                    fm_reps_valid, valid_smiles_fm_val = get_unimol_features(smiles_batch_valid, mol_fm, device, dtype, args.skip_unimol_errors)
                    if fm_reps_valid is None: continue

                    proj_param_dtype = next(projector.parameters()).dtype
                    projected_valid = projector(fm_reps_valid.to(proj_param_dtype))

                    # target_text_valid = get_llm_text_features(valid_smiles_fm_val, llm_model, llm_tokenizer,
                    #                                           args.llm_target_layer_k, device, proj_param_dtype,
                    #                                           args.llm_max_length)
                    

                    target_text_valid, projected_valid = get_llm_text_and_projected_features(
                            smiles_batch=valid_smiles_fm_val,
                            projected_features=projected_valid,
                            llm=llm_model,
                            tokenizer=llm_tokenizer,
                            layer_k_arg=args.llm_target_layer_k,
                            device=device,
                            target_dtype=proj_param_dtype,
                            max_length=args.llm_max_length,
                        )

                    if projected_valid.shape[0] != target_text_valid.shape[0] or projected_valid.shape[0] == 0 : continue

                    valid_loss = info_nce_loss(projected_valid, target_text_valid, args.contrastive_temperature_tau)
                    total_valid_loss += valid_loss.item()
                    valid_batches += 1
                    progress_bar_valid.set_postfix({"loss": f"{valid_loss.item():.4f}"})

            avg_valid_loss = total_valid_loss / valid_batches if valid_batches > 0 else float('inf')
            wandb.log({"valid/epoch_loss": avg_valid_loss, "epoch": epoch + 1})
            print(f"Epoch [{epoch+1}/{args.num_epochs}] Validation Loss: {avg_valid_loss:.4f}")

            # --- Save Best Model ---
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                model_path = os.path.join(args.output_dir, f"{tasks_str}_best_projector.pth")
                torch.save(projector.state_dict(), model_path)
                if os.path.exists(model_path): wandb.save(model_path) # Save best model to WANDB if file exists
                print(f"Saved best model to {model_path} with validation loss {best_valid_loss:.4f}")

        # --- Save Latest Model Periodically ---
        if save_every_n_epochs > 0 and ((epoch + 1) % save_every_n_epochs == 0 or (epoch + 1) == args.num_epochs):
            latest_model_path = os.path.join(args.output_dir, f"{tasks_str}_latest_projector_epoch{epoch+1}.pth")
            torch.save(projector.state_dict(), latest_model_path)
            print(f"Saved latest model checkpoint to {latest_model_path}")

        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- Final Model Saving ---
    final_model_path = os.path.join(args.output_dir, f"{tasks_str}_final_projector.pth")
    torch.save(projector.state_dict(), final_model_path)
    if os.path.exists(final_model_path): wandb.save(final_model_path)
    print(f"Training complete. Final projector saved to {final_model_path}")
    wandb.finish()
    print("--- Training Finished ---")


if __name__ == "__main__":
    args = parse_args()
    print("--- Starting Contrastive Projector Training ---")
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("-" * 30)
    train_projector(args)