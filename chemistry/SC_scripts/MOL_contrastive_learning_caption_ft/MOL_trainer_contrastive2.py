import argparse
import datetime
import gc
import os
# set wandb offline
# os.environ["WANDB_MODE"] = "offline"
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
from chemistry.utils.models import MLP_Linear
from chemistry.utils.utils import *

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
    parser = argparse.ArgumentParser(description="Train a projector for SMILES representations using contrastive learning.")
    parser.add_argument("--task_name", type=str, default="ESOL", help="Name of the dataset/task to use (e.g., ESOL, Caco2_wang).")
    parser.add_argument("--base_data_path", type=str, default="./../datasets/", help="Base path for datasets and representation caches.")
    parser.add_argument("--output_dir", type=str, default="./contrastive_output", help="Directory to save trained models and logs.")

    parser.add_argument("--llm_model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Hugging Face model name for LLM.")
    parser.add_argument("--llm_target_layer_k", type=int, default=-1, help="Target layer in LLM to extract features from (e.g., 12, -1 for last). Transformer block index, 1-indexed if positive.")
    parser.add_argument("--llm_max_length", type=int, default=128, help="Max sequence length for LLM tokenizer.")

    parser.add_argument("--unimol_feature_dim", type=int, default=512, help="Feature dimension of UniMol output. Official UniMol is 512 for unimol_mof.pt, may vary with other checkpoints. For `drug_mol_bert_ জায়ান্ট_pre_트레인드.j2`, it is 640")
    parser.add_argument("--projector_arch", type=str, default="1024-2048", help="Architecture of the MLP projector (e.g., '512-1024', or empty for linear).")
    parser.add_argument("--projector_dropout", type=float, default=0.1, help="Dropout rate for projector MLP.")


    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--contrastive_temperature_tau", type=float, default=0.07, help="Temperature parameter for InfoNCE loss.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer.")


    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader.") # Set to 0 for easier debugging on GPU
    parser.add_argument("--use_bfloat16", action='store_true', help="Use bfloat16 for LLM and projector if available.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients before an optimizer step.")

    parser.add_argument("--wandb_project", type=str, default="contrastive_smiles_projector", help="WANDB project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WANDB entity (username or team).") # User should fill this
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WANDB run name (optional, defaults to auto-generated).")
    parser.add_argument("--skip_unimol_errors", action='store_true', help="Skip SMILES that cause errors in UniMol processing instead of crashing.")
    parser.add_argument("--base_run_name", type=str, default="contrastive_projector", help="Base name for WANDB run.")

    return parser.parse_args()

# --- Dataset Loading & Preparation ---
class SmilesDataset(Dataset):
    def __init__(self, smiles_list):
        # Filter out any non-string or empty SMILES if they somehow get here
        self.smiles_list = [s for s in smiles_list if isinstance(s, str) and s.strip()]
    def __len__(self):
        return len(self.smiles_list)
    def __getitem__(self, idx):
        return self.smiles_list[idx]

def get_dataloaders(args):
    # This section is adapted from MOL_trainer_pca_OT_icl.py's data loading logic
    train_inputs_raw, valid_inputs_raw = None, None

    if args.task_name == "Caco2_wang" or args.task_name == "Solubility_AqSolDB" or \
       args.task_name == "Half_Life_Obach" or args.task_name == "Clearance_Hepatocyte_AZ" or \
       args.task_name == "HydrationFreeEnergy_FreeSolv" or args.task_name == "Lipophilicity_AstraZeneca":
        data = data_split_loading_ADME(dataset_name=args.task_name, split_type='random', seed=args.random_seed, split_fracs=[0.7, 0.1, 0.2])
        train_inputs_raw, valid_inputs_raw = data["train"]["SMILES"], data["valid"]["SMILES"]
    elif args.task_name == "ESOL":
        train_inputs_raw, _, valid_inputs_raw, _, _, _ = load_ESOL()
    elif args.task_name == "LD50_Zhu":
        data = data_split_loading_Tox(dataset_name=args.task_name, split_type='random', seed=args.random_seed, split_fracs=[0.7, 0.1, 0.2])
        train_inputs_raw, valid_inputs_raw = data["train"]["SMILES"], data["valid"]["SMILES"]
    else:
        raise ValueError(f"Unsupported task_name: {args.task_name}")

    # Ensure inputs are flat lists of SMILES strings
    def flatten_smiles(inputs_raw):
        if inputs_raw is None: return []
        if isinstance(inputs_raw, (pd.Series, pd.DataFrame)):
            inputs_raw = inputs_raw.to_numpy()
        if isinstance(inputs_raw, np.ndarray):
            inputs_raw = inputs_raw.flatten().tolist()
        
        processed_smiles = []
        for item in inputs_raw:
            if isinstance(item, (list, np.ndarray)):
                processed_smiles.append(str(item[0])) # Take the first element if it's a list/array
            elif isinstance(item, str):
                processed_smiles.append(item)
            # else: skip
        return [s for s in processed_smiles if s.strip()]


    train_smiles = flatten_smiles(train_inputs_raw)
    valid_smiles = flatten_smiles(valid_inputs_raw)

    print(f"Loaded {len(train_smiles)} training SMILES and {len(valid_smiles)} validation SMILES for task {args.task_name}.")

    # Optional: Pre-filter SMILES with UniMol if preprocess_smiles_data or similar is available and desired
    # This could be done here or handled robustly in the batch processing loop
    if args.task_name == "Solubility_AqSolDB" or args.task_name == "HydrationFreeEnergy_FreeSolv":
        train_inputs, train_y, valid_inputs, valid_y, test_inputs, test_y = preprocess_smiles_data(mol_fm=None, 
            train_inputs = train_inputs, train_y = train_y, valid_inputs = valid_inputs, 
            valid_y = valid_y, test_inputs =test_inputs, test_y = test_y, base_data_path = args.base_data_path,skip_check=True)

    train_dataset = SmilesDataset(train_smiles)
    valid_dataset = SmilesDataset(valid_smiles)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return train_dataloader, valid_dataloader


# --- Model Initialization ---
def get_models(args, device, dtype):
    # UniMol (Molecule Foundation Model)
    # The unimol_clf is assumed to handle its own model loading.
    # We need to ensure its output dimension matches `args.unimol_feature_dim`.
    # The official UniMol output dimension might be 512 or 640 depending on the checkpoint.
    # User should verify `args.unimol_feature_dim`.
    mol_fm = unimol_clf(avoid_loading_model=False, rep_cache_path=f'{args.base_data_path}/smile_to_rep_store.pkl')
    if hasattr(mol_fm, 'model') and mol_fm.model is not None: # If it's a PyTorch model
         mol_fm.model.to(device) # Send UniMol's internal model to device if possible
         mol_fm.model.eval()
    elif isinstance(mol_fm, torch.nn.Module): # If unimol_clf itself is a Module
        mol_fm.to(device)
        mol_fm.eval()
    else: # Fallback for placeholder or other types
        try: mol_fm.to(device) # Try calling .to()
        except: pass
        try: mol_fm.eval() # Try calling .eval()
        except: pass


    # LLM (for extracting target text features)
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_model_name)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token # Common practice
        llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id

    llm_tokenizer.padding_side = "left"
    
    llm_model_kwargs = {"torch_dtype": dtype} if args.use_bfloat16 and torch.cuda.is_bf16_supported() else {}
    if not args.use_bfloat16: # If not bfloat16, try float16 if available on GPU
        if dtype == torch.float16 and torch.cuda.is_available():
             llm_model_kwargs = {"torch_dtype": torch.float16}


    llm_model = AutoModelForCausalLM.from_pretrained(args.llm_model_name, **llm_model_kwargs).to(device)
    llm_model.eval() # Set to evaluation mode, no training for LLM

    projector_input_dim = args.unimol_feature_dim
    # Infer LLM embedding dimension for projector output
    # This can vary, e.g., Llama-3-8B is 4096. We get it from the loaded LLM.
    try:
        projector_output_dim = llm_model.config.hidden_size
    except AttributeError:
        print("Could not infer LLM hidden_size. Please set --projector_output_dim manually if needed.")
        # Fallback, user might need to specify this via args if LLM config is unusual
        projector_output_dim = args.unimol_feature_dim * 2 # A guess

    projector = MLP_Linear(
        ninput=projector_input_dim,
        noutput=projector_output_dim,
        layers=args.projector_arch,
    ).to(device).to(dtype) # Ensure projector is also on the correct device and dtype

    return mol_fm, llm_tokenizer, llm_model, projector, projector_output_dim


# --- Helper Functions ---
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
    print(f"Original UniMol dtype: {original_unimol_dtype}")


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

    print(f"Extracted {len(fm_reps_list)} valid UniMol features from batch of size {len(smiles_batch)}.")
    print(f"UniMol feature tensor shape: {fm_reps_batch.shape}, dtype: {fm_reps_batch.dtype}")

    print(f"Valid SMILES number in batch: {len(valid_smiles_in_batch)}")
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

# --- Main Training Function ---
def train_projector(args):
    set_random_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine dtype for models
    if args.use_bfloat16 and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        print("Using bfloat16.")
    elif torch.cuda.is_available(): # Default to float16 on GPU if not bfloat16
        dtype = torch.float16
        print("Using float16 on GPU.")
    else: # CPU uses float32
        dtype = torch.float32
        print("Using float32 on CPU.")

    # --- WANDB Initialization ---
    run_name = args.wandb_run_name if args.wandb_run_name else f"{args.base_run_name}_{args.task_name}_lr_{args.learning_rate}_batch_{args.batch_size}_epochs{args.num_epochs}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=args
    )

    # --- DataLoaders ---
    train_dataloader, valid_dataloader = get_dataloaders(args)

    # print samples
    print(f"Sample training SMILES: {train_dataloader.dataset.smiles_list[:5]}")
    print(f"Sample validation SMILES: {valid_dataloader.dataset.smiles_list[:5]}")

    # --- Models ---
    mol_fm, llm_tokenizer, llm_model, projector, projector_output_dim_actual = get_models(args, device, dtype)
    # Update args if projector_output_dim was inferred
    args.projector_output_dim = projector_output_dim_actual
    # wandb.config.update({"projector_output_dim_actual": projector_output_dim_actual}, allow_val_change=True)
    print(f"Projector output dimension: {projector_output_dim_actual}")
    print(f"Projector architecture: {args.projector_arch}")

    # --- Optimizer ---
    optimizer = optim.AdamW(projector.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # --- Training Loop ---
    best_valid_loss = float('inf')
    # add task name
    args.output_dir = os.path.join(args.output_dir, args.task_name)
    # add timestamp
    args.output_dir = os.path.join(args.output_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(args.output_dir, exist_ok=True)


    validation_frequency = 5

    for epoch in range(args.num_epochs):
        projector.train()
        total_train_loss = 0
        train_batches = 0
        
        optimizer.zero_grad() # Initialize gradients for accumulation

        progress_bar_train = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")
        for batch_idx, smiles_batch in progress_bar_train:
            if not smiles_batch: continue # Skip if batch is empty after filtering in Dataset

            # 1. Get UniMol molecular representations
            fm_reps_batch, valid_smiles_in_batch_fm = get_unimol_features(smiles_batch, mol_fm, device, dtype, args.skip_unimol_errors)
            
            if fm_reps_batch is None or fm_reps_batch.nelement() == 0:
                print(f"Skipping batch {batch_idx} due to no valid UniMol features.")
                continue

            # 2. Project UniMol features
            # Ensure projector input is float32 if projector is float32, or cast if projector expects specific type
            projected_features = projector(fm_reps_batch.to(projector.fc.weight.dtype if hasattr(projector, 'fc') else next(projector.parameters()).dtype))


            # 3. Get LLM text features (target) for the SMILES that were successfully processed by UniMol
            # target_text_features = get_llm_text_features(valid_smiles_in_batch_fm, llm_model, llm_tokenizer, 
            #                                              args.llm_target_layer_k, device, projected_features.dtype, # Match dtype with projected features
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


            if projected_features.shape[0] != target_text_features.shape[0] or projected_features.shape[0] == 0:
                print(f"Skipping batch {batch_idx} due to mismatch or zero samples after LLM processing. Projected: {projected_features.shape}, Target: {target_text_features.shape}")
                continue
            
            # 4. Calculate contrastive loss
            # Loss calculation might be more stable in float32
            loss = info_nce_loss(projected_features.float(), target_text_features.float(), args.contrastive_temperature_tau)
            loss = loss / args.gradient_accumulation_steps # Normalize loss for accumulation

            # 5. Backpropagation
            loss.backward()
            
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=1.0) # Optional gradient clipping
                optimizer.step()
                optimizer.zero_grad() # Reset gradients for the next accumulation cycle

            total_train_loss += loss.item() * args.gradient_accumulation_steps # Un-normalize for logging
            train_batches += 1
            progress_bar_train.set_postfix({"loss": loss.item() * args.gradient_accumulation_steps})
            wandb.log({"train/batch_loss": loss.item() * args.gradient_accumulation_steps})

        avg_train_loss = total_train_loss / train_batches if train_batches > 0 else 0
        wandb.log({"train/epoch_loss": avg_train_loss, "epoch": epoch})
        print(f"Epoch [{epoch+1}/{args.num_epochs}] Training Loss: {avg_train_loss:.4f}")

        # --- Validation Loop ---
        if validation_frequency > 0 and ((epoch + 1) % validation_frequency == 0 or (epoch + 1) == args.num_epochs):
            projector.eval()
            total_valid_loss = 0
            valid_batches = 0
            progress_bar_valid = tqdm(valid_dataloader, total=len(valid_dataloader), desc=f"Epoch {epoch+1}/{args.num_epochs} [Valid]")
            with torch.no_grad():
                for smiles_batch_valid in progress_bar_valid:
                    if not smiles_batch_valid: continue

                    fm_reps_valid, valid_smiles_fm_val = get_unimol_features(smiles_batch_valid, mol_fm, device, dtype, args.skip_unimol_errors)
                    if fm_reps_valid is None or fm_reps_valid.nelement() == 0: continue

                    projected_valid = projector(fm_reps_valid.to(projector.fc.weight.dtype if hasattr(projector, 'fc') else next(projector.parameters()).dtype))
                    
                    # target_text_valid = get_llm_text_features(valid_smiles_fm_val, llm_model, llm_tokenizer,
                    #                                         args.llm_target_layer_k, device, projected_valid.dtype,
                    #                                         args.llm_max_length)
                    
                    target_text_valid, projected_valid = get_llm_text_and_projected_features(
                            smiles_batch=valid_smiles_fm_val,
                            projected_features=projected_valid,
                            llm=llm_model,
                            tokenizer=llm_tokenizer,
                            layer_k_arg=args.llm_target_layer_k,
                            device=device,
                            target_dtype=projected_valid.dtype,
                            max_length=args.llm_max_length,
                        )
                    
                    if projected_valid.shape[0] != target_text_valid.shape[0] or projected_valid.shape[0] == 0 : continue

                    valid_loss = info_nce_loss(projected_valid.float(), target_text_valid.float(), args.contrastive_temperature_tau)
                    total_valid_loss += valid_loss.item()
                    valid_batches += 1
                    progress_bar_valid.set_postfix({"loss": valid_loss.item()})
            
            avg_valid_loss = total_valid_loss / valid_batches if valid_batches > 0 else 0
            wandb.log({"valid/epoch_loss": avg_valid_loss, "epoch": epoch})
            print(f"Epoch [{epoch+1}/{args.num_epochs}] Validation Loss: {avg_valid_loss:.4f}")

            # --- Save Model Checkpoint ---
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                model_path = os.path.join(args.output_dir, f"{args.task_name}_best_projector.pth")
                torch.save(projector.state_dict(), model_path)
                wandb.save(model_path) # Save best model to WANDB
                print(f"Saved best model to {model_path} with validation loss {best_valid_loss:.4f}")
            
            # Save latest model periodically
            if (epoch + 1) % 5 == 0 or (epoch + 1) == args.num_epochs :
                latest_model_path = os.path.join(args.output_dir, f"{args.task_name}_latest_projector_epoch{epoch+1}.pth")
                torch.save(projector.state_dict(), latest_model_path)
                # wandb.save(latest_model_path) # Optionally save all latest models

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Final Model Saving ---
    final_model_path = os.path.join(args.output_dir, f"{args.task_name}_final_projector.pth")
    torch.save(projector.state_dict(), final_model_path)
    wandb.save(final_model_path)
    print(f"Training complete. Final projector saved to {final_model_path}")
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    print("Starting contrastive projector training with arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    train_projector(args)