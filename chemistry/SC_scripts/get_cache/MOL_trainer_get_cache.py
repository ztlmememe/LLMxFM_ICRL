import argparse
import datetime
import gc
import os
# set wandb to offline mode
# os.environ["WANDB_MODE"] = "offline"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import logging
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("absl").setLevel(logging.WARNING)
import transformers
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
from datasets import load_dataset
from chemistry.utils.unimol_utils import *
from chemistry.utils.models import MLP_Linear,MLP_Nonlinear,Vicl_Linear
from chemistry.utils.utils import *
import evaluate

# --- Configuration & Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train a projector for SMILES to text generation using LPM-24 dataset.")
    parser.add_argument("--base_data_path", type=str, default="./../datasets/", help="Base path for datasets and representation caches.")
    parser.add_argument("--output_dir", type=str, default="./lpm24_projector_output", help="Directory to save trained models and logs.")

    parser.add_argument("--llm_model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Hugging Face model name for LLM.")
    parser.add_argument("--llm_max_length", type=int, default=256, help="Max sequence length for LLM tokenizer.")

    parser.add_argument("--unimol_feature_dim", type=int, default=512, help="Feature dimension of UniMol output.")
    parser.add_argument("--projector_arch", type=str, default="1024-2048", help="Architecture of the MLP projector (e.g., '512-1024').")
    parser.add_argument("--projector_type", type=str, default="MLP_Linear",
                        help="Type of projector architecture (MLP_Nonlinear or MLP_Linear).")

    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer.")
    parser.add_argument("--max_samples", type=int, default=10000, help="Maximum number of samples to use for training.")

    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader.")
    parser.add_argument("--use_bfloat16", action='store_true', help="Use bfloat16 for training if available.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Number of steps to accumulate gradients.")

    # base_run_name
    parser.add_argument("--base_run_name", type=str, default="lpm24_projector", help="Base name for the run.")
    parser.add_argument("--wandb_project", type=str, default="lpm24_projector_training", help="WANDB project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WANDB entity (username or team).")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WANDB run name.")
    parser.add_argument("--skip_unimol_errors", action='store_true', help="Skip SMILES that cause errors in UniMol processing.")

    args = parser.parse_args()


    return args

# --- Dataset Classes ---
class LPM24Dataset(Dataset):
    def __init__(self, smiles_list, captions_list):
        # Filter out invalid entries
        valid_indices = []
        for i, (smiles, caption) in enumerate(zip(smiles_list, captions_list)):
            if isinstance(smiles, str) and isinstance(caption, str) and smiles.strip() and caption.strip():
                valid_indices.append(i)

        self.smiles_list = [smiles_list[i] for i in valid_indices]
        self.captions_list = [captions_list[i] for i in valid_indices]

        print(f"Created dataset with {len(self.smiles_list)} valid SMILES-caption pairs")

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        return self.smiles_list[idx], self.captions_list[idx]

def load_lpm24_data(max_samples=None):
    """Load LPM-24 dataset from Hugging Face"""
    print("Loading LPM-24 dataset...")
    try:
        from datasets import logging
        logging.set_verbosity_error() # Reduce verbosity
        dataset = load_dataset("language-plus-molecules/LPM-24_train",
                               data_files={'split_train': 'data/split_train-00000-of-00001.parquet',
                                           'split_valid': 'data/split_valid-00000-of-00001.parquet',
                                           'train': 'data/train-00000-of-00001.parquet'
                                           })

        train_data = dataset['split_train']
        valid_data = dataset['split_valid']

        # if max_samples and max_samples > 0:
        #     train_data = train_data.select(range(min(max_samples, len(train_data))))
        #     valid_data = valid_data.select(range(min(max_samples // 5, len(valid_data)))) # Use 1/5 for validation
        # print(f"Loaded {len(train_data)} training samples and {len(valid_data)} validation samples.")

        train_smiles = train_data['molecule']
        train_captions = train_data['caption']
        valid_smiles = valid_data['molecule']
        valid_captions = valid_data['caption']

        print(f"Train: {len(train_smiles)}, Valid: {len(valid_smiles)}")

        return train_smiles, train_captions, valid_smiles, valid_captions

    except Exception as e:
        print(f"Error loading LPM-24 dataset: {e}")
        raise

def get_dataloaders(args):
    """Create data loaders for training and validation"""
    train_smiles, train_captions, valid_smiles, valid_captions = load_lpm24_data(args.max_samples)

    illegal_path = "/mnt/ssd/ztl/LLMxFM/chemistry/datasets/illegal_smiles.txt"
    if os.path.exists(illegal_path):
        with open(illegal_path, 'r') as f:
            illegal_smiles = set(line.strip() for line in f if line.strip() and not line.startswith("#"))
    else:
        illegal_smiles = set()

    def filter_data(smiles, captions):
        filtered_smiles, filtered_captions = [], []
        for s, c in zip(smiles, captions):
            if s not in illegal_smiles:
                filtered_smiles.append(s)
                filtered_captions.append(c)
            else:
                print(f"Skipping illegal SMILES: {s}")
        return filtered_smiles, filtered_captions

    train_smiles, train_captions = filter_data(train_smiles, train_captions)
    valid_smiles, valid_captions = filter_data(valid_smiles, valid_captions)

    train_dataset = LPM24Dataset(train_smiles, train_captions)
    valid_dataset = LPM24Dataset(valid_smiles, valid_captions)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda batch: list(zip(*batch))  # Custom collate function
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda batch: list(zip(*batch))
    )

    return train_dataloader, valid_dataloader


def get_models(args, device, dtype):
    """Initialize all models"""
    # UniMol
    mol_fm = unimol_clf(avoid_loading_model=False, rep_cache_path=f'{args.base_data_path}/smile_to_rep_store.pkl')
    if hasattr(mol_fm, 'model') and mol_fm.model is not None:
        mol_fm.model.to(device)
        mol_fm.model.eval()
    elif isinstance(mol_fm, torch.nn.Module):
        mol_fm.to(device)
        mol_fm.eval()


    return mol_fm

# --- Helper Functions ---
# @torch.no_grad()
# def get_unimol_features(smiles_batch, mol_fm, device, target_dtype, skip_errors=False):
#     """Extract UniMol features for a batch of SMILES"""
#     fm_reps_list = []
#     valid_smiles_in_batch = []

#     for smiles_str in smiles_batch:
#         try:
#             unimol_rep = mol_fm.get_unimol_rep_tensor(smiles_str)
#             if unimol_rep is None or unimol_rep.nelement() == 0:
#                 raise ValueError("UniMol returned empty tensor.")
#             fm_reps_list.append(unimol_rep.to(device, dtype=target_dtype))
#             valid_smiles_in_batch.append(smiles_str)
#         except Exception as e:
#             if skip_errors:
#                 print(f"Skipping SMILES '{smiles_str}' due to UniMol error: {e}")
#                 continue
#             else:
#                 raise e

#     if not fm_reps_list:
#         return None, []

#     fm_reps_batch = torch.cat(fm_reps_list, dim=0)
#     return fm_reps_batch, valid_smiles_in_batch

@torch.no_grad()
def get_unimol_features(smiles_batch, mol_fm, device, target_dtype, skip_errors=False):
    """Batch extract UniMol features for a list of SMILES"""
    try:
        unimol_reps = mol_fm.get_unimol_rep_tensor(smiles_batch)
        if unimol_reps is None or unimol_reps.nelement() == 0:
            raise ValueError("UniMol returned empty tensor.")

        return unimol_reps.to(device, dtype=target_dtype), smiles_batch

    except Exception as e:

        illegal_path = "/mnt/ssd/ztl/LLMxFM/chemistry/datasets/illegal_smiles.txt"
        # write the invalid smiles to a file
        import re
        err_msg = str(e)
        extracted_smiles = None
        match = re.search(r'SMILES rule is illegal:\s*(.*)', err_msg)
        extracted_smiles = match.group(1).strip()


        print(f"Skipping SMILES due to error: {extracted_smiles} → {err_msg}")
        try:
            with open(illegal_path, "a") as f:
                f.write(extracted_smiles + "\n")
        except Exception as log_error:
            print(f"Failed to log illegal SMILES: {log_error}")


        if skip_errors:
            print(f"Skipping entire batch due to UniMol error: {e}")
            return None, []
        else:
            raise e


# --- Main Training Function ---
def train_projector(args):
    set_random_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine dtype
    if args.use_bfloat16 and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        print("Using bfloat16.")
    elif torch.cuda.is_available():
        dtype = torch.float16
        print("Using float16 on GPU.")
    else:
        dtype = torch.float32
        print("Using float32 on CPU.")


    # Data loaders
    train_dataloader, valid_dataloader = get_dataloaders(args)

    # Models
    mol_fm = get_models(args, device, dtype)



    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

    for batch_idx, (smiles_batch, captions_batch) in progress_bar:
        if not smiles_batch or not captions_batch:
            continue

        # Get UniMol features
        fm_reps_batch, valid_smiles = get_unimol_features(
            smiles_batch, mol_fm, device, dtype, args.skip_unimol_errors
        )

    progress_bar = tqdm(enumerate(valid_dataloader), total=len(valid_dataloader))

    for batch_idx, (smiles_batch, captions_batch) in progress_bar:
        if not smiles_batch or not captions_batch:
            continue

        # Get UniMol features
        fm_reps_batch, valid_smiles = get_unimol_features(
            smiles_batch, mol_fm, device, dtype, args.skip_unimol_errors
        )
     

if __name__ == "__main__":
    args = parse_args()
    print("Starting LPM-24 projector training with arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")


    train_projector(args)