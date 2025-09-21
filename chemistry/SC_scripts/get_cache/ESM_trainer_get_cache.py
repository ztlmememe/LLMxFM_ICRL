import gc
import random
from tqdm import tqdm
import datetime
import torch
from chemistry.utils.esm_utils import *
from chemistry.utils.unimol_utils import *
from chemistry.utils.utils import *
from chemistry.utils.LMM_api_utils import *
import traceback
from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM, BitsAndBytesConfig, EsmModel
import os
from huggingface_hub import login
import numpy as np
import pandas as pd
import json
import deepchem as dc
from sklearn.decomposition import PCA
import pickle
import joblib
from chemistry.utils.models import MLP_Mapper, MLP_Mapper_withoutbn,MLP_Linear
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch

class ProteinDataset(Dataset):
    def __init__(self, protein_sequences):
        self.protein_sequences = protein_sequences

    def __len__(self):
        return len(self.protein_sequences)

    def __getitem__(self, idx):
        seq = self.protein_sequences[idx]
        if isinstance(seq, (np.ndarray, list, tuple)) and len(seq) == 1:
            seq = seq[0]
        return seq


task_type_dict = {
    "BindingDB_Ki": "DTI",
    "BindingDB_IC50": "DTI",
    "KIBA": "DTI",
    "DAVIS": "DTI",

    "Fluorescence": "ESM",
    "Stability": "ESM",
    "Beta_Lactamase": "ESM",
    "PPI_Affinity": "ESM",

    "Caco2_wang":"MOL",
    "ESOL": "MOL",
    "Solubility_AqSolDB": "MOL",
    "Half_Life_Obach": "MOL",
}
import argparse
import joblib
import io # Required for CPU_Unpickler

# Define CPU_Unpickler if it's not imported (needed for cross-device loading)
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def save_cache_safely(cache_path, data_map):
    """
    Saves the provided dictionary to a pickle file using a temporary file 
    for atomicity and includes existing cache merging.
    """
    print(f"Attempting to save/update cache to {cache_path}...")
    
    final_map = {}
    
    # 1. Load existing cache (if any) and merge
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                # Use CPU_Unpickler for robustness if cache might be loaded on CPU
                existing_map = joblib.load(f)
            final_map.update(existing_map)
            print(f"Loaded {len(existing_map)} existing entries.")
        except Exception as e:
            print(f"Warning: Could not load or parse existing cache at {cache_path}: {e}")

    # 2. Add/Overwrite with new data (ensuring tensors are on CPU for pickling)
    for key, tensor in data_map.items():
        if isinstance(tensor, torch.Tensor):
            final_map[key] = tensor.cpu().numpy()
        else:
            final_map[key] = tensor

    print(f"Total entries to save: {len(final_map)}")

    # 3. Save safely

    temp_file_path = cache_path + ".tmp"
    try:
        joblib.dump(final_map, temp_file_path, compress=3)
        os.replace(temp_file_path, cache_path)
        print("Successfully saved aaseq_to_rep_store (joblib, compressed).")
    except Exception as e:
        print(f"Error saving aaseq_rep_map: {e}")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

            
def parse_args():
    parser = argparse.ArgumentParser(description="Extract protein representations with ESM")
    parser.add_argument('--base_data_path', type=str, required=True,
                        help='Path to base dataset folder (e.g., ./datasets/)')
    parser.add_argument('--task_name', type=str, choices=["Stability", "Fluorescence", "Beta_Lactamase", "PPI_Affinity"],
                        required=True, help='Name of the dataset task (e.g., Stability)')
    return parser.parse_args()

def extract_protein_representations_by_layer(protein_fm, protein_seqs, layer_idx=-1, batch_size=512, device="cuda", cache_path=""):
    """
    Extract protein representations from a specific layer's CLS token.
    
    """
    if not protein_seqs:
        print("No protein sequences provided for extraction.")
        return

    dataset = ProteinDataset(protein_seqs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = protein_fm.fm
    tokenizer = protein_fm.tokenizer
    device_to_use = model.device

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting CLS token from layer {layer_idx}"):
            sequences = list(batch)
            inputs = tokenizer(
                sequences,
                return_tensors="pt",
                add_special_tokens=True,
                padding=True,
            ).to(device_to_use)

            # print(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]) )

            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # List[Tensor], length = num_layers + 1 (embedding + N layers)

            # print(f"hidden_states length: {len(hidden_states)}")

            if not (-len(hidden_states) <= layer_idx < len(hidden_states)):
                raise ValueError(f"Invalid layer_idx {layer_idx}. Valid range: [{-len(hidden_states)}, {len(hidden_states)-1}]")

            selected_layer_output = hidden_states[layer_idx]  # shape: (batch_size, seq_len, hidden_dim)
            cls_outputs = selected_layer_output[:, 0, :]       # CLS token (position 0)

            for seq, rep in zip(sequences, cls_outputs):
                protein_fm.aaseq_rep_map[seq] = rep.cpu()

            print("Updated representations for batch of size:", len(sequences))
            save_cache_safely(cache_path, protein_fm.aaseq_rep_map)

    print("Done extracting representations from layer:", layer_idx)




def extract_protein_representations(protein_fm, protein_seqs, batch_size=512, device="cuda",cache_path=""):
    """
    Extract protein representations using the provided protein_fm model.
    """
    if not protein_seqs:
        print("no protein sequences provided for extraction.")
        return

    dataset = ProteinDataset(protein_seqs)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0) 

    print(f"begin extracting representations for {len(protein_seqs)} sequences with batch size {batch_size}...")
    
    model_to_use = protein_fm.fm
    tokenizer_to_use = protein_fm.tokenizer
    device_to_use = protein_fm.fm.device

    model_to_use.eval()
    with torch.no_grad():

        for batch in tqdm(dataloader, desc="Extracting protein representations"):
            # 1. Tokenize
            sequences = list(batch)
            protein_input_ids = tokenizer_to_use(
                sequences, 
                return_tensors="pt", 
                add_special_tokens=True,
                padding=True,
            ).to(device_to_use)

            


            protein_fm_outputs = model_to_use(**protein_input_ids)
            
            protein_seq_features = protein_fm_outputs.last_hidden_state 
            pooled_protein_features = protein_seq_features[:, 0, :] 

            for seq, rep in zip(sequences, pooled_protein_features):
                protein_fm.aaseq_rep_map[seq] = rep.cpu()

            print("Updated representations for batch of size:", len(sequences))
            save_cache_safely(cache_path, protein_fm.aaseq_rep_map)
    
    print("Done extracting representations.")

def run(
    base_data_path, task_name
):
    """
    Executes the test suite.
    """

    torch.set_grad_enabled(False)

    set_random_seed(42)

    all_sequences = set()
    available_tasks = ["Stability", "Fluorescence"] #, "Beta_Lactamase", "PPI_Affinity"]
    
    tasks_to_run = available_tasks

    for task in tasks_to_run:
        if task == "Stability": 
            train_inputs, train_y, valid_inputs, valid_y, test_inputs, test_y = load_Stability(-1, -1,base_data_path) 

            cache_path = f'/mnt/ssd/ztl/LLMxFM/chemistry/datasets/aaseq_to_rep_store_layer_6.pkl'
            layer_idx = 6
        elif task == "Fluorescence": 
            train_inputs, train_y, valid_inputs, valid_y, test_inputs, test_y = load_Fluorescence(-1, -1,base_data_path)  

            cache_path = f'/mnt/ssd/ztl/LLMxFM/chemistry/datasets/aaseq_to_rep_store_layer_16.pkl'
            layer_idx = 16

        if isinstance(train_inputs, pd.DataFrame):
            train_inputs = train_inputs.to_numpy()
            valid_inputs = valid_inputs.to_numpy()
            test_inputs = test_inputs.to_numpy()
        if isinstance(train_y, pd.DataFrame):
            train_y = train_y.to_numpy()
            valid_y = valid_y.to_numpy()
            test_y = test_y.to_numpy()
            
        print("task_name: ", task)

        combined_y = np.concatenate([valid_y, test_y], axis=0).squeeze()
        combined_inputs = np.concatenate([valid_inputs, test_inputs], axis=0)

        test_inputs, test_y = combined_inputs,combined_y

        print("train_inputs.shape: ", train_inputs.shape)
        print("train_y.shape: ", train_y.shape) 
        print("test_inputs.shape: ", test_inputs.shape)
        print("test_y.shape: ", test_y.shape)
        
        # Combine and add all sequences to the set
        for inputs in [train_inputs, test_inputs]:
            if inputs is not None:
                if isinstance(inputs, pd.DataFrame): inputs = inputs.to_numpy()
                for seq_item in inputs:
                    seq = seq_item
                    if isinstance(seq, (np.ndarray, list, tuple)) and len(seq) == 1:
                        seq = seq[0]
                    all_sequences.add(str(seq))


        print(f"Found {len(all_sequences)} unique sequences across all specified tasks.")

        # cache_path = f'/mnt/ssd/ztl/LLMxFM/chemistry/datasets/aaseq_to_rep_store.pkl'
        
        print(f"Using cache path: {cache_path}")
        # FM Preparation - Load with cache to get existing reps
        protein_fm = esm_model(esm_model_name="facebook/esm2_t30_150M_UR50D", 
                            avoid_loading_model=False,
                            rep_cache_path=cache_path) 

        existing_sequences = set(protein_fm.aaseq_rep_map.keys())
        sequences_to_compute = list(all_sequences - existing_sequences)

        print(f"Found {len(sequences_to_compute)} new sequences to compute.")

        if sequences_to_compute:

            extract_protein_representations_by_layer(
                protein_fm=protein_fm,
                protein_seqs=sequences_to_compute,
                layer_idx=layer_idx, 
                batch_size=512,
                device="cuda" if torch.cuda.is_available() else "cpu",
                cache_path=cache_path
            )
            print("Done layer specific extraction.")
            del protein_fm 
            gc.collect()


            cache_path = f'/mnt/ssd/ztl/LLMxFM/chemistry/datasets/aaseq_to_rep_store_cls.pkl'
        
            print(f"Using cache path: {cache_path}")
            # FM Preparation - Load with cache to get existing reps
            protein_fm = esm_model(esm_model_name="facebook/esm2_t30_150M_UR50D", 
                                avoid_loading_model=False,
                                rep_cache_path=cache_path) 
            
            extract_protein_representations(
                protein_fm=protein_fm,
                protein_seqs=sequences_to_compute,
                batch_size=512,
                device="cuda" if torch.cuda.is_available() else "cpu",
                cache_path=cache_path
            )
            print("Done all.")
            del protein_fm
            gc.collect()
        

        else:
            print("No new sequences to compute. Skipping extraction.")


if __name__ == "__main__":
    args = parse_args()
    run(args.base_data_path, args.task_name)


# python extract_esm.py --base_data_path /mnt/ssd/ztl/LLMxFM/chemistry/datasets --task_name Stability
# CUDA_VISIBLE_DEVICES=0 python -m chemistry.SC_scripts.ESM_trainer_get_cache --base_data_path /mnt/ssd/ztl/LLMxFM/chemistry/datasets --task_name Fluorescence
