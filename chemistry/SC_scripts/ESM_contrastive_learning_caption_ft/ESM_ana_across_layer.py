import gc
import random
from tqdm import tqdm
import datetime
import torch
# Assuming these utility files exist and are in the PYTHONPATH
# If not, you might need to provide stubs or actual implementations

from chemistry.utils.esm_utils import *
from chemistry.utils.unimol_utils import *
from chemistry.utils.utils import *
from chemistry.utils.LMM_api_utils import *
from chemistry.utils.models import MLP_Mapper, MLP_Mapper_withoutbn,MLP_Linear
from transformers import BertModel, BertTokenizer
import traceback
from transformers import EsmModel, AutoTokenizer
import os
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


class ProteinDataset(Dataset):
    """
    Dataset class for handling protein sequences.
    """
    def __init__(self, protein_sequences):
        self.protein_sequences = protein_sequences

    def __len__(self):
        return len(self.protein_sequences)

    def __getitem__(self, idx):
        seq = self.protein_sequences[idx]
        # Ensure the sequence is a string
        if isinstance(seq, (np.ndarray, list, tuple)) and len(seq) == 1:
            seq = seq[0]
        return str(seq)

def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Extract all-layer protein representations with ESM and analyze similarity (No Caching)")
    parser.add_argument('--base_data_path', type=str, default='./datasets/',
                        help='Path to base dataset folder (e.g., ./datasets/)')
    parser.add_argument('--tasks', type=str, nargs='+', default=["Stability", "Fluorescence"],
                        help='List of dataset tasks to process (e.g., Stability Fluorescence)')
    parser.add_argument('--model_name', type=str, default="facebook/esm2_t30_150M_UR50D",
                        help='Name of the ESM model to use from Hugging Face.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for ESM model inference.')
    parser.add_argument('--plot_path', type=str, default='layer_similarity_no_cache.png',
                        help='Path to save the similarity plot.')
    return parser.parse_args()

import re
def preprocess_protein_sequence(seq: str):

    seq = re.sub(r"[UZOB]", "X", seq)

    return ' '.join(list(seq))

def extract_all_layer_representations_prot(model, tokenizer, protein_seqs, batch_size=64, device="cuda"):
    """
    Extracts representations from all layers of the ESM model for given protein sequences.

    Args:
        model (EsmModel): The pre-trained ESM model.
        tokenizer (AutoTokenizer): The ESM tokenizer.
        protein_seqs (list): A list of protein sequences (strings).
        batch_size (int): The batch size for processing.
        device (str): The device to run inference on ("cuda" or "cpu").

    Returns:
        dict: A dictionary mapping each sequence to a list of NumPy arrays,
              where each array is the mean-pooled representation from one layer.
    """
    if not protein_seqs:
        print("No protein sequences provided for extraction.")
        return {}

    dataset = ProteinDataset(protein_seqs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_reps_map = {}

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting all-layer protein reps"):
            sequences = list(batch)
            print(f"Processing batch of {len(sequences)} sequences...")
            sequences = [preprocess_protein_sequence(seq) for seq in sequences]  # Preprocess sequences
            
            inputs = tokenizer(
                sequences,
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,
                # truncation=True,
                # max_length=1022 # ESM standard limit, keep some buffer
            ).to(device)

            
            outputs = model(**inputs, output_hidden_states=True)
            print(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]) )

            hidden_states = outputs.hidden_states # Tuple of (batch, seq_len, dim)
            # attention_mask = inputs['attention_mask']

            # pooled_reps_batch_layers = []
            # for layer_hidden_state in hidden_states:
            #     # Masked mean pooling
            #     mask_expanded = attention_mask.unsqueeze(-1).expand(layer_hidden_state.size()).float()
            #     sum_hidden = torch.sum(layer_hidden_state * mask_expanded, 1)
            #     sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9) # Avoid division by zero
            #     pooled_output = sum_hidden / sum_mask
            #     pooled_reps_batch_layers.append(pooled_output.cpu()) # Move to CPU

            pooled_reps_batch_layers = []
            for layer_hidden_state in hidden_states:
                cls_output = layer_hidden_state[:, 0, :]  # shape: (batch_size, hidden_size)
                pooled_reps_batch_layers.append(cls_output.cpu())  # Move to CPU


            # Store reps: Transpose [layer][batch] -> [batch][layer] and map
            batch_size_current = len(sequences)
            num_layers = len(hidden_states)
            for i in range(batch_size_current):
                seq = sequences[i]
                # Store as list of NumPy arrays
                all_reps_map[seq] = [pooled_reps_batch_layers[l][i].numpy() for l in range(num_layers)]

    return all_reps_map


def extract_all_layer_representations(model, tokenizer, protein_seqs, batch_size=64, device="cuda"):
    """
    Extracts representations from all layers of the ESM model for given protein sequences.

    Args:
        model (EsmModel): The pre-trained ESM model.
        tokenizer (AutoTokenizer): The ESM tokenizer.
        protein_seqs (list): A list of protein sequences (strings).
        batch_size (int): The batch size for processing.
        device (str): The device to run inference on ("cuda" or "cpu").

    Returns:
        dict: A dictionary mapping each sequence to a list of NumPy arrays,
              where each array is the mean-pooled representation from one layer.
    """
    if not protein_seqs:
        print("No protein sequences provided for extraction.")
        return {}

    dataset = ProteinDataset(protein_seqs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_reps_map = {}

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting all-layer protein reps"):
            sequences = list(batch)
            print(f"Processing batch of {len(sequences)} sequences...")
            print(f"Example sequence: {sequences[0][:50]}...")  # Print first 50 chars of the first sequence
            inputs = tokenizer(
                sequences,
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,
                # truncation=True,
                # max_length=1022 # ESM standard limit, keep some buffer
            ).to(device)

            

            outputs = model(**inputs, output_hidden_states=True)
            print(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]) )

            hidden_states = outputs.hidden_states # Tuple of (batch, seq_len, dim)
            # attention_mask = inputs['attention_mask']

            # pooled_reps_batch_layers = []
            # for layer_hidden_state in hidden_states:
            #     # Masked mean pooling
            #     mask_expanded = attention_mask.unsqueeze(-1).expand(layer_hidden_state.size()).float()
            #     sum_hidden = torch.sum(layer_hidden_state * mask_expanded, 1)
            #     sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9) # Avoid division by zero
            #     pooled_output = sum_hidden / sum_mask
            #     pooled_reps_batch_layers.append(pooled_output.cpu()) # Move to CPU

            pooled_reps_batch_layers = []
            for layer_hidden_state in hidden_states:
                cls_output = layer_hidden_state[:, 0, :]  # shape: (batch_size, hidden_size)
                pooled_reps_batch_layers.append(cls_output.cpu())  # Move to CPU


            # Store reps: Transpose [layer][batch] -> [batch][layer] and map
            batch_size_current = len(sequences)
            num_layers = len(hidden_states)
            for i in range(batch_size_current):
                seq = sequences[i]
                # Store as list of NumPy arrays
                all_reps_map[seq] = [pooled_reps_batch_layers[l][i].numpy() for l in range(num_layers)]

    return all_reps_map

def calculate_and_visualize_similarity(all_reps_map, output_path="layer_similarity.png"):
    """
    Calculates the average pairwise cosine similarity for each layer and visualizes it.

    Args:
        all_reps_map (dict): Dictionary mapping sequences to their all-layer representations.
        output_path (str): Path to save the output plot image.
    """
    if not all_reps_map:
        print("No representations to analyze.")
        return

    first_key = next(iter(all_reps_map))
    num_layers = len(all_reps_map[first_key])
    print(f"Number of layers found: {num_layers}")

    layer_data = [[] for _ in range(num_layers)]
    for seq, reps_list in all_reps_map.items():
        if len(reps_list) == num_layers:
            for l_idx, rep in enumerate(reps_list):
                layer_data[l_idx].append(rep)
        else:
            print(f"Warning: Sequence {seq[:20]}... has {len(reps_list)} layers, expected {num_layers}. Skipping.")

    avg_similarities = []
    print("\nCalculating average cosine similarity per layer...")
    for l_idx, reps in enumerate(layer_data):
        if len(reps) < 2:
            print(f"Layer {l_idx}: Has fewer than 2 sequences ({len(reps)}), cannot compute similarity. Appending NaN.")
            avg_similarities.append(np.nan)
            continue

        reps_array = np.array(reps)
        sim_matrix = cosine_similarity(reps_array)
        indices = np.triu_indices_from(sim_matrix, k=1)
        
        if len(indices[0]) > 0:
            avg_sim = np.mean(sim_matrix[indices])
            avg_sim = round(avg_sim, 4)
        else:
            avg_sim = np.nan

        avg_similarities.append(avg_sim)
        print(f"Layer {l_idx:02d}: Average Cosine Similarity = {avg_sim:.4f}")

    print(f"\nGenerating and saving plot to {output_path}...")
    plt.figure(figsize=(12, 7))
    plt.plot(range(num_layers), avg_similarities, marker='o', linestyle='-', color='b')
    plt.title(f'Average Internal Cosine Similarity per ESM Layer ({len(all_reps_map)} sequences)', fontsize=16)
    plt.xlabel('ESM Layer Index (0 = Embeddings)', fontsize=12)
    plt.ylabel('Average Pairwise Cosine Similarity', fontsize=12)
    plt.xticks(range(num_layers))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f"Similarity plot saved successfully to {output_path}")


def run(base_data_path, tasks_to_run, model_name, batch_size, plot_path):
    """
    Main execution function: Loads data, extracts representations, and runs analysis.
    (This version does NOT use caching).
    """
    torch.set_grad_enabled(False)
    set_random_seed(42)
    # change args.plot_path to args.plot_path + model_name
    model_short_name = model_name.split("/")[-1]
    args.plot_path = os.path.join(args.plot_path, f"{model_short_name}.png")
    print(f"Plot will be saved to: {args.plot_path}")
    plot_path = args.plot_path


    all_sequences = set()

    print("Loading sequences from specified tasks...")
    for task in tasks_to_run:
        print(f"--- Processing Task: {task} ---")
        train_inputs, train_y, valid_inputs, valid_y, test_inputs, test_y = None, None, None, None, None, None
        try:
            if task == "Stability":
                train_inputs, train_y, valid_inputs, valid_y, test_inputs, test_y = load_Stability(5000, 1000, base_data_path)
            elif task == "Fluorescence":
                train_inputs, train_y, valid_inputs, valid_y, test_inputs, test_y = load_Fluorescence(5000, 1000, base_data_path)
            elif task in ["Beta_Lactamase", "PPI_Affinity"]:
                 print(f"Warning: Loader for task '{task}' not implemented/used. Skipping.")
                 continue
            else:
                 print(f"Warning: Unknown task '{task}'. Skipping.")
                 continue

            for inputs in [train_inputs, valid_inputs, test_inputs]:
                if inputs is not None:
                    if isinstance(inputs, pd.DataFrame): inputs = inputs.to_numpy()
                    for seq_item in inputs:
                        seq = seq_item
                        if isinstance(seq, (np.ndarray, list, tuple)) and len(seq) == 1:
                            seq = seq[0]
                        all_sequences.add(str(seq))

        except Exception as e:
            print(f"Error loading or processing task {task}: {e}")
            traceback.print_exc()

    print(f"\nFound {len(all_sequences)} unique sequences across all specified tasks.")

    if not all_sequences:
        print("No sequences found. Exiting.")
        return

    # --- Extraction (No Caching) ---
    all_reps_map = {}
    sequences_to_compute = list(all_sequences)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading ESM model '{model_name}' on device '{device}'...")
    try:
        
        if "esm" in model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = EsmModel.from_pretrained(model_name).to(device)

            all_reps_map = extract_all_layer_representations(
                model=model,
                tokenizer=tokenizer,
                protein_seqs=sequences_to_compute,
                batch_size=batch_size,
                device=device
            )
            
        elif "prot_bert" in model_name:


            tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
            model = BertModel.from_pretrained("Rostlab/prot_bert").to(device)

            # using re to replace all UZOB with X in sequences_to_compute
            # sequences_to_compute = [re.sub(r"[UZOB]", "X", seq) for seq in sequences_to_compute]

            all_reps_map = extract_all_layer_representations_prot(
                model=model,
                tokenizer=tokenizer,
                protein_seqs=sequences_to_compute,
                batch_size=batch_size,
                device=device
            )

            # sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
        print("Representation extraction complete.")
        
        

        # Clean up GPU memory
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"An error occurred during model loading or extraction: {e}")
        traceback.print_exc()
        return # Stop if extraction fails

    # --- Analysis and Visualization ---
    calculate_and_visualize_similarity(all_reps_map, output_path=plot_path)

    print("\n--- Script Finished ---")

if __name__ == "__main__":
    args = parse_args()
    run(
        base_data_path=args.base_data_path,
        tasks_to_run=args.tasks,
        model_name=args.model_name,
        batch_size=args.batch_size,
        plot_path=args.plot_path
    )



