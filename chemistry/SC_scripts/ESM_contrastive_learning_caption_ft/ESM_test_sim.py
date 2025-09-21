import argparse
import datetime
import gc
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
# Updated imports for protein
from chemistry.utils.esm_utils import *
from chemistry.utils.models import MLP_Linear, MLP_Nonlinear, Vicl_Linear
from chemistry.utils.utils import *

# --- Task Type Dictionary (Updated for Protein) ---
task_type_dict = {
    "Fluorescence": "PROTEIN",
    "Stability": "PROTEIN",
    "Beta_Lactamase": "PROTEIN",
    "PPI_Affinity": "PROTEIN",
}

# --- Configuration & Argument Parsing (Updated for Protein) ---
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained projector for proteins and visualize metrics across LLM layers.")
    parser.add_argument("--task_names", type=str, nargs='+', default=["Stability"],
                        help="List of protein dataset/task names for evaluation (e.g., Stability Fluorescence).")
    parser.add_argument("--base_data_path", type=str, default="./../datasets/",
                        help="Base path for datasets and representation caches.")
    parser.add_argument("--projector_path", type=str, required=True, # Made required for evaluation
                        help="Path to the trained protein projector model (.pth file).")
    parser.add_argument("--plot_path", type=str, default="protein_llm_layer_metrics.png", # Updated plot path
                        help="Path to save the output plot.")
    parser.add_argument("--llm_model_name", type=str, default="meta-llama/Meta-Llama-3-8B",
                        help="Hugging Face model name for LLM.")
    parser.add_argument("--llm_max_length", type=int, default=1024, # Increased for proteins
                        help="Max sequence length for LLM tokenizer.")
    # ESM arguments
    parser.add_argument("--esm_model_name", type=str, default="facebook/esm2_t33_650M_UR50D",
                        help="ESM model name for protein feature extraction.")
    parser.add_argument("--esm_feature_dim", type=int, default=1280, # Default for esm2_t33_650M
                        help="Feature dimension of ESM output.")
    # Projector arguments
    parser.add_argument("--projector_arch", type=str, default="1024-2048",
                        help="Architecture of the MLP projector (e.g., '1280-2048').")
    parser.add_argument("--projector_type", type=str, default="MLP_Linear",
                        help="Type of projector architecture (MLP_Linear, MLP_Nonlinear, Vicl_Linear).")
    # Other arguments
    parser.add_argument("--batch_size", type=int, default=4, # Reduced for potentially large models/seqs
                        help="Batch size for evaluation.")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_bfloat16", action='store_true')
    parser.add_argument("--skip_esm_errors", action='store_true', # Renamed
                        help="Skip sequences causing ESM errors.")
    # Placeholder args for protein loaders if needed
    parser.add_argument("--num_trains", type=int, default=10000)
    parser.add_argument("--num_tests", type=int, default=2000)

    args, unknown = parser.parse_known_args()
    return args

# --- Dataset Loading & Preparation (Updated for Protein) ---
class SequenceDataset(Dataset): # Renamed
    def __init__(self, sequence_list):
        self.sequence_list = [s for s in sequence_list if isinstance(s, str) and s.strip()]
    def __len__(self):
        return len(self.sequence_list)
    def __getitem__(self, idx):
        return self.sequence_list[idx]

def get_dataloaders(args):
    all_valid_sequences = [] # Only need validation/test set for evaluation

    def flatten_sequences(inputs_raw): # Renamed
        if inputs_raw is None: return []
        if isinstance(inputs_raw, (pd.Series, pd.DataFrame)): inputs_raw = inputs_raw.to_numpy()
        if isinstance(inputs_raw, np.ndarray): inputs_raw = inputs_raw.flatten().tolist()
        processed_sequences = [str(item[0]) if isinstance(item, (list, np.ndarray)) else str(item) for item in inputs_raw]
        return [s for s in processed_sequences if s.strip()]

    print(f"Loading data for tasks: {args.task_names}")
    for task_name in args.task_names:
        print(f"--> Loading task: {task_name}")
        train_inputs_raw, valid_inputs_raw = None, None
        args.base_data_path = os.path.join(os.path.dirname(__file__), "./../datasets/")

        try:
            # Load protein data - using validation set mainly
            if task_name == "Stability":
                _, _, valid_inputs_raw, _, _, _ = load_Stability(args.num_trains, args.num_tests, args.base_data_path)
            elif task_name == "Fluorescence":
                _, _, valid_inputs_raw, _, _, _ = load_Fluorescence(args.num_trains, args.num_tests, args.base_data_path)
            elif task_name == "Beta_Lactamase":
                _, _, valid_inputs_raw, _, _, _ = load_Beta_Lactamase(args.num_trains, args.num_tests, args.base_data_path)
            elif task_name == "PPI_Affinity":
                _, _, valid_inputs_raw, _, _, _ = load_ppi_affinity(args.num_trains, args.num_tests, args.base_data_path)
            else:
                print(f"    Warning: Unsupported task_name '{task_name}'. Skipping.")
                continue

            task_valid_sequences = flatten_sequences(valid_inputs_raw)
            # Optional: Add illegal sequence check if needed

            if not task_valid_sequences:
                 print(f"    Warning: No validation sequences found for task '{task_name}'.")
            else:
                print(f"    Loaded {len(task_valid_sequences)} valid sequences.")
                all_valid_sequences.extend(task_valid_sequences)

        except Exception as e:
            print(f"    Error loading data for task '{task_name}': {e}. Skipping this task.")
            continue

    if not all_valid_sequences:
        raise ValueError("No validation data loaded successfully. Cannot perform evaluation.")

    print("-" * 30)
    all_valid_sequences = sorted(list(set(all_valid_sequences)))
    print(f"Total unique validation sequences: {len(all_valid_sequences)}")
    print("-" * 30)

    if not all_valid_sequences:
        raise ValueError("Validation set is empty after processing. Cannot perform evaluation.")

    valid_dataset = SequenceDataset(all_valid_sequences)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return valid_dataloader

# --- Model Initialization (Updated for ESM) ---
def get_models(args, device, dtype):
    # ESM Model
    protein_fm = esm_model(
        esm_model_name=args.esm_model_name,
        avoid_loading_model=False,
        rep_cache_path=f'{args.base_data_path}/aaseq_to_rep_store.pkl'
    )
    if hasattr(protein_fm, 'model') and protein_fm.model is not None:
        protein_fm.model.to(device).eval()
    elif isinstance(protein_fm, nn.Module):
        protein_fm.to(device).eval()

    # LLM
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_model_name)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    llm_tokenizer.padding_side = "left"
    llm_model_kwargs = {"low_cpu_mem_usage": True}
    if args.use_bfloat16 and torch.cuda.is_bf16_supported(): llm_model_kwargs["torch_dtype"] = torch.bfloat16
    elif dtype == torch.float16 and torch.cuda.is_available(): llm_model_kwargs["torch_dtype"] = torch.float16
    llm_model = AutoModelForCausalLM.from_pretrained(args.llm_model_name, **llm_model_kwargs)
    if "device_map" not in llm_model_kwargs: llm_model = llm_model.to(device)
    llm_model.eval()

    # Projector
    projector_input_dim = args.esm_feature_dim # Use ESM dim
    projector_output_dim = llm_model.config.hidden_size

    if args.projector_type == "MLP_Nonlinear":
        projector = MLP_Nonlinear(ninput=projector_input_dim, noutput=projector_output_dim).to(device).to(dtype)
    elif args.projector_type == "MLP_Linear":
        projector = MLP_Linear(ninput=projector_input_dim, noutput=projector_output_dim, layers=args.projector_arch).to(device).to(dtype)
    elif args.projector_type == "Vicl_Linear":
        projector = Vicl_Linear(in_features=projector_input_dim, out_features=projector_output_dim).to(device).to(dtype)
    else:
        raise ValueError(f"Unknown projector_type: {args.projector_type}")

    if not os.path.exists(args.projector_path):
        raise FileNotFoundError(f"Projector model file not found at: {args.projector_path}")
    print(f"Loading trained projector from: {args.projector_path}")
    projector.load_state_dict(torch.load(args.projector_path, map_location=device))
    projector.eval()

    num_layers = llm_model.config.num_hidden_layers
    return protein_fm, llm_tokenizer, llm_model, projector, num_layers

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
            else: raise e
    if not fm_reps_list: return None, []
    fm_reps_batch = torch.cat(fm_reps_list, dim=0)
    return fm_reps_batch, valid_sequences_in_batch

@torch.no_grad()
def get_llm_text_features_all_layers(sequence_batch, llm, tokenizer, device, target_dtype, max_length=1024):
    """Extracts last token features from ALL hidden layers."""
    llm.eval()
    inputs = tokenizer(
        sequence_batch, # Use protein sequences
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_length,
    ).to(device)
    outputs = llm(**inputs, output_hidden_states=True)
    hidden_states_tuple = outputs.hidden_states

    all_layer_features = []
    for layer_hidden_states in hidden_states_tuple[1:]: # Skip embedding layer
        last_token_features = layer_hidden_states[:, -1, :]
        all_layer_features.append(last_token_features.to(dtype=target_dtype))

    return all_layer_features

@torch.no_grad()
def get_llm_projected_features_all_layers(projected_features, llm, device, target_dtype):
    """Feeds projected features as inputs_embeds and gets all layer outputs."""
    llm.eval()
    inputs_embeds = projected_features.unsqueeze(1).to(device)
    if inputs_embeds.dtype != llm.dtype: # Ensure dtype matches LLM
        inputs_embeds = inputs_embeds.to(llm.dtype)

    outputs = llm(inputs_embeds=inputs_embeds, output_hidden_states=True)
    hidden_states_tuple = outputs.hidden_states

    all_layer_features = []
    for layer_idx, layer_hidden_states in enumerate(hidden_states_tuple[1:]): # Skip input embeds
        # Shape is (batch_size, 1, hidden_size), so index 0 or -1 is fine
        token_features = layer_hidden_states[:, 0, :]
        # print(f"Proj Layer {layer_idx+1} shape: {layer_hidden_states.shape}") # Debug
        all_layer_features.append(token_features.to(dtype=target_dtype))

    return all_layer_features

# --- Evaluation Metrics Calculation (Remains the same) ---
def calculate_metrics(features_a, features_b):
    if features_a.shape != features_b.shape:
        raise ValueError(f"Feature dimensions must match! Got {features_a.shape} and {features_b.shape}")
    n = features_a.shape[0]
    if n <= 1: return {"mean_cosine_similarity": float('nan'), "top_1_accuracy": float('nan'), "mean_rank": float('nan'), "count": n}

    features_a_norm = F.normalize(features_a, p=2, dim=1)
    features_b_norm = F.normalize(features_b, p=2, dim=1)
    similarity_matrix = torch.matmul(features_a_norm, features_b_norm.T)
    mean_cos_sim = torch.diag(similarity_matrix).mean().item()

    sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    ranks = [(sorted_indices[i] == i).nonzero(as_tuple=True)[0].item() + 1 for i in range(n)]
    top_1_acc = sum(1 for r in ranks if r == 1) / n
    mean_rank = np.mean(ranks)

    return {"mean_cosine_similarity": mean_cos_sim, "top_1_accuracy": top_1_acc, "mean_rank": mean_rank, "count": n}

# --- Plotting Function (Remains mostly the same, maybe update title) ---
def plot_metrics_vs_layer(results, output_path):
    if not results: print("No results to plot."); return

    layers = [r['layer'] for r in results]
    sims = [r['mean_cosine_similarity'] for r in results]
    accs = [r['top_1_accuracy'] for r in results]
    ranks = [r['mean_rank'] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    # fig.suptitle('Protein Projector Evaluation Metrics vs. LLM Layer', fontsize=16, y=1.05) # Updated title

    def set_dynamic_ylim_strict(ax, data):
        valid_data = [d for d in data if d is not None and not np.isnan(d)]
        if not valid_data: ax.set_ylim(bottom=0, top=1); return
        min_val, max_val = np.min(valid_data), np.max(valid_data)
        padding = (max_val - min_val) * 0.1 if (max_val - min_val) > 0 else 0.1
        ax.set_ylim(bottom=min_val - padding, top=max_val + padding)

    axes[0].plot(layers, sims, 'o-', color='tab:red', label='Mean Cosine Similarity')
    axes[0].set_ylabel('Similarity'); axes[0].set_title('Mean Cosine Similarity ↑')
    axes[0].legend(loc='best'); axes[0].grid(True, linestyle='--', alpha=0.6)
    set_dynamic_ylim_strict(axes[0], sims); axes[0].set_xlabel('LLM Layer Number')

    axes[1].plot(layers, accs, 's--', color='tab:orange', label='Top-1 Accuracy')
    axes[1].set_ylabel('Accuracy'); axes[1].set_title('Top-1 Accuracy ↑')
    axes[1].legend(loc='best'); axes[1].grid(True, linestyle='--', alpha=0.6)
    set_dynamic_ylim_strict(axes[1], accs); axes[1].set_xlabel('LLM Layer Number')

    sample_count = results[0].get('count', 0)
    random_rank_guess = sample_count / 2.0 if sample_count > 0 else 0
    axes[2].plot(layers, ranks, '^-', color='tab:blue', label='Mean Rank')
    if random_rank_guess > 0:
        axes[2].axhline(y=random_rank_guess, color='purple', linestyle='--', alpha=0.7, label=f'Random Guess (N/2 = {random_rank_guess:.1f})')
    axes[2].set_ylabel('Rank'); axes[2].set_title('Mean Rank ↓')
    axes[2].legend(loc='best'); axes[2].grid(True, linestyle='--', alpha=0.6)
    set_dynamic_ylim_strict(axes[2], ranks); axes[2].set_xlabel('LLM Layer Number')

    for ax in axes:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=min(10, len(layers))))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nMetrics Plot saved to {output_path}")
    plt.close()

# --- Internal Similarity Functions (Remains the same) ---
def calculate_internal_similarity(features):
    n = features.shape[0]
    if n <= 1: return float('nan')
    features_norm = F.normalize(features, p=2, dim=1)
    similarity_matrix = torch.matmul(features_norm, features_norm.T)
    internal_sim = (similarity_matrix.sum() - n) / (n * (n - 1))
    return internal_sim.item()

def plot_internal_similarity_vs_layer(layers, proj_sims, text_sims, output_path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(layers, proj_sims, 'o-', color='tab:purple', label='Projected Features Internal Sim')
    ax.plot(layers, text_sims, 's--', color='tab:green', label='Text Features Internal Sim')
    ax.set_xlabel('LLM Layer Number')
    ax.set_ylabel('Mean Internal Cosine Similarity')
    ax.set_title('Internal Feature Similarity vs. LLM Layer')
    ax.legend(loc='best'); ax.grid(True, linestyle='--', alpha=0.6)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=min(10, len(layers))))
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Internal Similarity Plot saved to {output_path}")
    plt.close()

# --- Main Evaluation Function (Updated for Protein) ---
def evaluate_projector(args):
    set_random_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.use_bfloat16 and torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32)
    print(f"Using device: {device}, dtype: {dtype}")

    valid_dataloader = get_dataloaders(args)
    protein_fm, llm_tokenizer, llm_model, projector, num_layers = get_models(args, device, dtype)

    print(f"\n--- Starting Feature Extraction ({num_layers} LLM layers) ---")
    all_llm_text_features_cpu = [[] for _ in range(num_layers)]
    all_llm_proj_features_cpu = [[] for _ in range(num_layers)]
    all_valid_sequences_count = 0

    progress_bar = tqdm(valid_dataloader, desc="Extracting Features")
    with torch.no_grad():
        for sequence_batch in progress_bar:
            fm_reps_batch, valid_sequences = get_esm_features(sequence_batch, protein_fm, device, dtype, args.skip_esm_errors)
            if fm_reps_batch is None or not valid_sequences: continue

            llm_text_features = get_llm_text_features_all_layers(valid_sequences, llm_model, llm_tokenizer, device, dtype, args.llm_max_length)
            projected_features = projector(fm_reps_batch)
            llm_proj_features = get_llm_projected_features_all_layers(projected_features, llm_model, device, dtype)

            if len(llm_text_features) != num_layers or len(llm_proj_features) != num_layers:
                 print(f"Warning: Layer count mismatch. Expected {num_layers}, got Text={len(llm_text_features)}, Proj={len(llm_proj_features)}. Skipping batch.")
                 continue

            for i in range(num_layers):
                all_llm_text_features_cpu[i].append(llm_text_features[i].cpu())
                all_llm_proj_features_cpu[i].append(llm_proj_features[i].cpu())
            all_valid_sequences_count += len(valid_sequences)
            gc.collect(); torch.cuda.empty_cache()

    if not all_llm_text_features_cpu[0]: print("No features extracted."); return

    llm_text_full = [torch.cat(l, dim=0).to(device) for l in all_llm_text_features_cpu]
    llm_proj_full = [torch.cat(l, dim=0).to(device) for l in all_llm_proj_features_cpu]
    del all_llm_text_features_cpu, all_llm_proj_features_cpu; gc.collect()
    print(f"\n--- Extracted {all_valid_sequences_count} valid feature pairs ---")

    print("\n--- Calculating Retrieval Metrics Across Layers ---")
    results = []
    layers_list = list(range(1, num_layers + 1))
    for k in tqdm(range(num_layers), desc="Calculating Retrieval Metrics"):
        metrics = calculate_metrics(llm_proj_full[k], llm_text_full[k])
        metrics['layer'] = layers_list[k]
        results.append(metrics)
        print(f"  Layer {metrics['layer']:02d}: Sim={metrics['mean_cosine_similarity']:.4f}, Acc={metrics['top_1_accuracy']:.4f}, Rank={metrics['mean_rank']:.2f}")

    print("\n--- Calculating Internal Similarities ---")
    proj_internal_sims = []
    text_internal_sims = []
    for k in tqdm(range(num_layers), desc="Calculating Internal Sim"):
        proj_sim = calculate_internal_similarity(llm_proj_full[k])
        text_sim = calculate_internal_similarity(llm_text_full[k])
        proj_internal_sims.append(proj_sim)
        text_internal_sims.append(text_sim)
        print(f"  Layer {layers_list[k]:02d}: Proj Sim={proj_sim:.4f}, Text Sim={text_sim:.4f}")

    plot_metrics_vs_layer(results, args.plot_path)
    internal_plot_path = args.plot_path.replace(".png", "_internal_sim.png")
    plot_internal_similarity_vs_layer(layers_list, proj_internal_sims, text_internal_sims, internal_plot_path)

    print("\n--- Evaluation Finished ---")

# --- Main Execution Block ---
if __name__ == "__main__":
    args = parse_args()
    print("--- Starting Protein Projector Evaluation ---") # Updated title
    print("Arguments:")
    for k, v in vars(args).items(): print(f"  {k}: {v}")
    print("-" * 30)
    try:
        evaluate_projector(args)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()