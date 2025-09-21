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
    parser = argparse.ArgumentParser(description="Train a projector for SMILES to text generation using LPM-24 dataset with contrastive loss.")
    parser.add_argument("--base_data_path", type=str, default="./../datasets/", help="Base path for datasets and representation caches.")
    parser.add_argument("--output_dir", type=str, default="./lpm24_projector_output_contrastive", help="Directory to save trained models and logs.")

    parser.add_argument("--llm_model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Hugging Face model name for LLM.")
    parser.add_argument("--llm_max_length", type=int, default=256, help="Max sequence length for LLM tokenizer.")
    # Added for contrastive LLM feature extraction
    parser.add_argument("--llm_target_layer_k", type=int, default=-1,
                        help="Target layer in LLM to extract features from (-1 for last).")

    parser.add_argument("--unimol_feature_dim", type=int, default=512, help="Feature dimension of UniMol output.")
    parser.add_argument("--projector_arch", type=str, default="1024-2048", help="Architecture of the MLP projector (e.g., '512-1024').")
    parser.add_argument("--projector_type", type=str, default="MLP_Linear",
                        help="Type of projector architecture (MLP_Nonlinear or MLP_Linear).")

    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer.")
    parser.add_argument("--max_samples", type=int, default=10000, help="Maximum number of samples to use for training.")

    # Added for contrastive loss
    parser.add_argument("--contrastive_loss_weight", type=float, default=0.1, help="Weight for the contrastive loss component.")
    parser.add_argument("--contrastive_temperature_tau", type=float, default=0.07, help="Temperature parameter for InfoNCE loss.")

    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader.")
    parser.add_argument("--use_bfloat16", action='store_true', help="Use bfloat16 for training if available.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Number of steps to accumulate gradients.")

    parser.add_argument("--base_run_name", type=str, default="lpm24_proj_contrastive", help="Base name for the run.") # Updated base name
    parser.add_argument("--wandb_project", type=str, default="lpm24_projector_training", help="WANDB project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WANDB entity (username or team).")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WANDB run name.")
    parser.add_argument("--skip_unimol_errors", action='store_true', help="Skip SMILES that cause errors in UniMol processing.")

    args = parser.parse_args()
    return args

# --- Dataset Classes ---
class LPM24Dataset(Dataset):
    def __init__(self, smiles_list, captions_list):
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
    print("Loading LPM-24 dataset...")
    try:
        from datasets import logging
        logging.set_verbosity_error()
        dataset = load_dataset("language-plus-molecules/LPM-24_train",
                               data_files={'split_train': 'data/split_train-00000-of-00001.parquet',
                                           'split_valid': 'data/split_valid-00000-of-00001.parquet',
                                           'train': 'data/train-00000-of-00001.parquet'
                                           })
        train_data = dataset['split_train']
        valid_data = dataset['split_valid']
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
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=lambda batch: list(zip(*batch)))
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=lambda batch: list(zip(*batch)))
    return train_dataloader, valid_dataloader

def get_models(args, device, dtype):
    mol_fm = unimol_clf(avoid_loading_model=False, rep_cache_path=f'{args.base_data_path}/smile_to_rep_store.pkl')
    if hasattr(mol_fm, 'model') and mol_fm.model is not None:
        mol_fm.model.to(device)
        mol_fm.model.eval()
    elif isinstance(mol_fm, torch.nn.Module):
        mol_fm.to(device)
        mol_fm.eval()

    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_model_name)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
        llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id
    llm_tokenizer.padding_side = "left" # Set padding side for contrastive feature extraction

    llm_model = AutoModelForCausalLM.from_pretrained(args.llm_model_name, device_map="auto", torch_dtype=torch.bfloat16).to(device)

    projector_input_dim = args.unimol_feature_dim
    try:
        projector_output_dim = llm_model.config.hidden_size
        if projector_output_dim is None: raise AttributeError("llm_model.config.hidden_size is None.")
        print(f"Inferred LLM hidden_size for projector output: {projector_output_dim}")
    except Exception as e:
        default_fallback_dim = 4096
        print(f"Warning: Could not infer LLM hidden_size. Error: {e}. Using fallback: {default_fallback_dim}")
        projector_output_dim = default_fallback_dim

    if args.projector_type == "MLP_Nonlinear":
        projector = MLP_Nonlinear(ninput=projector_input_dim, noutput=projector_output_dim).to(device).to(dtype)
    elif args.projector_type == "MLP_Linear":
        projector = MLP_Linear(ninput=projector_input_dim, noutput=projector_output_dim, layers=args.projector_arch).to(device).to(dtype)
    elif args.projector_type == "Vicl_Linear":
        projector = Vicl_Linear(in_features=projector_input_dim, out_features=projector_output_dim).to(device).to(dtype)

    return mol_fm, llm_tokenizer, llm_model, projector

# --- Helper Functions ---
@torch.no_grad()
def get_unimol_features(smiles_batch, mol_fm, device, target_dtype, skip_errors=False):
    fm_reps_list = []
    valid_smiles_in_batch = []
    for smiles_str in smiles_batch:
        try:
            unimol_rep = mol_fm.get_unimol_rep_tensor(smiles_str)
            if unimol_rep is None or unimol_rep.nelement() == 0:
                raise ValueError("UniMol returned empty tensor.")
            fm_reps_list.append(unimol_rep.to(device, dtype=target_dtype))
            valid_smiles_in_batch.append(smiles_str)
        except Exception as e:
            if skip_errors:
                print(f"Skipping SMILES '{smiles_str}' due to UniMol error: {e}")
                continue
            else:
                raise e
    if not fm_reps_list:
        return None, []
    fm_reps_batch = torch.cat(fm_reps_list, dim=0)
    return fm_reps_batch, valid_smiles_in_batch

# --- NEW: Contrastive Loss Helper Functions ---
@torch.no_grad()
def get_llm_text_features(smiles_batch, llm, tokenizer, layer_k_arg, device, target_dtype, max_length=128):
    """Extracts text features (last token representation) from a specific layer of the LLM."""
    llm.eval()
    inputs = tokenizer(
        smiles_batch,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_length,
    ).to(device)

    outputs = llm(**inputs, output_hidden_states=True)
    hidden_states_tuple = outputs.hidden_states
    num_transformer_layers = len(hidden_states_tuple) - 1

    if layer_k_arg == -1 or layer_k_arg == num_transformer_layers:
        actual_layer_index = -1
    elif 0 < layer_k_arg <= num_transformer_layers:
        actual_layer_index = layer_k_arg
    else:
        raise ValueError(f"Invalid llm_target_layer_k: {layer_k_arg}. Max: {num_transformer_layers}")

    target_layer_hidden_states = hidden_states_tuple[actual_layer_index]
    last_token_features = target_layer_hidden_states[:, -1, :]
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
# --- END: Contrastive Loss Helper Functions ---


def compute_generation_loss(llm_model, tokenizer, projected_features, target_captions, prefix_embedding, postfix_embedding,
                            device, llm_max_length=256):
    """Compute cross-entropy loss for text generation."""
    batch_size = projected_features.shape[0]
    if prefix_embedding.dim() == 2: prefix_embedding = prefix_embedding.unsqueeze(0)
    if postfix_embedding.dim() == 2: postfix_embedding = postfix_embedding.unsqueeze(0)

    prefix_embeds = prefix_embedding.expand(batch_size, -1, -1)
    postfix_embeds = postfix_embedding.expand(batch_size, -1, -1)
    projected_embeds = projected_features.unsqueeze(1)

    prefix_len, proj_len, postfix_len = prefix_embeds.shape[1], projected_embeds.shape[1], postfix_embeds.shape[1]
    max_caption_len = llm_max_length - prefix_len - proj_len - postfix_len

    # Use original tokenizer padding side (right) for generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    captions_with_eos = [cap + tokenizer.eos_token for cap in target_captions]
    caption_tokenized = tokenizer(
        captions_with_eos, return_tensors="pt", padding=True, truncation=True,
        add_special_tokens=False, max_length=max_caption_len
    ).to(device)
    tokenizer.padding_side = original_padding_side # Restore padding side

    caption_ids = caption_tokenized.input_ids
    caption_attention_mask = caption_tokenized.attention_mask
    caption_embeds = llm_model.get_input_embeddings()(caption_ids)

    input_embeds = torch.cat([prefix_embeds, projected_embeds, postfix_embeds, caption_embeds], dim=1)
    prefix_labels = torch.full((batch_size, prefix_len), -100, dtype=torch.long, device=device)
    proj_labels = torch.full((batch_size, proj_len), -100, dtype=torch.long, device=device)
    postfix_labels = torch.full((batch_size, postfix_len), -100, dtype=torch.long, device=device)
    caption_labels = caption_ids.clone()
    caption_labels[caption_attention_mask == 0] = -100
    labels = torch.cat([prefix_labels, proj_labels, postfix_labels, caption_labels], dim=1)

    prefix_mask = torch.ones((batch_size, prefix_len), dtype=torch.long, device=device)
    proj_mask = torch.ones((batch_size, proj_len), dtype=torch.long, device=device)
    postfix_mask = torch.ones((batch_size, postfix_len), dtype=torch.long, device=device)
    attention_mask = torch.cat([prefix_mask, proj_mask, postfix_mask, caption_attention_mask], dim=1)

    outputs = llm_model(inputs_embeds=input_embeds, attention_mask=attention_mask, labels=labels)
    return outputs.loss


@torch.no_grad()
def generate_text_and_compute_bleu(llm_model, tokenizer, projected_features, target_captions, prefix_embedding, postfix_embedding, device, max_new_tokens=100):
    """Generate text and compute BLEU score"""
    generated_texts = []
    batch_size = projected_features.shape[0]
    prefix_embeds = prefix_embedding.expand(batch_size, -1, -1)
    postfix_embeds = postfix_embedding.expand(batch_size, -1, -1)
    projected_embeds = projected_features.unsqueeze(1)
    input_embeds = torch.cat([prefix_embeds, projected_embeds, postfix_embeds], dim=1)
    attention_mask = torch.ones(input_embeds.shape[:2], device=device)

    try:
        outputs = llm_model.generate(
            inputs_embeds=input_embeds, attention_mask=attention_mask, max_new_tokens=128,
            temperature=0.2, do_sample=True, pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        prompt_len = input_embeds.shape[1]
        generated_texts = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
    except Exception as e:
        print(f"Error in generation: {e}")
        generated_texts = [""] * batch_size

    metric_bleu = evaluate.load("sacrebleu")
    metric_rouge = evaluate.load("rouge")
    for pred, ref in zip(generated_texts, target_captions):
        metric_bleu.add(prediction=pred, reference=[ref])
        metric_rouge.add(prediction=pred, reference=ref)
    bleu_result = metric_bleu.compute()
    rouge_result = metric_rouge.compute()
    return bleu_result, rouge_result, generated_texts

# --- Main Training Function ---
def train_projector(args):
    set_random_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.use_bfloat16 and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        print("Using bfloat16.")
    elif torch.cuda.is_available():
        dtype = torch.float16
        print("Using float16 on GPU.")
    else:
        dtype = torch.float32
        print("Using float32 on CPU.")

    run_name_prefix = f"{args.base_run_name}_{args.projector_type}_lr_{args.learning_rate}_wd_{args.weight_decay}_bs_{args.batch_size}_ep_{args.num_epochs}_clw_{args.contrastive_loss_weight}"
    run_name = args.wandb_run_name if args.wandb_run_name else f"{run_name_prefix}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    wandb.init(project=args.wandb_project, name=run_name, config=args, entity=args.wandb_entity)

    train_dataloader, valid_dataloader = get_dataloaders(args)
    mol_fm, llm_tokenizer, llm_model, projector = get_models(args, device, dtype)
    optimizer = optim.AdamW(projector.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    num_training_steps = (len(train_dataloader) * args.num_epochs) // args.gradient_accumulation_steps
    num_warmup_steps = int(0.1 * num_training_steps)
    lr_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    args.output_dir = os.path.join(args.output_dir, "lpm24", run_name_prefix, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(args.output_dir, exist_ok=True)

    main_text, closing_text = "(", ")'s molecule caption is: "
    with torch.no_grad():
        prefix_token = llm_tokenizer.encode(main_text, return_tensors="pt", add_special_tokens=True).to(llm_model.device)
        postfix_token = llm_tokenizer.encode(closing_text, return_tensors="pt", add_special_tokens=False).to(llm_model.device)
        prefix_embedding = llm_model.get_input_embeddings()(prefix_token).to(dtype)
        postfix_embedding = llm_model.get_input_embeddings()(postfix_token).to(dtype)

    projector.train()
    llm_model.eval()
    for param in llm_model.parameters():
        param.requires_grad = False

    best_bleu_score = 0.0

    for epoch in range(args.num_epochs):
        total_train_loss = 0
        total_gen_loss = 0
        total_con_loss = 0
        train_batches = 0
        optimizer.zero_grad()
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{args.num_epochs}")

        for batch_idx, (smiles_batch, captions_batch) in progress_bar:
            if not smiles_batch or not captions_batch: continue

            fm_reps_batch, valid_smiles = get_unimol_features(smiles_batch, mol_fm, device, dtype, args.skip_unimol_errors)
            if fm_reps_batch is None or fm_reps_batch.shape[0] == 0: continue
            valid_captions = [captions_batch[smiles_batch.index(smiles)] for smiles in valid_smiles if smiles in smiles_batch]
            if len(valid_captions) != fm_reps_batch.shape[0]: continue

            projected_features = projector(fm_reps_batch)

            # --- Calculate Generation Loss ---
            gen_loss = compute_generation_loss(
                llm_model, llm_tokenizer, projected_features, valid_captions, prefix_embedding, postfix_embedding,
                device, args.llm_max_length
            )

            # --- Calculate Contrastive Loss ---
            llm_text_features = get_llm_text_features(
                valid_smiles, llm_model, llm_tokenizer, args.llm_target_layer_k,
                device, dtype, args.llm_max_length
            )
            con_loss = info_nce_loss(
                projected_features, llm_text_features, args.contrastive_temperature_tau
            )

            # --- Combine Losses ---
            loss = gen_loss + args.contrastive_loss_weight * con_loss
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            current_loss = loss.item() * args.gradient_accumulation_steps
            total_train_loss += current_loss
            total_gen_loss += gen_loss.item()
            total_con_loss += con_loss.item()
            train_batches += 1

            progress_bar.set_postfix({
                "Total L": f"{current_loss:.3f}",
                "Gen L": f"{gen_loss.item():.3f}",
                "Con L": f"{con_loss.item():.3f}",
                "LR": f"{lr_scheduler.get_last_lr()[0]:.1e}"
            })
            wandb.log({
                "train/batch_loss_total": current_loss,
                "train/batch_loss_gen": gen_loss.item(),
                "train/batch_loss_con": con_loss.item(),
                "lr": lr_scheduler.get_last_lr()[0]
            })

            gc.collect()
            torch.cuda.empty_cache()

        avg_train_loss = total_train_loss / train_batches if train_batches > 0 else 0
        avg_gen_loss = total_gen_loss / train_batches if train_batches > 0 else 0
        avg_con_loss = total_con_loss / train_batches if train_batches > 0 else 0
        wandb.log({
            "train/epoch_loss_total": avg_train_loss,
            "train/epoch_loss_gen": avg_gen_loss,
            "train/epoch_loss_con": avg_con_loss,
            "epoch": epoch
        })
        print(f"Epoch [{epoch+1}/{args.num_epochs}] Train Loss: {avg_train_loss:.4f} (Gen: {avg_gen_loss:.4f}, Con: {avg_con_loss:.4f})")

        # Validation
        projector.eval()
        llm_model.eval()
        total_valid_gen_loss, total_valid_con_loss = 0, 0
        valid_batches = 0
        all_bleu_scores, all_rouge_scores = [], []

        with torch.no_grad():
            for val_batch_idx, (smiles_batch, captions_batch) in tqdm(enumerate(valid_dataloader), total=len(valid_dataloader), desc="Validation"):
                if not smiles_batch or not captions_batch: continue
                fm_reps_batch, valid_smiles = get_unimol_features(smiles_batch, mol_fm, device, dtype, args.skip_unimol_errors)
                if fm_reps_batch is None or fm_reps_batch.shape[0] == 0: continue
                valid_captions = [captions_batch[smiles_batch.index(smiles)] for smiles in valid_smiles if smiles in smiles_batch]
                if len(valid_captions) != fm_reps_batch.shape[0]: continue

                projected_features = projector(fm_reps_batch)

                try:
                    valid_gen_loss = compute_generation_loss(llm_model, llm_tokenizer, projected_features, valid_captions, prefix_embedding, postfix_embedding, device, args.llm_max_length)
                    llm_text_features_val = get_llm_text_features(valid_smiles, llm_model, llm_tokenizer, args.llm_target_layer_k, device, dtype, args.llm_max_length)
                    valid_con_loss = info_nce_loss(projected_features, llm_text_features_val, args.contrastive_temperature_tau)

                    total_valid_gen_loss += valid_gen_loss.item()
                    total_valid_con_loss += valid_con_loss.item()
                    valid_batches += 1

                    if val_batch_idx % 10 == 0:
                        bleu_score, rouge_score, _ = generate_text_and_compute_bleu(llm_model, llm_tokenizer, projected_features, valid_captions, prefix_embedding, postfix_embedding, device)
                        all_bleu_scores.append(bleu_score["score"])
                        all_rouge_scores.append(rouge_score["rouge1"])
                except Exception as e:
                    print(f"Error in validation batch {val_batch_idx}: {e}")
                    continue
            gc.collect()
            torch.cuda.empty_cache()

        avg_valid_gen_loss = total_valid_gen_loss / valid_batches if valid_batches > 0 else 0
        avg_valid_con_loss = total_valid_con_loss / valid_batches if valid_batches > 0 else 0
        avg_valid_total_loss = avg_valid_gen_loss + args.contrastive_loss_weight * avg_valid_con_loss
        avg_bleu_score = np.mean(all_bleu_scores) if all_bleu_scores else 0.0
        avg_rouge_score = np.mean(all_rouge_scores) if all_rouge_scores else 0.0

        wandb.log({
            "valid/epoch_loss_total": avg_valid_total_loss,
            "valid/epoch_loss_gen": avg_valid_gen_loss,
            "valid/epoch_loss_con": avg_valid_con_loss,
            "valid/bleu_score": avg_bleu_score,
            "valid/rouge1_score": avg_rouge_score,
            "epoch": epoch
        })

        print(f"Epoch [{epoch+1}/{args.num_epochs}] Valid Loss: {avg_valid_total_loss:.4f} (Gen: {avg_valid_gen_loss:.4f}, Con: {avg_valid_con_loss:.4f}), BLEU: {avg_bleu_score:.4f}, ROUGE-1: {avg_rouge_score:.4f}")

        if avg_bleu_score > best_bleu_score:
            best_bleu_score = avg_bleu_score
            model_path = os.path.join(args.output_dir, "best_projector.pth")
            torch.save(projector.state_dict(), model_path)
            wandb.save(model_path)
            print(f"Saved best model with BLEU score: {best_bleu_score:.4f}")

        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    final_model_path = os.path.join(args.output_dir, "final_projector.pth")
    torch.save(projector.state_dict(), final_model_path)
    wandb.save(final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")
    wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    print("Starting LPM-24 contrastive projector training with arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    train_projector(args)