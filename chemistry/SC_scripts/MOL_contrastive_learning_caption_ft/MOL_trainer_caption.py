import argparse
import datetime
import gc
import os
# set wandb to offline mode
os.environ["WANDB_MODE"] = "offline"
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

    # LLM
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_model_name)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
        llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id

    # llm_model_kwargs = {"torch_dtype": dtype} if (args.use_bfloat16 and torch.cuda.is_bf16_supported()) or (not args.use_bfloat16 and torch.cuda.is_available()) else {}
    llm_model = AutoModelForCausalLM.from_pretrained(args.llm_model_name, device_map="auto", torch_dtype=torch.bfloat16).to(device)

    #  device_map="balanced_low_0", torch_dtype=torch.bfloat16, attn_implementation=attn_implementation, trust_remote_code=True

    # Get LLM embedding dimension
    # llm_embed_dim = llm_model.config.hidden_size

    # Projector
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

    elif args.projector_type == "Vicl_Linear":
        projector = Vicl_Linear(
            in_features=projector_input_dim,
            out_features=projector_output_dim,
        ).to(device).to(dtype)

    return mol_fm, llm_tokenizer, llm_model, projector

# --- Helper Functions ---
@torch.no_grad()
def get_unimol_features(smiles_batch, mol_fm, device, target_dtype, skip_errors=False):
    """Extract UniMol features for a batch of SMILES"""
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

def compute_generation_loss(llm_model, tokenizer, projected_features, target_captions, prefix_embedding, postfix_embedding,
                            device, llm_max_length=256):
    """Compute cross-entropy loss for text generation following the reference script's pattern."""

    batch_size = projected_features.shape[0]

    # Ensure prefix and postfix embeddings are correctly shaped (1, seq_len, embed_dim)
    if prefix_embedding.dim() == 2:
        prefix_embedding = prefix_embedding.unsqueeze(0)
    if postfix_embedding.dim() == 2:
        postfix_embedding = postfix_embedding.unsqueeze(0)

    # Expand prefix and postfix to batch size
    prefix_embeds = prefix_embedding.expand(batch_size, -1, -1)
    postfix_embeds = postfix_embedding.expand(batch_size, -1, -1)
    projected_embeds = projected_features.unsqueeze(1) # (batch_size, 1, embed_dim)

    # print(f"Prefix shape: {prefix_embeds.shape}, Projected shape: {projected_embeds.shape}, Postfix shape: {postfix_embeds.shape}")

    prefix_len = prefix_embeds.shape[1]
    proj_len = projected_embeds.shape[1]
    postfix_len = postfix_embeds.shape[1]

    # Calculate max length for captions
    max_caption_len = llm_max_length - prefix_len - proj_len - postfix_len

    # Tokenize target captions WITH padding and truncation
    # Add EOS token manually if not present, as we need it for labels.
    captions_with_eos = [cap + tokenizer.eos_token for cap in target_captions]

    caption_tokenized = tokenizer(
        captions_with_eos,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=False,
        max_length=max_caption_len
    ).to(device)

    caption_ids = caption_tokenized.input_ids
    caption_attention_mask = caption_tokenized.attention_mask

    # Get caption embeddings
    caption_embeds = llm_model.get_input_embeddings()(caption_ids)

    # Combine embeddings: [Prefix, Projection, Postfix, Caption]
    input_embeds = torch.cat([prefix_embeds, projected_embeds, postfix_embeds, caption_embeds], dim=1)

    # Create labels: -100 for non-caption parts, caption_ids for caption part
    prefix_labels = torch.full((batch_size, prefix_len), -100, dtype=torch.long, device=device)
    proj_labels = torch.full((batch_size, proj_len), -100, dtype=torch.long, device=device)
    postfix_labels = torch.full((batch_size, postfix_len), -100, dtype=torch.long, device=device)

    # Mask padding tokens in caption_ids with -100
    caption_labels = caption_ids.clone()
    caption_labels[caption_attention_mask == 0] = -100

    # Combine labels
    labels = torch.cat([prefix_labels, proj_labels, postfix_labels, caption_labels], dim=1)

    # Create attention mask for the full sequence
    prefix_mask = torch.ones((batch_size, prefix_len), dtype=torch.long, device=device)
    proj_mask = torch.ones((batch_size, proj_len), dtype=torch.long, device=device)
    postfix_mask = torch.ones((batch_size, postfix_len), dtype=torch.long, device=device)
    attention_mask = torch.cat([prefix_mask, proj_mask, postfix_mask, caption_attention_mask], dim=1)


    # decoded_captions = tokenizer.batch_decode(caption_ids, skip_special_tokens=False)
    # print("=== Decoded Captions ===")
    # for i, cap in enumerate(decoded_captions):
    #     print(f"[{i}] {cap}")

    decoded_input_token_ids = find_nearest_token_ids(input_embeds, llm_model.model.embed_tokens.weight)
    decoded_final_input_texts = tokenizer.batch_decode(decoded_input_token_ids, skip_special_tokens=False)
    print("=== Decoded Input Texts ===")
    for i, text in enumerate(decoded_final_input_texts):
        print(f"[{i}] {text}")

    # Forward pass through LLM
    outputs = llm_model(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        labels=labels
    )

    return outputs.loss



@torch.no_grad()
def generate_text_and_compute_bleu(llm_model, tokenizer, projected_features, target_captions, prefix_embedding, postfix_embedding, device, max_new_tokens=100):
    """Generate text and compute BLEU score"""
    generated_texts = []
    batch_size = projected_features.shape[0]

    # Expand prefix and postfix to batch size
    prefix_embeds = prefix_embedding.expand(batch_size, -1, -1)
    postfix_embeds = postfix_embedding.expand(batch_size, -1, -1)
    projected_embeds = projected_features.unsqueeze(1) # (batch_size, 1, embed_dim)

    # Combine embeddings: [Prefix, Projection, Postfix] - This is the prompt
    input_embeds = torch.cat([prefix_embeds, projected_embeds, postfix_embeds], dim=1)

    # print(f"Input Embeds shape: {input_embeds.shape}")
    
    # Create attention mask for the prompt
    # Input Embeds shape: torch.Size([10, 10, 4096])
    attention_mask = torch.ones(input_embeds.shape[:2], device=device)
    #  Validation Loss: 3.7136, BLEU Score: 3.2427

    # decoded_input_token_ids = find_nearest_token_ids(input_embeds, llm_model.model.embed_tokens.weight)
    # decoded_final_input_texts = tokenizer.batch_decode(decoded_input_token_ids, skip_special_tokens=False)
    # print("=== Decoded Input Texts ===")
    # for i, text in enumerate(decoded_final_input_texts):
    #     print(f"[{i}] {text}")

    try:
        # Generate text
        outputs = llm_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=128,
            temperature=0.2, 
            do_sample=True,
            # return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        # Decode generated texts, skipping the prompt part
        prompt_len = input_embeds.shape[1]
        generated_texts = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)

    except Exception as e:
        print(f"Error in generation: {e}")
        generated_texts = [""] * batch_size

    metric_bleu = evaluate.load("sacrebleu")
    metric_rouge = evaluate.load("rouge")

    for pred, ref in zip(generated_texts, target_captions):
        # Evaluate expects prediction as str, reference as list[str]
        # print(f"Generated: {pred}, Target: {ref}")
        metric_bleu.add(prediction=pred, reference=[ref])
        metric_rouge.add(prediction=pred, reference=ref)

    bleu_result = metric_bleu.compute()
    rouge_result = metric_rouge.compute()

    print("BLEU:", bleu_result)
    print("ROUGE:", rouge_result)
    return bleu_result, rouge_result,generated_texts

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

    if args.projector_type == "MLP_Nonlinear":
        run_name_prefix = f"{args.base_run_name}_Nonlinear_lr_{args.learning_rate}_weight_decay_{args.weight_decay}_batch_{args.batch_size}_epochs{args.num_epochs}_lpm24"
    if args.projector_type == "Vicl_Linear":
        run_name_prefix = f"{args.base_run_name}_Vicl_Linear_lr_{args.learning_rate}_weight_decay_{args.weight_decay}_batch_{args.batch_size}_epochs{args.num_epochs}_lpm24"
    elif args.projector_type == "MLP_Linear":
        run_name_prefix = f"{args.base_run_name}_Linear_arch_{args.projector_arch}_lr_{args.learning_rate}_weight_decay_{args.weight_decay}_batch_{args.batch_size}_epochs{args.num_epochs}_lpm24"
    
    run_name = args.wandb_run_name if args.wandb_run_name else f"{run_name_prefix}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=args,
        entity=args.wandb_entity # Added entity
    )

    # Data loaders
    train_dataloader, valid_dataloader = get_dataloaders(args)

    # Models
    mol_fm, llm_tokenizer, llm_model, projector = get_models(args, device, dtype)

    # Optimizer
    optimizer = optim.AdamW(projector.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    num_training_steps = (len(train_dataloader) * args.num_epochs) // args.gradient_accumulation_steps
    num_warmup_steps = int(0.1 * num_training_steps) # 10% 预热步数
    
    lr_scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    
    # Training loop
    best_bleu_score = 0.0
    args.output_dir = os.path.join(args.output_dir, "lpm24")

    # add run_name_prefix
    args.output_dir = os.path.join(args.output_dir, run_name_prefix)

    # add timestamp
    args.output_dir = os.path.join(args.output_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(args.output_dir, exist_ok=True)


    main_text = "("
    closing_text = ")'s molecule caption is: " # Added a space for better prompting

    # Get prefix and postfix embeddings
    with torch.no_grad():
        prefix_token = llm_tokenizer.encode(main_text, return_tensors="pt", add_special_tokens=True).to(llm_model.device)
        postfix_token = llm_tokenizer.encode(closing_text, return_tensors="pt", add_special_tokens=False).to(llm_model.device)
        prefix_embedding = llm_model.get_input_embeddings()(prefix_token).to(dtype)
        postfix_embedding = llm_model.get_input_embeddings()(postfix_token).to(dtype)

    projector.train()
    llm_model.eval()
    for param in llm_model.parameters():
        param.requires_grad = False

    for epoch in range(args.num_epochs):

        total_train_loss = 0
        train_batches = 0

        optimizer.zero_grad()

        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{args.num_epochs}")

        for batch_idx, (smiles_batch, captions_batch) in progress_bar:
            if not smiles_batch or not captions_batch:
                continue

            # Get UniMol features
            fm_reps_batch, valid_smiles = get_unimol_features(
                smiles_batch, mol_fm, device, dtype, args.skip_unimol_errors
            )

            if fm_reps_batch is None or fm_reps_batch.shape[0] == 0:
                continue

            # Get corresponding valid captions
            valid_captions = [captions_batch[smiles_batch.index(smiles)] for smiles in valid_smiles if smiles in smiles_batch]

            if len(valid_captions) != fm_reps_batch.shape[0]:
                print(f"Skipping batch {batch_idx}: Mismatch between features ({fm_reps_batch.shape[0]}) and captions ({len(valid_captions)})")
                continue

            # Project UniMol features
            projected_features = projector(fm_reps_batch)

            # Compute generation loss
            loss = compute_generation_loss(
                llm_model, llm_tokenizer, projected_features, valid_captions, prefix_embedding, postfix_embedding,
                device, args.llm_max_length
            )
            loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step() # <--- 更新学习率
                optimizer.zero_grad()

            current_loss = loss.item() * args.gradient_accumulation_steps
            total_train_loss += current_loss
            train_batches += 1

            # progress_bar.set_postfix({"loss": current_loss})
            # wandb.log({"train/batch_loss": current_loss})

            progress_bar.set_postfix({"loss": current_loss, "lr": lr_scheduler.get_last_lr()[0]})
            wandb.log({"train/batch_loss": current_loss, "lr": lr_scheduler.get_last_lr()[0]})


            gc.collect() # Try to clear memory
            torch.cuda.empty_cache()


        avg_train_loss = total_train_loss / train_batches if train_batches > 0 else 0
        wandb.log({"train/epoch_loss": avg_train_loss, "epoch": epoch})
        print(f"Epoch [{epoch+1}/{args.num_epochs}] Training Loss: {avg_train_loss:.4f}")

        # Validation
        projector.eval()
        llm_model.eval()

        total_valid_loss = 0
        valid_batches = 0
        all_bleu_scores = []
        all_rouge_scores = []
        
        with torch.no_grad():
            for val_batch_idx, (smiles_batch, captions_batch) in tqdm(enumerate(valid_dataloader), total=len(valid_dataloader), desc="Validation"):
                if not smiles_batch or not captions_batch:
                    continue

                fm_reps_batch, valid_smiles = get_unimol_features(
                    smiles_batch, mol_fm, device, dtype, args.skip_unimol_errors
                )

                if fm_reps_batch is None or fm_reps_batch.shape[0] == 0:
                    continue

                valid_captions = [captions_batch[smiles_batch.index(smiles)] for smiles in valid_smiles if smiles in smiles_batch]

                if len(valid_captions) != fm_reps_batch.shape[0]:
                    continue

                projected_features = projector(fm_reps_batch)

                try:
                    # Compute validation loss
                    valid_loss = compute_generation_loss(
                        llm_model, llm_tokenizer, projected_features, valid_captions, prefix_embedding, postfix_embedding, # Fixed call
                        device, args.llm_max_length
                    )
                    total_valid_loss += valid_loss.item()
                    valid_batches += 1

                    # Compute BLEU score for a subset (e.g., first batch or every N batches)
                    if val_batch_idx % 10 == 0: # Compute BLEU every 10 batches to save time
                        bleu_score, rouge_score, _ = generate_text_and_compute_bleu(
                             llm_model, llm_tokenizer, projected_features, valid_captions, prefix_embedding, postfix_embedding, device
                         )
                        all_bleu_scores.append(bleu_score["score"])
                        all_rouge_scores.append(rouge_score["rouge1"])

                except Exception as e:
                    print(f"Error in validation batch {val_batch_idx}: {e}")
                    continue
            gc.collect()
            torch.cuda.empty_cache()

        avg_valid_loss = total_valid_loss / valid_batches if valid_batches > 0 else 0
        avg_bleu_score = np.mean(all_bleu_scores) if all_bleu_scores else 0.0

        wandb.log({
            "valid/epoch_loss": avg_valid_loss,
            "valid/bleu_score": avg_bleu_score,
            "epoch": epoch
        })

        print(f"Epoch [{epoch+1}/{args.num_epochs}] Validation Loss: {avg_valid_loss:.4f}, BLEU Score: {avg_bleu_score:.4f}")

        # Save best model based on BLEU score
        if avg_bleu_score > best_bleu_score:
            best_bleu_score = avg_bleu_score
            model_path = os.path.join(args.output_dir, "best_projector.pth")
            torch.save(projector.state_dict(), model_path)
            wandb.save(model_path) # Save best model to wandb
            print(f"Saved best model with BLEU score: {best_bleu_score:.4f}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final model saving
    final_model_path = os.path.join(args.output_dir, "final_projector.pth")
    torch.save(projector.state_dict(), final_model_path)
    wandb.save(final_model_path) # Save final model to wandb
    print(f"Training complete. Final model saved to {final_model_path}")
    wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    print("Starting LPM-24 projector training with arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")


    train_projector(args)