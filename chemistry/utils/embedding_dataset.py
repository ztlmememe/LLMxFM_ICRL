"""
Dataset utilities for embedding-based LoRA fine-tuning of LLMs.
This module provides classes and functions for preparing embedding datasets.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import logging
import random
from typing import Dict, List, Tuple, Optional, Union
from chemistry.utils.data_generator import prepare_data


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def prepare_datasets(args, model, tokenizer, data_dict,system_prompt):
    """
    Prepare training and testing datasets.
    
    Args:
        args: Parsed arguments
        model: Language model
        tokenizer: Tokenizer
        data_dict: Dictionary containing training and testing data
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    
    # Generate data
    # logger.info("Generating data for debugging")
    # data_dict = prepare_data(args)
    
    # Create datasets with answer weight parameter
    logger.info("Creating datasets")
    train_dataset = EmbeddingDataset(
        input_embeddings=data_dict['train_inputs'],
        target_values=data_dict['train_y'],
        tokenizer=tokenizer,
        model=model,
        system_prompt=system_prompt,
        max_length=args.max_length,
        mapping_ids=data_dict['train_map_ids'],
        answer_weight=args.answer_weight
    )
    
    test_dataset = EmbeddingDataset(
        input_embeddings=data_dict['test_inputs'],
        target_values=data_dict['test_y'],
        tokenizer=tokenizer,
        model=model,
        system_prompt=system_prompt,
        max_length=args.max_length,
        mapping_ids=data_dict['test_map_ids'],
        answer_weight=args.answer_weight
    )
    
    return train_dataset, test_dataset





class EmbeddingDataset(Dataset):
    """Dataset for LoRA fine-tuning with embedding inputs and text outputs."""
    
    def __init__(
        self, 
        input_embeddings: torch.Tensor, 
        target_values: torch.Tensor,
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
        system_prompt: str,
        max_length: int,
        mapping_ids: Optional[List[int]] = None,
        answer_weight: float = 5.0 
    ):
        """
        Initialize dataset with embedding inputs and target values.
        
        Args:
            input_embeddings: Tensor of shape (n_samples, embedding_dim)
            target_values: Tensor of shape (n_samples, 1) containing target float values
            tokenizer: Tokenizer for the model
            model: The language model
            system_prompt: System prompt to guide the model
            max_length: Maximum sequence length
            mapping_ids: Optional list of mapping IDs for tracking
            answer_weight: Weight to apply to the "Answer: X.XX" part (default: 5.0)
        """
        self.input_embeddings = input_embeddings
        self.target_values = target_values
        self.tokenizer = tokenizer
        self.model = model
        self.system_prompt = system_prompt
        self.max_length = max_length
        self.mapping_ids = mapping_ids if mapping_ids is not None else [-1] * len(input_embeddings)
        self.hidden_size = model.config.hidden_size
        self.answer_weight = answer_weight 
        
        # Ensure the tokenizer has the special tokens we'll need
        # special_tokens = {"additional_special_tokens": ["<rep>", "</rep>"]}
        # self.tokenizer.add_special_tokens(special_tokens)
        # self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Process all items at initialization to avoid repeated computation
        logger.info("Preparing dataset items...")
        self.processed_items = []
        for i in tqdm(range(len(self.input_embeddings))):
            item = self._process_item(i)
            self.processed_items.append(item)
        logger.info(f"Prepared {len(self.processed_items)} dataset items")
    
    def __len__(self):
        return len(self.input_embeddings)
    
    def __getitem__(self, idx):
        return self.processed_items[idx]
    
    def collate_fn(self, batch):
        """
        Custom collate function that properly handles embedding inputs.
        Ensures all tensors are on CPU for proper pinning by DataLoader.
        """
        batch_inputs = []
        batch_attention_masks = []
        batch_labels = []
        batch_weight_masks = []
        batch_mapping_ids = []
        
        for item in batch:
            # Create a CPU version of inputs
            cpu_inputs = {
                'input_ids_prefix': item['input_ids_prefix'].cpu(),
                'input_indicator_ids': item['input_indicator_ids'].cpu(),
                'rep_start_token_id': item['rep_start_token_id'].cpu(),
                'model_input_embedding': item['model_input_embedding'].detach().cpu(),
                'rep_end_token_id': item['rep_end_token_id'].cpu(),
                'completion_prefix_ids': item['completion_prefix_ids'].cpu(),
                'completion_tokens': item['completion_tokens'].cpu(),
                'attention_mask': item['attention_mask'].cpu(),
                'labels': item['labels'].cpu(),
                'weight_mask': item['weight_mask'].cpu(),
                'mapping_id': item['mapping_id'],
                'embedding_start_idx': item['embedding_start_idx'],
                'embedding_end_idx': item['embedding_end_idx'],
                'completion_start_idx': item['completion_start_idx'],
                'answer_start_idx': item.get('answer_start_idx', -1),
                'answer_end_idx': item.get('answer_end_idx', -1)
            }
            
            # Add to batch
            batch_inputs.append(cpu_inputs)
            batch_attention_masks.append(item['attention_mask'].cpu())
            batch_labels.append(item['labels'].cpu())
            batch_weight_masks.append(item['weight_mask'].cpu())
            batch_mapping_ids.append(item['mapping_id'])
        
        return {
            'inputs': batch_inputs,
            'attention_mask': torch.nn.utils.rnn.pad_sequence(batch_attention_masks, batch_first=True, padding_value=0),
            'labels': torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=-100),
            'weight_masks': torch.nn.utils.rnn.pad_sequence(batch_weight_masks, batch_first=True, padding_value=0), 
            'mapping_ids': batch_mapping_ids
        }


    def _process_item(self, idx):
        """Process a single dataset item with embeddings and target value."""
        embedding = self.input_embeddings[idx]
        target_value = self.target_values[idx]
        mapping_id = self.mapping_ids[idx]
        
        # Format data with embeddings and weights
        formatted_data = format_data_with_embeddings(
            model_input=embedding,
            target_value=target_value,
            tokenizer=self.tokenizer,
            model=self.model,
            system_prompt=self.system_prompt,
            max_length=self.max_length,
            map_id=mapping_id,
            detach_from_device=True,  # Detach from device
            answer_weight=self.answer_weight
        )
        
        return formatted_data
    
def normalize_emb_mean_var(model_input_embeddings, target_mean, target_var):
    """
    normalize the mean and variance of the embeddings
    """

    mean = model_input_embeddings.mean(-1, keepdim=True)
    var = model_input_embeddings.var(-1, keepdim=True)

    model_input_embeddings = (model_input_embeddings - mean) * torch.sqrt(target_var / var) + target_mean

    return model_input_embeddings


def format_data_with_embeddings(
    model_input, 
    target_value, 
    tokenizer, 
    model, 
    system_prompt, 
    max_length, 
    map_id=None,
    detach_from_device=False,
    answer_weight=5.0
):
    """
    Format embedding input and text output for training.
    
    Args:
        model_input: Embedding tensor input
        target_value: Target float value to be converted to text
        tokenizer: Tokenizer for the model
        model: The language model
        system_prompt: System prompt to guide the model
        max_length: Maximum sequence length
        map_id: Optional mapping ID for tracking
        detach_from_device: Whether to detach tensors from device
        answer_weight: Weight to apply to the "Answer: X.XX" part
        
    Returns:
        Dictionary with formatted input for training that preserves embeddings
    """
    # Device for processing
    device = model.device
    embedding_dim = model_input.shape[-1]
    
    # Get embedding statistics for normalization
    try:
        embedding_matrix = model.model.embed_tokens.weight
    except:
        embedding_matrix = model.model.model.embed_tokens.weight


    non_zero_embeddings = embedding_matrix[~torch.all(torch.abs(embedding_matrix) < 1e-10, dim=1)]

    # Get statistics of non-zero embeddings
    non_zero_embeddings_mean = non_zero_embeddings.mean(-1, keepdim=True).mean().detach().item()
    non_zero_embeddings_var = non_zero_embeddings.var(-1, keepdim=True).mean().detach().item()
    
    # Format the target value as a string with 2 decimal places
    y_str = f"{float(target_value):.2f}"
    
    # Determine model type for appropriate prompt formatting
    model_name = model.config._name_or_path.lower()
    
    if "llama-3" in model_name:
        model_type = "llama3"
    elif "llama-2" in model_name:
        model_type = "llama2"
    elif "qwen" in model_name:
        model_type = "qwen2"
    elif "command" in model_name:
        model_type = "c4ai-command-r-plus"
    else:
        # Default formatting for other models
        model_type = "generic"
    
    # Get token embeddings for special tokens
    rep_start_token = "<rep>"
    rep_end_token = "</rep>"

    # print("model_name", model_name)
    
    # Get the actual token IDs for system message and other text parts
    if "qwen" in model_name.lower():
        system_prefix = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n"
        input_indicator = "\nInput vector:"
        completion_prefix = "\n<|im_end|>\n<|im_start|>assistant\n"
    elif "llama-3" in model_name.lower():
        system_prefix = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>"
        input_indicator = "\nInput vector:"
        completion_prefix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    elif model_type == "c4ai-command-r-plus":
        system_prefix = f"{system_prompt}<|START_OF_TURN_TOKEN|><|USER_TOKEN|>"
        input_indicator = "\nInput vector:"
        completion_prefix = "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|ASSISTANT_TOKEN|>"
    else:
        # Generic formatting
        system_prefix = f"System: {system_prompt}\nUser: "
        input_indicator = "\nInput vector:"
        completion_prefix = "\nAssistant: "
    
    # Tokenize text parts to get input IDs
    system_prefix_ids = tokenizer(
        system_prefix, 
        return_tensors="pt", 
        add_special_tokens=False
    ).input_ids[0].to(device)
    
    input_indicator_ids = tokenizer(
        input_indicator, 
        return_tensors="pt", 
        add_special_tokens=False
    ).input_ids[0].to(device)
    
    # Tokenize rep tokens
    rep_start_token_id = tokenizer.encode(rep_start_token, add_special_tokens=False, return_tensors="pt")[0].to(device)
    rep_end_token_id = tokenizer.encode(rep_end_token, add_special_tokens=False, return_tensors="pt")[0].to(device)
    
    completion_prefix_ids = tokenizer(
        completion_prefix, 
        return_tensors="pt", 
        add_special_tokens=False
    ).input_ids[0].to(device)
    
    # Normalize the embedding input
    normalized_embedding = model_input.clone()
    if normalized_embedding.dim() == 1:
        normalized_embedding = normalized_embedding.unsqueeze(0)  # Add batch dimension if needed
    
    normalized_embedding = normalize_emb_mean_var(normalized_embedding, non_zero_embeddings_mean, non_zero_embeddings_var).to(device)
    
    # Format template parts separately to identify "Answer:" part
    template_start = "\n\n---BEGIN FORMAT TEMPLATE FOR QUESTION---\nAnswer: "
    answer_part = f"{y_str}"
    template_end = "\n---END FORMAT TEMPLATE FOR QUESTION---"


    
    # Tokenize each part separately
    template_start_tokens = tokenizer(
        template_start, 
        return_tensors="pt", 
        add_special_tokens=False
    ).input_ids[0].to(device)
    
    answer_tokens = tokenizer(
        answer_part, 
        return_tensors="pt", 
        add_special_tokens=False
    ).input_ids[0].to(device)
    
    template_end_tokens = tokenizer(
        template_end, 
        return_tensors="pt", 
        add_special_tokens=False
    ).input_ids[0].to(device)
    
    # Combine for the full completion
    completion_tokens = torch.cat([template_start_tokens, answer_tokens, template_end_tokens])
    
    # Create position indices for different parts of input
    # The input will have this structure:
    # [system_prefix, input_indicator, rep_start, EMBEDDING, rep_end, completion_prefix, template_start, answer, template_end]
    
    # Track token positions for embedding insertion
    embedding_start_idx = len(system_prefix_ids) + len(input_indicator_ids) + len(rep_start_token_id)
    embedding_length = normalized_embedding.shape[1] if normalized_embedding.dim() > 1 else 1
    embedding_end_idx = embedding_start_idx + embedding_length
    
    # Calculate position indices for completion parts
    completion_start_idx = embedding_end_idx + len(rep_end_token_id) + len(completion_prefix_ids)
    answer_start_idx = completion_start_idx + len(template_start_tokens)
    answer_end_idx = answer_start_idx + len(answer_tokens)
    
    # Create attention mask
    attention_mask_parts = [
        torch.ones_like(system_prefix_ids),  # System prefix
        torch.ones_like(input_indicator_ids),  # Input indicator
        torch.ones_like(rep_start_token_id),  # Rep start token
        torch.ones(embedding_length, dtype=torch.long, device=device),  # Embedding
        torch.ones_like(rep_end_token_id),  # Rep end token
        torch.ones_like(completion_prefix_ids),  # Completion prefix
        torch.ones_like(template_start_tokens),  # Template start
        torch.ones_like(answer_tokens),  # Answer tokens
        torch.ones_like(template_end_tokens)  # Template end
    ]
    
    attention_mask = torch.cat(attention_mask_parts)
    
    # Create labels (-100 for prompt to ignore in loss calculation)
    labels_parts = [
        torch.full_like(system_prefix_ids, -100),  # System prefix (ignored)
        torch.full_like(input_indicator_ids, -100),  # Input indicator (ignored)
        torch.full_like(rep_start_token_id, -100),  # Rep start token (ignored)
        torch.full(size=(embedding_length,), fill_value=-100, dtype=torch.long, device=device),  # Embedding (ignored)
        torch.full_like(rep_end_token_id, -100),  # Rep end token (ignored)
        torch.full_like(completion_prefix_ids, -100),  # Completion prefix (ignored)
        template_start_tokens,  # Template start (included in loss)
        answer_tokens,  # Answer tokens (included in loss)
        template_end_tokens,  # Template end (included in loss)
    ]
    
    labels = torch.cat(labels_parts)
    
    # Create weight mask for the answer part
    weight_mask_parts = [
        torch.zeros_like(system_prefix_ids),  # System prefix (ignored)
        torch.zeros_like(input_indicator_ids),  # Input indicator (ignored)
        torch.zeros_like(rep_start_token_id),  # Rep start token (ignored)
        torch.zeros(embedding_length, dtype=torch.float, device=device),  # Embedding (ignored)
        torch.zeros_like(rep_end_token_id),  # Rep end token (ignored)
        torch.zeros_like(completion_prefix_ids),  # Completion prefix (ignored)

        torch.ones_like(template_start_tokens, dtype=torch.float),  # Template start (weight 1.0)
        torch.full_like(answer_tokens, fill_value=answer_weight, dtype=torch.float),  # Answer tokens (weight answer_weight)
        torch.ones_like(template_end_tokens, dtype=torch.float)  # Template end (weight 1.0)
    ]

    # print("template_start_tokens.shape", template_start_tokens.shape)
    # print("answer_tokens.shape", answer_tokens.shape)
    # print("template_end_tokens.shape", template_end_tokens.shape)
    
    weight_mask = torch.cat(weight_mask_parts)
    
    # Prepare final result dictionary
    result = {
        'input_ids_prefix': system_prefix_ids,
        'input_indicator_ids': input_indicator_ids,
        'rep_start_token_id': rep_start_token_id,
        'model_input_embedding': normalized_embedding,
        'rep_end_token_id': rep_end_token_id,
        'completion_prefix_ids': completion_prefix_ids,
        'completion_tokens': completion_tokens,
        'attention_mask': attention_mask,
        'labels': labels,
        'weight_mask': weight_mask,  # Add weight mask for loss calculation
        'embedding_start_idx': embedding_start_idx,
        'embedding_end_idx': embedding_end_idx,
        'completion_start_idx': completion_start_idx,
        'answer_start_idx': answer_start_idx,
        'answer_end_idx': answer_end_idx,
        'mapping_id': map_id if map_id is not None else -1
    }
    
    # Detach from device if requested
    if detach_from_device:
        for key in result:
            if isinstance(result[key], torch.Tensor):
                result[key] = result[key].cpu()
    
    return result
