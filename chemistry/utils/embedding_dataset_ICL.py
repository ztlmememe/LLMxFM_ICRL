"""
Revised embedding handling for ICL dataset preparation with grouped answers.
This module implements the query-then-answers format with fixed embedding position tracking.
"""

import torch
import logging
from typing import List, Dict, Tuple, Optional, Union

logger = logging.getLogger(__name__)



def format_icl_batch_with_embeddings(
    model,
    tokenizer,
    system_prompt,
    example_embeddings,
    example_targets,
    query_embeddings,
    query_targets,
    max_length,
    answer_weight=5.0,
    example_map_ids=None,
    query_map_ids=None
):
    """
    Format a batch of ICL examples and queries with embeddings.
    Revised to implement the query-then-answers format with fixed embedding positions.
    
    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        system_prompt: System prompt to guide the model
        example_embeddings: List of example embedding tensors
        example_targets: List of example target values
        query_embeddings: List of query embedding tensors
        query_targets: List of query target values
        max_length: Maximum sequence length
        answer_weight: Weight to apply to the answer part
        example_map_ids: Optional mapping IDs for examples
        query_map_ids: Optional mapping IDs for queries
        
    Returns:
        Dictionary with formatted ICL batch
    """
    device = model.device
    
    # Get embedding statistics for normalization
    try:
        embedding_matrix = model.model.embed_tokens.weight
    except AttributeError:
        try:
            embedding_matrix = model.model.model.embed_tokens.weight
        except AttributeError:
            embedding_matrix = model.get_input_embeddings().weight
    
    non_zero_embeddings = embedding_matrix[~torch.all(torch.abs(embedding_matrix) < 1e-10, dim=1)]
    
    # Get statistics of non-zero embeddings
    non_zero_embeddings_mean = non_zero_embeddings.mean().detach().item()
    non_zero_embeddings_var = non_zero_embeddings.var().detach().item()
    
    # Format system prefix based on model type
    model_name = model.config._name_or_path.lower()
    
    if "qwen" in model_name.lower():
        system_prefix = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n\n"
        completion_prefix = "\n<|im_end|>\n<|im_start|>assistant\n\n"
    elif "llama-3" in model_name.lower():
        system_prefix = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        completion_prefix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "command" in model_name.lower():
        system_prefix = f"{system_prompt}<|START_OF_TURN_TOKEN|><|USER_TOKEN|>\n\n"
        completion_prefix = "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|ASSISTANT_TOKEN|>\n\n"
    else:
        # Generic formatting
        system_prefix = f"System: {system_prompt}\nUser: \n\n"
        completion_prefix = "\n\nAssistant: \n\n"
    
    # Normalize examples and query embeddings
    norm_example_embeddings = []
    for emb in example_embeddings:
        norm_emb = normalize_embedding(emb, non_zero_embeddings_mean, non_zero_embeddings_var)
        norm_example_embeddings.append(norm_emb.to(device))
    
    norm_query_embeddings = []
    for emb in query_embeddings:
        norm_emb = normalize_embedding(emb, non_zero_embeddings_mean, non_zero_embeddings_var)
        norm_query_embeddings.append(norm_emb.to(device))
    
    # Initialize the prompt text and embedding positions list
    full_prompt_text = system_prefix
    embedding_positions = []
    
    # Encode the system prefix to calculate initial position
    system_prefix_ids = tokenizer.encode(system_prefix, add_special_tokens=False)
    current_position = len(system_prefix_ids)
    
    # Add examples with their embeddings
    for i, (embedding, target) in enumerate(zip(norm_example_embeddings, example_targets)):
        target_str = f"{float(target):.2f}"
        
        # Create the example text (without the embedding)
        example_text_before_rep = "Input vector: "
        example_text_after_rep = "\nAnswer: " + target_str + "\n\n"
        
        # Calculate positions
        example_before_ids = tokenizer.encode(example_text_before_rep, add_special_tokens=False)
        rep_start_ids = tokenizer.encode("<Rep>", add_special_tokens=False)
        rep_end_ids = tokenizer.encode("</Rep>", add_special_tokens=False)
        example_after_ids = tokenizer.encode(example_text_after_rep, add_special_tokens=False)
        
        # Calculate embedding position
        embedding_start = current_position + len(example_before_ids) + len(rep_start_ids)
        # For single embedding vector, the end is start + 1
        if embedding.dim() == 1 or (embedding.dim() > 1 and embedding.shape[0] == 1):
            embedding_end = embedding_start + 1
        else:
            # For multi-dimensional embedding
            embedding_end = embedding_start + embedding.shape[0]
        
        # Store the embedding position
        embedding_positions.append((i, embedding_start, embedding_end, embedding))
        
        # Update the full prompt text
        full_prompt_text += example_text_before_rep + "<Rep> </Rep>" + example_text_after_rep
        
        # Update the current position
        current_position += len(example_before_ids) + len(rep_start_ids) + 1 + len(rep_end_ids) + len(example_after_ids)
    
    # Add all queries first (without answers)
    all_queries_text = ""
    query_embedding_start_positions = []
    
    for i, embedding in enumerate(norm_query_embeddings):
        query_num = i + 1
        
        # Create the query text (without the embedding)
        query_text_before_rep = f"Question {query_num}: given the following input vector, predict the output.\nInput vector: "
        query_text_after_rep = "\n\n"
        
        # Calculate positions relative to the all_queries_text
        query_before_ids = tokenizer.encode(query_text_before_rep, add_special_tokens=False)
        rep_start_ids = tokenizer.encode("<Rep>", add_special_tokens=False)
        rep_end_ids = tokenizer.encode("</Rep>", add_special_tokens=False)
        query_after_ids = tokenizer.encode(query_text_after_rep, add_special_tokens=False)
        
        # Calculate embedding position within the queries section
        query_text_position = len(tokenizer.encode(all_queries_text, add_special_tokens=False))
        embedding_start = query_text_position + len(query_before_ids) + len(rep_start_ids)
        
        # Store for later adjustment
        query_embedding_start_positions.append((i, embedding_start))
        
        # Add to queries text
        all_queries_text += query_text_before_rep + "<Rep> </Rep>" + query_text_after_rep
    
    # Encode the full prompt text so far
    current_prompt_ids = tokenizer.encode(full_prompt_text, add_special_tokens=False)
    
    # Add queries section
    full_prompt_text += all_queries_text
    
    # Now adjust query embedding positions to be relative to the full prompt
    full_prompt_before_queries_position = len(current_prompt_ids)
    
    for i, (query_idx, relative_start) in enumerate(query_embedding_start_positions):
        actual_start = full_prompt_before_queries_position + relative_start
        
        # Calculate the end position based on embedding dimensions
        embedding = norm_query_embeddings[query_idx]
        if embedding.dim() == 1 or (embedding.dim() > 1 and embedding.shape[0] == 1):
            actual_end = actual_start + 1
        else:
            actual_end = actual_start + embedding.shape[0]
        
        # Add to embedding positions list
        embedding_positions.append((len(example_embeddings) + query_idx, actual_start, actual_end, embedding))
    
    # Add model-specific completion token after queries and before answers
    completion_prefix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"




    full_prompt_text += completion_prefix

    # check embedding size of full_prompt_text
    # inputs_embeds = model.model.model.embed_tokens(tokenizer.encode(full_prompt_text, add_special_tokens=False, return_tensors="pt").to(device))
    # print(inputs_embeds.size())
    # torch.Size([1, 248, 3072])




    
    # Build the answers section
    answers_text = "<Answer>\n"
    for i, target in enumerate(query_targets):
        target_str = f"{float(target):.2f}"
        answers_text += f"Answer {i+1}: {target_str}\n"
    answers_text += "</Answer>\n<|eot_id|>"

    # answers_text += "<|eot_id|>"
    
    # Add the answers section to the full prompt
    full_prompt_text += answers_text

    
    
    # Encode the complete prompt
    input_ids = tokenizer.encode(full_prompt_text, add_special_tokens=False, return_tensors="pt")[0].to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    
    # Find positions of the answer section
    answer_start_tag = tokenizer.encode("<Answer>", add_special_tokens=False, return_tensors="pt")[0].to(device)
    answer_end_tag = tokenizer.encode("</Answer>", add_special_tokens=False, return_tensors="pt")[0].to(device)

    
    
    answer_start_pos = -1
    answer_end_pos = -1
    
    # Find the start and end of the answer section
    for i in range(len(input_ids) - len(answer_start_tag) + 1):
        if torch.all(input_ids[i:i+len(answer_start_tag)] == answer_start_tag):
            answer_start_pos = i
            break
    
    for i in range(len(input_ids) - len(answer_end_tag) + 1):
        if torch.all(input_ids[i:i+len(answer_end_tag)] == answer_end_tag):
            answer_end_pos = i + len(answer_end_tag)
            break
    
    # Create labels tensor (-100 for ignored positions)
    labels = torch.full_like(input_ids, -100)
    
    # Set labels for the answer section only
    # set answer_start_pos: instead of answer_start_pos:answer_end_pos to include the last token
    if answer_start_pos >= 0 and answer_end_pos > answer_start_pos:
        labels[answer_start_pos:] = input_ids[answer_start_pos:]

    # add answer token
    # if answer_start_pos >= 0 and answer_end_pos > answer_start_pos:
    #     labels[answer_start_pos-1:answer_end_pos+1] = input_ids[answer_start_pos-1:answer_end_pos+1]
    
    # Create weight mask
    weight_mask = torch.zeros_like(input_ids, dtype=torch.float)
    
    # Set regular weights (1.0) for the entire answer section
    if answer_start_pos >= 0 and answer_end_pos > answer_start_pos:
        # weight_mask[answer_start_pos:answer_end_pos] = 1.0
        weight_mask[answer_start_pos:] = 1.0
    
    # Apply higher weights to the numerical values in the answer section
    if answer_start_pos >= 0:
        answer_section_text = tokenizer.decode(input_ids[answer_start_pos:answer_end_pos])
        
        # Find positions of answer values in the answer section
        for i, target in enumerate(query_targets):
            target_str = f"{float(target):.2f}"
            answer_marker = f"Answer {i+1}: {target_str}"
            
            # Find this pattern in the decoded text
            pattern_position = answer_section_text.find(answer_marker)
            if pattern_position >= 0:
                # Find the value part within this pattern
                value_position = pattern_position + len(f"{i+1}: ")
                value_text = target_str

                # print("Value position:", value_position)
                # print("Value text:", value_text)
                
                # Convert text positions to token positions
                value_tokens_prefix = tokenizer.encode(answer_section_text[:value_position], add_special_tokens=False)
                value_tokens = tokenizer.encode(value_text, add_special_tokens=False)
                
                # Calculate absolute position in the full sequence
                value_start_pos = answer_start_pos + len(value_tokens_prefix) - len(tokenizer.encode(answer_section_text[:pattern_position], add_special_tokens=False))
                value_end_pos = value_start_pos + len(value_tokens)
                
                # Apply higher weight to the value tokens
                if value_start_pos < len(weight_mask) and value_end_pos <= len(weight_mask):
                    weight_mask[value_start_pos:value_end_pos] = answer_weight
    
    # Return the formatted batch
    return {
        'input_ids_prefix': input_ids,
        'example_embeddings': [e[3] for e in embedding_positions if e[0] < len(example_embeddings)],
        'query_embeddings': [e[3] for e in embedding_positions if e[0] >= len(example_embeddings)],
        'attention_mask': attention_mask,
        'labels': labels,
        'weight_mask': weight_mask,
        'embedding_positions': embedding_positions,
        'answer_section': (answer_start_pos, answer_end_pos),
        'example_map_ids': example_map_ids if example_map_ids is not None else [-1] * len(example_embeddings),
        'query_map_ids': query_map_ids if query_map_ids is not None else [-1] * len(query_embeddings),
        'query_targets': query_targets
    }

def normalize_embedding(embedding, target_mean, target_var):
    """
    Normalize embedding mean and variance.
    
    Args:
        embedding: Input embedding tensor
        target_mean: Target mean value
        target_var: Target variance value
        
    Returns:
        Normalized embedding tensor
    """
    # Ensure embedding is 2D
    if embedding.dim() == 1:
        embedding = embedding.unsqueeze(0)
    
    # Calculate statistics
    mean = embedding.mean(-1, keepdim=True)
    var = embedding.var(-1, keepdim=True)
    
    # Normalize
    normalized = (embedding - mean) * torch.sqrt(target_var / (var + 1e-7)) + target_mean
    
    return normalized

def process_icl_embedding_batch(batch_inputs, batch_attention_mask, batch_labels, model, device=None):
    """
    Process an ICL batch for the model by inserting embeddings at the correct positions.
    
    Args:
        batch_inputs: List of batch input dictionaries
        batch_attention_mask: Batch attention mask tensor
        batch_labels: Batch labels tensor
        model: The model (for accessing embedding layer)
        device: Device to put tensors on (defaults to model's device)
        
    Returns:
        Dictionary with processed inputs for the model
    """
    if device is None:
        device = model.device
    
    # Get the maximum sequence length in this batch
    max_length = batch_attention_mask.shape[1]
    
    # Get the model's hidden dimension
    hidden_size = model.config.hidden_size
    
    # Initialize placeholder for combined embeddings
    batch_size = len(batch_inputs)
    combined_embeddings = torch.zeros(
        (batch_size, max_length, hidden_size), 
        dtype=torch.float16, 
        device=device
    )
    
    # Process each example in the batch
    for batch_idx, inputs in enumerate(batch_inputs):
        # Get the input IDs prefix
        input_ids_prefix = inputs['input_ids_prefix'].to(device)
        
        # Get the embedding layer
        try:
            embedding_layer = model.model.model.embed_tokens
        except AttributeError:
            try:
                embedding_layer = model.model.embed_tokens
            except AttributeError:
                embedding_layer = model.get_input_embeddings()
        
        # Get token embeddings for the input IDs
        with torch.no_grad():
            token_embeddings = embedding_layer(input_ids_prefix)
        
        # First, fill in with all the token embeddings
        seq_len = min(token_embeddings.shape[0], max_length)
        combined_embeddings[batch_idx, :seq_len] = token_embeddings[:seq_len]
        
        # Now, replace the embeddings at the specified positions
        for emb_idx, (idx, start_idx, end_idx, emb) in enumerate(inputs['embedding_positions']):
            if start_idx >= max_length:
                continue  # Skip if position is beyond max_length
                
            # Determine actual embedding dimension
            if emb.dim() == 1:
                # Handle 1D case
                if start_idx < max_length:
                    combined_embeddings[batch_idx, start_idx] = emb
            else:
                # Handle 2D case
                emb_len = min(emb.shape[0], max_length - start_idx)
                if emb_len > 0:
                    combined_embeddings[batch_idx, start_idx:start_idx + emb_len] = emb[:emb_len]
    
    # Return the processed inputs
    return {
        "inputs_embeds": combined_embeddings,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels
    }

class ICLEmbeddingDataset(torch.utils.data.Dataset):
    """ICL Embedding Dataset with query-then-answers format."""
    
    def __init__(
        self, 
        input_embeddings, 
        target_values,
        tokenizer,
        model,
        system_prompt,
        max_length,
        mapping_ids=None,
        answer_weight=5.0,
        num_examples=3,
        query_size=1,
    ):
        """Initialize ICL dataset with embedding inputs and target values."""
        self.input_embeddings = input_embeddings
        self.target_values = target_values
        self.tokenizer = tokenizer
        self.model = model
        self.system_prompt = system_prompt
        self.max_length = max_length
        self.mapping_ids = mapping_ids if mapping_ids is not None else [-1] * len(input_embeddings)
        self.hidden_size = model.config.hidden_size
        self.answer_weight = answer_weight
        self.num_examples = num_examples
        self.query_size = query_size
        # self.is_test = is_test
        
        # Ensure the tokenizer has the special tokens we'll need
        special_tokens = {"additional_special_tokens": ["<Rep>", "</Rep>", "<Answer>", "</Answer>"]}
        num_added = tokenizer.add_special_tokens(special_tokens)
        # logger.info(f"Added {num_added} special tokens to tokenizer")
        
        # Prepare ICL batches
        logger.info("Preparing ICL dataset batches...")
        self.icl_batches = self._prepare_icl_batches()
        logger.info(f"Prepared {len(self.icl_batches)} ICL batches")
    
    def _prepare_icl_batches(self):
        """Prepare batches of ICL examples and queries."""
        import random
        
        # Total available samples
        total_samples = len(self.input_embeddings)
        
        # Ensure we have enough samples for ICL
        if total_samples < self.num_examples + self.query_size:
            raise ValueError(f"Not enough samples for ICL format. Need at least {self.num_examples + self.query_size} samples.")
        
        # Create batches
        batches = []
        
        # Group samples by mapping ID
        mapping_id_groups = {}
        for i, map_id in enumerate(self.mapping_ids):
            if map_id not in mapping_id_groups:
                mapping_id_groups[map_id] = []
            mapping_id_groups[map_id].append(i)
        
        # Shuffle each group
        for map_id in mapping_id_groups:
            random.shuffle(mapping_id_groups[map_id])
        
        # Create batches, trying to use similar mappings
        num_batches = total_samples // (self.num_examples + self.query_size)
        for _ in range(num_batches):
            # Find mapping with most samples
            available_map_ids = [map_id for map_id, indices in mapping_id_groups.items() 
                               if len(indices) >= (self.num_examples + self.query_size)]
            
            if available_map_ids:
                # Use a single mapping for this batch
                map_id = max(available_map_ids, key=lambda x: len(mapping_id_groups[x]))
                indices = mapping_id_groups[map_id]
                
                examples = indices[:self.num_examples]
                queries = indices[self.num_examples:self.num_examples + self.query_size]
                
                # Remove used samples
                mapping_id_groups[map_id] = indices[self.num_examples + self.query_size:]
            else:
                # Mix samples from different mappings
                mixed_indices = []
                for map_id in mapping_id_groups:
                    mixed_indices.extend(mapping_id_groups[map_id])
                
                if len(mixed_indices) < self.num_examples + self.query_size:
                    # Not enough samples left
                    break
                
                random.shuffle(mixed_indices)
                examples = mixed_indices[:self.num_examples]
                queries = mixed_indices[self.num_examples:self.num_examples + self.query_size]
                
                # Remove used samples
                used_indices = set(examples + queries)
                for map_id in mapping_id_groups:
                    mapping_id_groups[map_id] = [idx for idx in mapping_id_groups[map_id] 
                                             if idx not in used_indices]
            
            # Create batch
            batch = {
                'example_indices': examples,
                'query_indices': queries
            }
            batches.append(batch)
        
        return batches
    
    def __len__(self):
        return len(self.icl_batches)
    
    def __getitem__(self, idx):
        batch = self.icl_batches[idx]
        
        # Process the batch to create ICL format
        example_indices = batch['example_indices']
        query_indices = batch['query_indices']

        return format_icl_batch_with_embeddings(
                model=self.model,
                tokenizer=self.tokenizer,
                system_prompt=self.system_prompt,
                example_embeddings=[self.input_embeddings[i] for i in example_indices],
                example_targets=[self.target_values[i] for i in example_indices],
                query_embeddings=[self.input_embeddings[i] for i in query_indices],
                query_targets=[self.target_values[i] for i in query_indices],
                max_length=self.max_length,
                answer_weight=self.answer_weight,
                example_map_ids=[self.mapping_ids[i] for i in example_indices],
                query_map_ids=[self.mapping_ids[i] for i in query_indices]
            )
        
    
    def collate_fn(self, batch):
        """Custom collate function for ICL batches."""
        batch_inputs = []
        batch_attention_masks = []
        batch_labels = []
        batch_weight_masks = []
        batch_mapping_ids = []
        batch_query_targets = []
        
        for item in batch:
            # Create a CPU version of inputs with corrected embedding_positions
            # The critical fix is here - properly converting embedding_positions tensors to CPU
            embedding_positions_cpu = []
            for pos in item['embedding_positions']:
                # Each position is (idx, start_idx, end_idx, emb)
                if isinstance(pos[3], torch.Tensor):
                    # Make sure the embedding tensor is on CPU
                    embedding_positions_cpu.append((pos[0], pos[1], pos[2], pos[3].detach().cpu()))
                else:
                    embedding_positions_cpu.append(pos)
            
            cpu_inputs = {
                'input_ids_prefix': item['input_ids_prefix'].cpu(),
                'example_embeddings': [emb.detach().cpu() for emb in item['example_embeddings']],
                'query_embeddings': [emb.detach().cpu() for emb in item['query_embeddings']],
                'attention_mask': item['attention_mask'].cpu(),
                'labels': item['labels'].cpu(),
                'weight_mask': item['weight_mask'].cpu(),
                'example_map_ids': item['example_map_ids'],
                'query_map_ids': item['query_map_ids'],
                'embedding_positions': embedding_positions_cpu,  # Use the fixed CPU version
                'answer_section': item['answer_section'],
                'query_targets': item['query_targets']
            }
            
            batch_inputs.append(cpu_inputs)
            batch_attention_masks.append(item['attention_mask'].cpu())
            batch_labels.append(item['labels'].cpu())
            batch_weight_masks.append(item['weight_mask'].cpu())
            batch_mapping_ids.append(item['query_map_ids'])  # Use query map IDs
            batch_query_targets.extend(item['query_targets'])
        
        return {
            'inputs': batch_inputs,
            'attention_mask': torch.nn.utils.rnn.pad_sequence(batch_attention_masks, batch_first=True, padding_value=0),
            'labels': torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=-100),
            'weight_masks': torch.nn.utils.rnn.pad_sequence(batch_weight_masks, batch_first=True, padding_value=0),
            'mapping_ids': batch_mapping_ids,
            'query_targets': batch_query_targets
        }

def prepare_icl_datasets(args, model, tokenizer, data_dict, system_prompt):
    """
    Prepare training and testing datasets with ICL format.
    Uses the query-then-answers format.
    
    Args:
        args: Parsed arguments including icl_num_examples and icl_query_size
        model: Language model
        tokenizer: Tokenizer
        data_dict: Dictionary containing training and testing data
        system_prompt: System prompt for the task
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    # Get ICL parameters
    num_examples = getattr(args, 'icl_num_examples', 3)
    query_size = getattr(args, 'icl_query_size', 1)
    
    logger.info(f"Creating ICL datasets with {num_examples} examples and {query_size} queries per batch")
    
    # Create datasets with ICL format
    train_dataset = ICLEmbeddingDataset(
        input_embeddings=data_dict['train_inputs'],
        target_values=data_dict['train_y'],
        tokenizer=tokenizer,
        model=model,
        system_prompt=system_prompt,
        max_length=args.max_length,
        mapping_ids=data_dict['train_map_ids'],
        answer_weight=args.answer_weight,
        num_examples=num_examples,
        query_size=query_size,
    )

    # print key of the input of the first batch
    
    
    # Create test dataset with only queries (no answers)
    test_dataset = ICLEmbeddingDataset(
        input_embeddings=data_dict['test_inputs'],
        target_values=data_dict['test_y'],
        tokenizer=tokenizer,
        model=model,
        system_prompt=system_prompt,
        max_length=args.max_length,
        mapping_ids=data_dict['test_map_ids'],
        answer_weight=args.answer_weight,
        num_examples=num_examples,
        query_size=query_size,
    )

    return train_dataset, test_dataset