"""
Data generation utilities for linear mapping tasks.
This module provides functions for generating synthetic data for LoRA fine-tuning.
"""

import torch
import numpy as np
import random
import logging
from typing import Dict, List, Tuple, Optional, Union
from sklearn.cluster import KMeans

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def sample_seeds(seed_range, num_seeds):
    """
    Sample random seeds from a specific range.
    
    Args:
        seed_range: Tuple of (min_seed, max_seed)
        num_seeds: Number of seeds to sample
        
    Returns:
        List of sampled seeds
    """
    min_seed, max_seed = seed_range
    if max_seed - min_seed + 1 < num_seeds:
        logger.warning(f"Range {seed_range} contains fewer than {num_seeds} seeds. Using all seeds in range.")
        return list(range(min_seed, max_seed + 1))
    
    return random.sample(range(min_seed, max_seed + 1), num_seeds)

def prepare_data(args, llm_embs=None):
    """
    Prepare training and testing data for multiple linear mapping tasks.
    
    Args:
        args: Configuration arguments containing:
            - embedding_dim: Dimension of embeddings
            - num_train_maps: Number of different training linear mappings
            - num_test_maps: Number of different testing linear mappings
            - samples_per_map: Number of samples per mapping
            - train_seed_range: Range of seeds for training (min, max)
            - test_seed_range: Range of seeds for testing (min, max)
            - experiment_type: Type of experiment (embedding_row, prototype, random_noise, mixed)
            - mixed_ratio: Ratio of random_noise to embedding_row samples when using mixed type
            - embedding_path: Path to pre-computed embeddings (if needed)
            - train_map_seeds: Optional specific seeds for training mappings
            - test_map_seeds: Optional specific seeds for testing mappings
    
    Returns:
        Dictionary with train/test inputs, labels, and mapping info
    """
    logger.info(f"Preparing data using {args.experiment_type} approach")
    
    embedding_dim = args.embedding_dim
    experiment_type = args.experiment_type
    samples_per_map = args.samples_per_map
    
    # Set default mixed ratio if not provided
    if experiment_type == "mixed" and not hasattr(args, "mixed_ratio"):
        args.mixed_ratio = 0.5
        logger.info(f"No mixed_ratio specified, using default value of {args.mixed_ratio}")
    
    # Process seeds for train and test mappings
    if args.train_map_seeds:
        # Use provided seeds
        train_map_seeds = [int(seed) for seed in args.train_map_seeds.split(',')]
        num_train_maps = len(train_map_seeds)
        logger.info(f"Using {num_train_maps} provided training seeds: {train_map_seeds}")
    else:
        # Sample seeds from specified range
        num_train_maps = args.num_train_maps
        min_train_seed, max_train_seed = args.train_seed_range
        train_map_seeds = sample_seeds((min_train_seed, max_train_seed), num_train_maps)
        logger.info(f"Sampled {num_train_maps} training seeds from range {args.train_seed_range}: {train_map_seeds}")
    
    if args.test_map_seeds:
        # Use provided seeds
        test_map_seeds = [int(seed) for seed in args.test_map_seeds.split(',')]
        num_test_maps = len(test_map_seeds)
        logger.info(f"Using {num_test_maps} provided testing seeds: {test_map_seeds}")
    else:
        # Sample seeds from specified range
        num_test_maps = args.num_test_maps
        min_test_seed, max_test_seed = args.test_seed_range
        test_map_seeds = sample_seeds((min_test_seed, max_test_seed), num_test_maps)
        logger.info(f"Sampled {num_test_maps} testing seeds from range {args.test_seed_range}: {test_map_seeds}")
    
    # Prepare containers for data
    all_train_inputs = []
    all_train_y = []
    all_test_inputs = []
    all_test_y = []
    all_train_map_ids = []
    all_test_map_ids = []
    
    # Load embeddings if needed
    if experiment_type in ['embedding_row', 'prototype', 'mixed']:
        if llm_embs is None:
            # args.embedding_path = args.embedding_path
            raise ValueError(f"llm_embs must be provided for {experiment_type} experiment")
        
        logger.info(f"Using llm_embs of shape {llm_embs.shape}")
    
    # Generate training data
    for map_idx, seed in enumerate(train_map_seeds):
        logger.info(f"Generating training data for mapping {map_idx+1}/{num_train_maps} (seed: {seed})")
        
        if hasattr(args, "single_w") and args.single_w:
            logger.info(f"Using shared W for all mappings (seed: {args.shared_w_seed})")
            set_seed(args.shared_w_seed)
            W = torch.randn((embedding_dim, 1))
            set_seed(seed)
        else:
            set_seed(seed)
            W = torch.randn((embedding_dim, 1))
        
        # Generate input vectors X based on experiment type
        if experiment_type == "embedding_row":
            indices = np.random.choice(llm_embs.shape[0], samples_per_map, replace=False)
            X = torch.tensor(llm_embs[indices]).float()
        
        elif experiment_type == "prototype":
            # Use K-means on a subset of embeddings to find prototypes
            sample_size = min(50000, llm_embs.shape[0])
            subset_indices = np.random.choice(llm_embs.shape[0], sample_size, replace=False)
            subset_embs = llm_embs[subset_indices]
            
            logger.info(f"Running KMeans to find {samples_per_map} prototypes...")
            kmeans = KMeans(n_clusters=samples_per_map, random_state=seed, n_init=10)
            kmeans.fit(subset_embs)
            X = torch.tensor(kmeans.cluster_centers_).float()
        
        elif experiment_type == "random_noise":
            X = torch.randn((samples_per_map, embedding_dim))
            
        elif experiment_type == "mixed":
            # Calculate number of samples for each type
            noise_samples = int(samples_per_map * args.mixed_ratio)
            emb_samples = samples_per_map - noise_samples
            
            logger.info(f"Mixed data generation: {noise_samples} random noise + {emb_samples} embedding row samples")
            
            # Generate random noise samples
            X_noise = torch.randn((noise_samples, embedding_dim))
            
            # Generate embedding row samples
            indices = np.random.choice(llm_embs.shape[0], emb_samples, replace=False)
            X_emb = torch.tensor(llm_embs[indices]).float()
            
            # Combine the two types
            X = torch.cat([X_noise, X_emb], dim=0)
        
        # Calculate outputs Y = WX
        Y = X @ W
        Y = torch.round(Y, decimals=2)
        
        # Add data with mapping IDs
        all_train_inputs.append(X)
        all_train_y.append(Y)
        all_train_map_ids.extend([map_idx] * len(X))
    
    # Generate testing data
    for map_idx, seed in enumerate(test_map_seeds):
        logger.info(f"Generating testing data for mapping {map_idx+1}/{num_test_maps} (seed: {seed})")
        
        if hasattr(args, "single_w") and args.single_w:
            set_seed(args.shared_w_seed)
            W = torch.randn((embedding_dim, 1))
            set_seed(seed)
        else:
            set_seed(seed)
            W = torch.randn((embedding_dim, 1))
        
        # Generate input vectors X based on experiment type
        if experiment_type == "embedding_row":
            indices = np.random.choice(llm_embs.shape[0], samples_per_map, replace=False)
            X = torch.tensor(llm_embs[indices]).float()
        
        elif experiment_type == "prototype":
            # Use K-means for prototypes
            sample_size = min(50000, llm_embs.shape[0])
            subset_indices = np.random.choice(llm_embs.shape[0], sample_size, replace=False)
            subset_embs = llm_embs[subset_indices]
            
            logger.info(f"Running KMeans to find {samples_per_map} prototypes...")
            kmeans = KMeans(n_clusters=samples_per_map, random_state=seed, n_init=10)
            kmeans.fit(subset_embs)
            X = torch.tensor(kmeans.cluster_centers_).float()
        
        elif experiment_type == "random_noise":
            X = torch.randn((samples_per_map, embedding_dim))
            
        elif experiment_type == "mixed":
            # Calculate number of samples for each type
            noise_samples = int(samples_per_map * args.mixed_ratio)
            emb_samples = samples_per_map - noise_samples
            
            logger.info(f"Mixed data generation: {noise_samples} random noise + {emb_samples} embedding row samples")
            
            # Generate random noise samples
            X_noise = torch.randn((noise_samples, embedding_dim))
            
            # Generate embedding row samples
            indices = np.random.choice(llm_embs.shape[0], emb_samples, replace=False)
            X_emb = torch.tensor(llm_embs[indices]).float()
            
            # Combine the two types
            X = torch.cat([X_noise, X_emb], dim=0)
        
        # Calculate outputs Y = WX
        Y = X @ W
        Y = torch.round(Y, decimals=2)
        
        # Add data with mapping IDs
        all_test_inputs.append(X)
        all_test_y.append(Y)
        all_test_map_ids.extend([map_idx] * len(X))
    
    # Combine all data
    combined_train_inputs = torch.cat(all_train_inputs)
    combined_train_y = torch.cat(all_train_y)
    combined_test_inputs = torch.cat(all_test_inputs)
    combined_test_y = torch.cat(all_test_y)
    
    # Final shuffle to mix samples from different mappings
    train_perm = torch.randperm(len(combined_train_inputs))
    test_perm = torch.randperm(len(combined_test_inputs))
    
    combined_train_inputs = combined_train_inputs[train_perm]
    combined_train_y = combined_train_y[train_perm]
    combined_test_inputs = combined_test_inputs[test_perm]
    combined_test_y = combined_test_y[test_perm]
    
    # Update map_ids to match the shuffled order
    train_map_ids = [all_train_map_ids[i] for i in train_perm]
    test_map_ids = [all_test_map_ids[i] for i in test_perm]
    
    logger.info(f"Created {len(combined_train_inputs)} training samples and {len(combined_test_inputs)} test samples")
    
    return {
        'train_inputs': combined_train_inputs,
        'train_y': combined_train_y,
        'test_inputs': combined_test_inputs,
        'test_y': combined_test_y,
        'train_map_ids': train_map_ids,
        'test_map_ids': test_map_ids,
        'train_map_seeds': train_map_seeds,
        'test_map_seeds': test_map_seeds
    }