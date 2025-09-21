import os
import pandas as pd
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
import torch
import math
import torch.nn.functional as F
import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from rdkit.Chem import rdFingerprintGenerator, MolFromSmiles
from rdkit.DataStructs import TanimotoSimilarity
from pandas.errors import EmptyDataError
import deepchem as dc
import json
import GPUtil
from tdc.multi_pred import DTI
from tdc.single_pred import ADME
from sklearn.model_selection import train_test_split
import re
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
import torch
import random
from transformers import set_seed
import hashlib
from tdc.single_pred import Tox
import os
import numpy as np


torch.set_default_dtype(torch.float16)


def zero_pad_tensor(input_tensor, target_dim):
    """
    Pad the input tensor with zeros to match the target dimension.

    Args:
        input_tensor (torch.Tensor): The input tensor to pad.
        target_dim (int): The target dimension size for the last dimension.

    Returns:
        torch.Tensor: The padded tensor.
    """
    current_dim = input_tensor.shape[-1]
    if current_dim < target_dim:
        pad_size = target_dim - current_dim
        padding = torch.zeros(*input_tensor.shape[:-1], pad_size, device=input_tensor.device)
        return torch.cat([input_tensor, padding], dim=-1)
    elif current_dim > target_dim:
        return input_tensor[..., :target_dim]
    return input_tensor


def pad_and_concatenate(tensor_list, padding_value=0):
    """
    Pad tensors in the list to the maximum sequence length and reshape into (num_samples, seq_length, original_dim).

    Args:
        tensor_list (list of torch.Tensor): List of tensors with varying sequence lengths.
        padding_value (float, optional): Value to use for padding. Defaults to 0.

    Returns:
        torch.Tensor: Tensor reshaped to (num_samples, seq_length, original_dim).
    """
    # Find the maximum sequence length among all tensors
    max_length = max(tensor.shape[0] for tensor in tensor_list)
    
    # # Ensure all tensors have the same feature dimension
    # feature_dim = tensor_list[0].shape[-1]
    # if not all(tensor.shape[1] == feature_dim for tensor in tensor_list):
    #     raise ValueError("All tensors must have the same feature dimension.")
    
    # Pad each tensor to the maximum length
    padded_tensors = [
        F.pad(tensor, (0, 0, 0, max_length - tensor.shape[0]), value=padding_value)  # Pad along the sequence dimension
        for tensor in tensor_list
    ]
    
    # Stack the padded tensors into a new dimension
    reshaped_tensor = torch.stack(padded_tensors, dim=0)  # Shape: (num_samples, max_length, feature_dim)
    
    return reshaped_tensor


def load_ppi_affinity(num_trains, num_tests, base_data_path): 
    print("in load ppi affinity")
    f_train = open(f"{base_data_path}/ppi_affinity/ppi_affinity_train.json")
    train = json.load(f_train)

    f_valid = open(f"{base_data_path}/ppi_affinity/ppi_affinity_valid.json")
    valid = json.load(f_valid)

    f_test = open(f"{base_data_path}/ppi_affinity/ppi_affinity_test.json")
    test = json.load(f_test)
    train_ids = np.array([[( entry["Value"]["primary_1"], entry["Value"]["primary_2"])] for entry in train if isinstance(entry["Value"], dict) and "primary_1" in entry["Value"] and "primary_2" in entry["Value"]])
    train_y = np.array([float(entry["Value"]["interaction"]) for entry in train if isinstance(entry["Value"], dict) and "interaction" in entry["Value"]])

    valid_ids = np.array([[(entry["Value"]["primary_1"], entry["Value"]["primary_2"])] for entry in valid if isinstance(entry["Value"], dict) and "primary_1" in entry["Value"] and "primary_2" in entry["Value"]])
    valid_y = np.array([float(entry["Value"]["interaction"]) for entry in valid if isinstance(entry["Value"], dict) and "interaction" in entry["Value"]])

    test_ids = np.array([[(entry["Value"]["primary_1"], entry["Value"]["primary_2"])] for entry in test if isinstance(entry["Value"], dict) and "primary_1" in entry["Value"] and "primary_2" in entry["Value"]])
    test_y = np.array([float(entry["Value"]["interaction"]) for entry in test if isinstance(entry["Value"], dict) and "interaction" in entry["Value"]])

    return train_ids[:num_trains], train_y[:num_trains], valid_ids[:num_tests], valid_y[:num_tests], test_ids[:num_tests], test_y[:num_tests]


def stratified_sample_test(inputs, labels, num_samples, random_seed=42):
    """
    Perform stratified sampling to maintain the label distribution.

    Args:
        inputs (numpy.ndarray): Input data.
        labels (numpy.ndarray): Corresponding labels.
        num_samples (int): Number of samples to extract.
        random_seed (int): Seed for reproducibility.

    Returns:
        numpy.ndarray: Sampled inputs.
        numpy.ndarray: Sampled labels.
    """
    if len(inputs) != len(labels):
        raise ValueError("Inputs and labels must have the same length.")

    if num_samples > len(inputs):
        print(f"Number of samples ({num_samples}) is greater than the number of data points ({len(inputs)}).")
        return inputs, labels
        # Ensure inputs and labels are 2D for stacking

    # Random sampling if the number of samples is greater than 1/3 of the data
    if num_samples > 1/3 * len(inputs):
        print("Random sampling...")
        # sampled_indices = np.random.choice(len(inputs), num_samples, replace=False)

        rng = np.random.default_rng(seed=random_seed)
        sampled_indices = rng.choice(len(inputs), num_samples, replace=False)


        sampled_inputs = np.array(inputs[sampled_indices])
        sampled_labels = np.array(labels[sampled_indices])


        return sampled_inputs, sampled_labels
    


    if inputs.ndim == 1:
        inputs = inputs.reshape(-1, 1)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    # Combine inputs and labels into a single array for easier handling
    combined = np.hstack((inputs, labels))

    # Sort by labels to ensure stratified selection
    combined = combined[combined[:, -1].argsort()]

    gap_size = len(combined) // (num_samples + 1)
    selected = combined[[i * gap_size for i in range(1, num_samples + 1)], :]

    # Shuffle if needed
    rng = np.random.default_rng(seed=random_seed)
    permutation = rng.permutation(len(selected))
    selected = selected[permutation]

    # Separate inputs and labels
    sampled_inputs, sampled_labels = selected[:, :-1], selected[:, -1]

    # convert as numpy array
    sampled_inputs = np.array(sampled_inputs)
    sampled_labels = np.array(sampled_labels)

    return sampled_inputs, sampled_labels


def load_Beta_Lactamase(num_trains, num_tests, base_data_path): 
    print("in load beta lactamase")
    f_train = open(f"{base_data_path}/beta_lactamase/beta_lactamase_train.json")
    train = json.load(f_train)

    f_valid = open(f"{base_data_path}/beta_lactamase/beta_lactamase_valid.json")
    valid = json.load(f_valid)

    f_test = open(f"{base_data_path}/beta_lactamase/beta_lactamase_test.json")
    test = json.load(f_test)

    #train_ids = np.array([[entry["Value"]["primary"]] for entry in train])
    train_ids = np.array([[entry["Value"]["primary"]] for entry in train if isinstance(entry["Value"], dict) and "primary" in entry["Value"]])
    train_y = np.array([float(entry["Value"]["scaled_effect1"]) for entry in train if isinstance(entry["Value"], dict) and "scaled_effect1" in entry["Value"]])

    #valid_ids = np.array([[entry["Value"]["primary"]] for entry in valid])
    valid_ids = np.array([[entry["Value"]["primary"]] for entry in valid if isinstance(entry["Value"], dict) and "primary" in entry["Value"]])
    valid_y = np.array([float(entry["Value"]["scaled_effect1"]) for entry in valid if isinstance(entry["Value"], dict) and "scaled_effect1" in entry["Value"]])

    #test_ids = np.array([[entry["Value"]["primary"]] for entry in test])
    test_ids = np.array([[entry["Value"]["primary"]] for entry in test if isinstance(entry["Value"], dict) and "primary" in entry["Value"]])
    test_y = np.array([float(entry["Value"]["scaled_effect1"]) for entry in test if isinstance(entry["Value"], dict) and "scaled_effect1" in entry["Value"]])

    return train_ids[:num_trains], train_y[:num_trains], valid_ids[:num_tests], valid_y[:num_tests], test_ids[:num_tests], test_y[:num_tests]


def preprocess_smiles_data(mol_fm,
                           train_inputs,
                           train_y,
                           valid_inputs,
                           valid_y,
                           test_inputs,
                           test_y,
                           illegal_smiles_file="illegal_smiles.txt",
                           base_data_path=None,
                           skip_check=False):
    """
    Preprocess SMILES datasets by removing illegal SMILES.

    Parameters:
    - mol_fm: An object containing the method get_unimol_rep_tensor to check SMILES validity.
    - train_inputs, train_y: Training SMILES and their corresponding labels.
    - valid_inputs, valid_y: Validation SMILES and their corresponding labels.
    - test_inputs, test_y: Test SMILES and their corresponding labels.
    - illegal_smiles_file: The path to the file that records illegal SMILES.
    - skip_check: If True and the illegal_smiles_file exists, skip legality checks and use the file contents.

    Returns:
    - clean_train_inputs, clean_train_y, clean_valid_inputs, clean_valid_y, clean_test_inputs, clean_test_y
      All as NumPy arrays.
    """

    # Load existing illegal SMILES if available
    illegal_smiles_file = os.path.join(base_data_path, illegal_smiles_file)
    illegal_smiles = set()
    file_exists = os.path.exists(illegal_smiles_file)
    if file_exists:
        with open(illegal_smiles_file, 'r') as f:
            illegal_smiles = set(line.strip() for line in f if line.strip())

    def is_illegal_smile(smile):
        try:
            mol_fm.get_unimol_rep_tensor(smile)
            return False
        except ValueError:
            return True

    # Function to filter a dataset split
    def filter_dataset(inputs, labels, check_smiles):
        clean_inputs = []
        clean_labels = []
        for s, y in zip(inputs, labels):
            # Check if already known as illegal
            if s in illegal_smiles:
                if check_smiles:
                    # Recheck to see if status has changed
                    if is_illegal_smile(s):
                        continue
                    else:
                        # If no longer illegal, remove from the illegal set
                        illegal_smiles.remove(s)
                        clean_inputs.append(s)
                        clean_labels.append(y)
                else:
                    # Skip if not rechecking
                    continue
            else:
                # If not in illegal_smiles, perform normal checking
                if check_smiles:
                    if is_illegal_smile(s):
                        illegal_smiles.add(s)
                    else:
                        clean_inputs.append(s)
                        clean_labels.append(y)
                else:
                    clean_inputs.append(s)
                    clean_labels.append(y)
        return clean_inputs, clean_labels

    # Decide whether to perform legality checks
    check_smiles = not (skip_check and file_exists)

    # Filter datasets
    clean_train_inputs, clean_train_y = filter_dataset(train_inputs, train_y, check_smiles)
    clean_valid_inputs, clean_valid_y = filter_dataset(valid_inputs, valid_y, check_smiles)
    clean_test_inputs, clean_test_y = filter_dataset(test_inputs, test_y, check_smiles)

    if check_smiles:
        # Write the updated illegal_smiles set to the file
        with open(illegal_smiles_file, 'w') as f:
            if len(illegal_smiles) == 0:
                # Write a message if no illegal SMILES
                f.write("# No illegal SMILES found.\n")
            else:
                for s in sorted(illegal_smiles):
                    f.write(f"{s}\n")

    # Convert to NumPy arrays
    clean_train_inputs = np.array(clean_train_inputs)
    clean_train_y = np.array(clean_train_y)
    clean_valid_inputs = np.array(clean_valid_inputs)
    clean_valid_y = np.array(clean_valid_y)
    clean_test_inputs = np.array(clean_test_inputs)
    clean_test_y = np.array(clean_test_y)

    return clean_train_inputs, clean_train_y, clean_valid_inputs, clean_valid_y, clean_test_inputs, clean_test_y

def string_to_tensor(string, shape=(3, 3), min_val=-1.0, max_val=1.0):

    if isinstance(string, np.ndarray):
        string = str(string[0])
    elif not isinstance(string, str):
        raise TypeError("Input must be a string or a NumPy array.")
    
    hash_object = hashlib.sha256(string.encode('utf-8'))
    seed = int(hash_object.hexdigest(), 16) % (2**32)

    np.random.seed(seed)

    random_values = np.random.uniform(min_val, max_val, size=shape)

    tensor = torch.tensor(random_values, dtype=torch.float16)

    return tensor

def data_split_loading_ADME(dataset_name, split_type,seed,split_fracs):
    # Inputs: dataset name (from list of TDC datasets), split type (random, scaffold), random seed, and split fractions ([training, validation, test])
    # Output: list in the following order: [list of train SMILES, list of train labels, list of valid SMILES, list of valid labels, list of test SMILES, list of test labels]
    # Load data from TDC
    data = ADME(name=dataset_name)
    # Get the splits using the type of split
    data_df = data.get_data()
    splits = data.get_split(method=split_type,seed=seed,frac=split_fracs)
    # Splits give training DF with columns of Drug IDs, SMILES strings, and labels
    # Need to extract columns and separate
    split_info = {}
    for split in splits.keys():
        split_info[split] = {}
        split_df = splits[split]
        smiles_df = split_df["Drug"]
        labels = split_df["Y"]
        split_info[split]["SMILES"] = smiles_df
        split_info[split]["Labels"] = labels
    return split_info


def data_split_loading_Tox(dataset_name, split_type,seed,split_fracs):
    # Inputs: dataset name (from list of TDC datasets), split type (random, scaffold), random seed, and split fractions ([training, validation, test])
    # Output: list in the following order: [list of train SMILES, list of train labels, list of valid SMILES, list of valid labels, list of test SMILES, list of test labels]
    # Load data from TDC
    data = Tox(name=dataset_name)
    # Get the splits using the type of split
    data_df = data.get_data()
    splits = data.get_split(method=split_type,seed=seed,frac=split_fracs)
    # Splits give training DF with columns of Drug IDs, SMILES strings, and labels
    # Need to extract columns and separate
    split_info = {}
    for split in splits.keys():
        split_info[split] = {}
        split_df = splits[split]
        smiles_df = split_df["Drug"]
        labels = split_df["Y"]
        split_info[split]["SMILES"] = smiles_df
        split_info[split]["Labels"] = labels
    return split_info

def print_label_distribution(label_counts, set_type):
    print(f"Label distribution in {set_type} set:")
    print(f"{'-'*50}")
    # Limit to first 5 labels
    for i, (label, count) in enumerate(label_counts.items()):
        if i < 5:
            print(f"Label: {label}, Samples: {count}")
        else:
            print("... More labels ...")
            break
        

def set_random_seed(random_seed):
    # Fix the random seed for Python, NumPy, and PyTorch
    np.random.seed(random_seed)  # Set the random seed for NumPy
    random.seed(random_seed)  # Set the random seed for Python's random module
    torch.manual_seed(random_seed)  # Set the random seed for CPU in PyTorch
    torch.cuda.manual_seed(random_seed)  # Set the random seed for the current GPU in PyTorch
    torch.cuda.manual_seed_all(random_seed)  # Set the random seed for all GPUs in PyTorch

    # Set cuDNN to use deterministic algorithms
    torch.backends.cudnn.deterministic = True  # Force cuDNN to use deterministic algorithms
    torch.backends.cudnn.benchmark = False  # Disable the algorithm selection to avoid non-deterministic behavior

    # Fix Python's hash randomization
    os.environ['PYTHONHASHSEED'] = str(random_seed)  # Set the PYTHONHASHSEED environment variable to ensure consistent hash values

    # Fix the random seed for Hugging Face transformers
    set_seed(random_seed)  # Set the random seed for the Hugging Face transformers library

from sklearn.neighbors import KernelDensity


def kde_resample(train_y, combined_y, combined_inputs, desired_size, kernel='gaussian'):
    """
    Perform kernel density estimation (KDE) resampling based on training labels.
    """
    train_y = train_y.reshape(-1, 1)
    kde = KernelDensity(kernel=kernel, bandwidth=0.1).fit(train_y)

    combined_y_ = combined_y.reshape(-1, 1)
    log_probs = kde.score_samples(combined_y_)
    probs = np.exp(log_probs)
    probs /= probs.sum()

    rng = np.random.default_rng(42)
    sampled_indices = rng.choice(len(combined_y), size=desired_size, replace=True, p=probs)

    sampled_inputs = combined_inputs[sampled_indices]
    sampled_labels = combined_y[sampled_indices]
    return sampled_inputs, sampled_labels

def load_Stability(num_trains, num_tests,base_data_path): 
    # cwd = os.getcwd() 
    f_train = open(f"{base_data_path}/stability/stability_train.json")
    train = json.load(f_train)

    f_valid = open(f"{base_data_path}/stability/stability_valid.json")
    valid = json.load(f_valid)

    f_test = open(f"{base_data_path}/stability/stability_test.json")
    test = json.load(f_test) #open file

    train_ids = np.array([[d["primary"]] for d in train])
    train_y = np.array([d["stability_score"] for d in train])

    valid_ids = np.array([[d["primary"]] for d in valid]) 
    valid_y = np.array([d["stability_score"] for d in valid])

    test_ids = np.array([[d["primary"]] for d in test])
    test_y = np.array([d["stability_score"] for d in test])

    return train_ids[:num_trains], train_y[:num_trains], valid_ids[:num_tests], valid_y[:num_tests], test_ids[:num_tests], test_y[:num_tests]

def load_Fluorescence(num_trains, num_tests,base_data_path): 
    print("in load fluorescence")
    # cwd = os.getcwd() 
    f_train = open(f"{base_data_path}/fluorescence/fluorescence_train.json")
    train = json.load(f_train)

    f_valid = open(f"{base_data_path}/fluorescence/fluorescence_valid.json")
    valid = json.load(f_valid)

    f_test = open(f"{base_data_path}/fluorescence/fluorescence_test.json")
    test = json.load(f_test) #open file

    train_ids = np.array([[d["primary"]] for d in train])
    train_y = np.array([d["log_fluorescence"] for d in train])

    valid_ids = np.array([[d["primary"]] for d in valid]) 
    valid_y = np.array([d["log_fluorescence"] for d in valid])

    test_ids = np.array([[d["primary"]] for d in test])
    test_y = np.array([d["log_fluorescence"] for d in test])

    return train_ids[:num_trains], train_y[:num_trains], valid_ids[:num_tests], valid_y[:num_tests], test_ids[:num_tests], test_y[:num_tests]

def sample_eval_data(test_inputs, test_y, sample_ratio=0.06, sampling_method='stratified', random_seed=42):
    """
    Perform sampling on the data, using either stratified sampling (based on class sufficiency) or random sampling.
    
    Parameters:
    - test_inputs (pd.DataFrame or np.ndarray): Feature data.
    - test_y (pd.Series or np.ndarray): Label data.
    - sample_ratio (float): Proportion of the data to sample, default is 0.06.
    - sampling_method (str): Sampling method, 'stratified' for stratified sampling, 'random' for random sampling.
    - random_seed (int): Random seed for reproducibility.
    
    Returns:
    - sampled_inputs (pd.DataFrame): Sampled feature data.
    - sampled_y (pd.DataFrame): Sampled label data.
    """
    
    # Ensure label data is in DataFrame format
    test_y = pd.DataFrame(test_y)
    
    # Random sampling path - sample directly without considering class sufficiency
    if sampling_method == 'random':
        # Perform a random sample

        # print("Random sampling...")
        # print("test_inputs.shape: ", test_inputs.shape)
        # print("test_y.shape: ", test_y.shape)

        test_inputs_sampled, _, test_y_sampled, _ = train_test_split(
            test_inputs, test_y, test_size=(1 - sample_ratio), random_state=random_seed
        )

        print("Random sampling done.")
        # print("Sampled test_inputs.shape: ", test_inputs_sampled.shape)
        # print("Sampled test_y.shape: ", test_y_sampled.shape)
        
    # Stratified sampling path
    elif sampling_method == 'stratified':
        # Get the sample count for each class
        class_counts = test_y[0].value_counts()
        
        # Separate classes with sufficient samples (count >= 2) and minority classes with fewer samples
        sufficient_classes = class_counts[class_counts >= 2].index
        insufficient_classes = class_counts[class_counts < 2].index
        
        # Apply stratified sampling for classes with sufficient samples
        mask_sufficient = test_y[0].isin(sufficient_classes)
        test_inputs_sufficient = test_inputs[mask_sufficient]
        test_y_sufficient = test_y[mask_sufficient]
        
        # Stratified sampling for classes with sufficient samples
        _, test_inputs_sampled_sufficient, _, test_y_sampled_sufficient = train_test_split(
            test_inputs_sufficient, test_y_sufficient, test_size=sample_ratio, stratify=test_y_sufficient, random_state=random_seed
        )
        
        # For minority classes, retain all samples
        mask_insufficient = test_y[0].isin(insufficient_classes)
        test_inputs_insufficient = test_inputs[mask_insufficient]
        test_y_insufficient = test_y[mask_insufficient]
        
        # Concatenate stratified sampled data with minority class samples
        test_inputs_sampled = np.concatenate([test_inputs_sampled_sufficient, test_inputs_insufficient], axis=0)
        test_y_sampled = np.concatenate([test_y_sampled_sufficient, test_y_insufficient], axis=0)

        print("Stratified sampling done.")
    
    else:
        raise ValueError("Invalid sampling method. Choose 'stratified' or 'random'.")
    
    # Convert result to DataFrame format
    sampled_inputs = pd.DataFrame(test_inputs_sampled)
    sampled_y = pd.DataFrame(test_y_sampled)
    
    return sampled_inputs, sampled_y

def align_matrix_columns(A, B):  
    # used a linear transformation to get matrix A close to the distribution of matrix B.
    A_aligned = torch.zeros_like(A)
    
    # Process each column independently
    for j in range(A.shape[1]):
        # Calculate mean and standard deviation for the j-th column of A and B
        mean_A = torch.mean(A[:, j])
        std_A = torch.std(A[:, j])
        mean_B = torch.mean(B[:, j])
        std_B = torch.std(B[:, j])
        
        # Perform a linear transformation on A's j-th column to match B's mean and std
        if std_A != 0:
            # Scale by the ratio of standard deviations and shift by the difference in means
            A_aligned[:, j] = (A[:, j] - mean_A) * (std_B / std_A) + mean_B
        else:
            # If std_A is zero (constant column), adjust only the mean
            A_aligned[:, j] = A[:, j] + (mean_B - mean_A)
    
    return A_aligned

def apply_alignment(X, scales, shifts):

    device = X.device
    scales = scales.to(device)
    shifts = shifts.to(device)
    
    return X * scales + shifts

def compute_alignment_params(A, B):
    scales = torch.zeros(A.shape[1])
    shifts = torch.zeros(A.shape[1])
    
    for j in range(A.shape[1]):
        mean_A = torch.mean(A[:, j])
        std_A = torch.std(A[:, j])
        mean_B = torch.mean(B[:, j])
        std_B = torch.std(B[:, j])
        
        if std_A != 0:
            scales[j] = std_B / std_A
            shifts[j] = mean_B - scales[j] * mean_A
        else:
            scales[j] = 0
            shifts[j] = mean_B - mean_A
    
    return scales, shifts

def compute_alignment_params_weighted(A, B, weights_B):
    """
    Compute column-wise scaling and shifting parameters to align source A to weighted target B.
    
    A: torch.Tensor, shape [N, D], source data
    B: torch.Tensor, shape [M, D], target data (e.g., token embeddings)
    weights_B: torch.Tensor, shape [M], weights for each row of B (must sum to 1)
    """
    assert A.shape[1] == B.shape[1], "Feature dimensions must match"
    assert B.shape[0] == weights_B.shape[0], "Weights must match number of rows in B"

    scales = torch.zeros(A.shape[1], device=A.device)
    shifts = torch.zeros(A.shape[1], device=A.device)

    for j in range(A.shape[1]):
        # Source mean and std
        mean_A = torch.mean(A[:, j])
        std_A = torch.std(A[:, j])

        # Weighted target mean and std
        mean_B = torch.sum(weights_B * B[:, j])
        var_B = torch.sum(weights_B * (B[:, j] - mean_B) ** 2)
        std_B = torch.sqrt(var_B)  # add epsilon for stability

        # Alignment parameters
        if std_A != 0:
            scales[j] = std_B / std_A
            shifts[j] = mean_B - scales[j] * mean_A
        else:
            scales[j] = 0
            shifts[j] = mean_B - mean_A

    return scales, shifts


def stratified_sample(fm_reps_tensor, rounded_train_y, n_samples, random_seed=42):
    """
    Perform stratified sampling based on sorted values and equal intervals
    
    Args:
        fm_reps_tensor: Feature matrix tensor
        rounded_train_y: Labels/targets
        n_samples: Number of samples to select
        random_seed: Random seed for shuffling
        
    Returns:
        example_data: Selected sample data
        example_labels: Selected sample labels
        remaining_data: Remaining data
        remaining_labels: Remaining labels
    """
    # Combine features and labels
    combined_data = np.column_stack((fm_reps_tensor, rounded_train_y))
    
    # Sort by labels
    sorted_indices = combined_data[:, -1].argsort()
    sorted_data = combined_data[sorted_indices]
    
    # Calculate gap size for stratified sampling
    gap_size = len(sorted_data) // (n_samples + 1)
    
    # Select sample indices
    sample_indices = [i * gap_size for i in range(1, n_samples + 1)]
    
    # Get sample data
    selected_samples = sorted_data[sample_indices]
    
    # Optionally shuffle the selected samples
    if random_seed is not None:
        rng = np.random.default_rng(seed=random_seed)
        permutation = rng.permutation(len(selected_samples))
        selected_samples = selected_samples[permutation]
    
    # Create mask for remaining data
    mask = np.ones(len(sorted_data), dtype=bool)
    mask[sample_indices] = False
    remaining_data = sorted_data[mask]
    
    # Split features and labels
    example_data = selected_samples[:, :-1]
    example_labels = selected_samples[:, -1]
    remaining_features = remaining_data[:, :-1]
    remaining_labels = remaining_data[:, -1]
    
    return example_data, example_labels, remaining_features, remaining_labels

    
def calculate_similarity_difference_metrics(fm_llm_rep_enc, reps):

    num_reps = len(reps)

    original_similarities = []
    for i in range(num_reps):
        for j in range(i + 1, num_reps):
            sim = F.cosine_similarity(reps[i].unsqueeze(0), reps[j].unsqueeze(0)).item()
            original_similarities.append(sim)
    

    try:
        mapped_reps = [fm_llm_rep_enc(rep.unsqueeze(0)).squeeze(0) for rep in reps]
    except:

        mapped_reps_tensor = fm_llm_rep_enc(reps)
        mapped_reps = mapped_reps_tensor.unbind(0) 

    mapped_similarities = []
    for i in range(num_reps):
        for j in range(i + 1, num_reps):
            sim = F.cosine_similarity(mapped_reps[i].unsqueeze(0), mapped_reps[j].unsqueeze(0)).item()
            mapped_similarities.append(sim)
            
    pearson_correlation, pearson_value = pearsonr(original_similarities, mapped_similarities)
    spearman_correlation, spearman_value = spearmanr(original_similarities, mapped_similarities)

    return {
        "Pearson Correlation": pearson_correlation,
        "Pearson Value": pearson_value,
        "Spearman Correlation": spearman_correlation,
        "Spearman Value": spearman_value,
    }





# def extract_answer_floats(text, num_answers=None):
#     """Extracts floating-point numbers after 'Answer *:' from LLM-generated text.

#     Args:
#         text: The string containing the LLM-generated responses.

#     Returns:
#         A list of the extracted floating-point numbers, or an empty list if none are found.
#     """

#     # Regular expression pattern to match "Answer *:" followed by a float
#     pattern = r"Answer\s*\d+:\s*(-?\d+\.?\d*)"

#     # Find all matches in the text
#     matches = re.findall(pattern, text)

#     # Convert the matched strings to floats (and handle potential errors)
#     float_answers = []
#     for i, match in enumerate(matches):
#         try:
#             float_answers.append(float(match))
#         except ValueError:
#             # Skip values that can't be converted to floats
#             print(f"Warning: Could not convert '{match}' from {i}th match to a float.")
#             pass

#     if num_answers is not None:
#         float_answers = float_answers[:num_answers]

#     return float_answers


def extract_answer_floats(text, num_answers=None):
    """Extracts floating-point numbers from various answer formats in LLM-generated text.

    Args:
        text (str): The string containing the LLM-generated responses.
        num_answers (int, optional): Maximum number of floats to extract.

    Returns:
        list: A list of extracted floating-point numbers, or an empty list if none are found.
    """

    # Patterns to match various answer formats
    patterns = [
        r"Answer\s*\d+:\s*(-?\d+\.?\d*)",  # Format: "Answer 1: 0.123"
        r"Answer\s*:\s*(-?\d+\.?\d*)",       # Format: "Answer: 0.123"
        # r"Solubility:\s*(-?\d+\.?\d*)",    # Format: "Solubility: 0.551"
        # r"Answer\s*\d+:\s*\w+:\s*(-?\d+\.?\d*)"  # Format: "Answer 1: Solubility: 0.551"
    ]

    # Collect all matches from the text
    float_answers = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for i, match in enumerate(matches):
            try:
                float_answers.append(float(match))
            except ValueError:
                print(f"Warning: Could not convert '{match}' from {i}th match to a float.")
                pass

    # Limit to the specified number of answers, if provided
    if num_answers is not None:
        float_answers = float_answers[:num_answers]

    return float_answers


def fixed_sig_fig_str_tensor(tensor, sig_figs=3, delimiter=", "):
    """Converts a tensor of floats to a string with fixed significant figures.

    Args:
        tensor: A vector of floats (e.g., "[1.2345, 9.876, 0.0056789]").
        sig_figs: The desired number of significant figures for each float.

    Returns:
        A string representing the modified vector.
    """
    if isinstance(tensor, torch.Tensor):
        vector = tensor.cpu().numpy()
    else:
        vector = tensor
    fixed_vector = [f"{x:.{sig_figs}g}" for x in vector]
    # return json.dumps(fixed_vector)
    return "[" + delimiter.join(fixed_vector) + "]"


def get_action(text):
    """
    Gets first action from text. All actions are in the format of "action: ACTION_NAME[ARG1, ARG2, ...]"
    """
    text_by_lines = text.split("\n")
    length_of_action_prefix = len("action: ")

    for line in text_by_lines:
        # Found first action.
        if line.startswith("action:"):
            line = line[length_of_action_prefix:]
            action_name = line.split("[")[0]
            args = line[line.find("[") + 1 : line.find("]")].split(",")
            return action_name, args
        else:
            continue

    raise ValueError("No action found in text.")

def replace_action_result(text, action_result):
    """
    Replaces the first hallucinated action result in text with the corresponding actual value. Drops proceeding text.
    """
    text_by_lines = text.split("\n")

    for i in range(len(text_by_lines)):
        cur_line = text_by_lines[i]
        # Found first action.
        if cur_line.startswith("action:"):
            text_by_lines = text_by_lines[: i + 1]
            text_by_lines.append(f"observation: {action_result}")
            return "\n".join(text_by_lines)

    raise ValueError("No action found in text.")


def normalize_emb_mean_var(model_input_embeddings, target_mean, target_var):
    """
    normalize the mean and variance of the embeddings
    """

    mean = model_input_embeddings.mean(-1, keepdim=True)
    var = model_input_embeddings.var(-1, keepdim=True)

    model_input_embeddings = (model_input_embeddings - mean) * torch.sqrt(target_var / var) + target_mean

    return model_input_embeddings

def inject_embeddings(
    model_input_embeddings,
    embeddings_to_inject,
    emb_start_token_embedding,
    emb_end_token_embedding,
    normalize_injection=False,
    target_mean=None,
    target_var=None,
    dummy_rep=False,
    backtranslate_rep=False,
    model=None,
    embedding_matrix=None,
    rep_mapper=None
):
    """
    Concatenates embeddings to the model input embeddings which are passed as tokens via model_input.
    Chunks/pads the action embeddings to fit the model embeddings.

    """
    assert len(embeddings_to_inject.shape) == 1, "Injected embeddings must be 1D."

    device = model_input_embeddings.device
    embeddings_to_inject = embeddings_to_inject.to(device)
    

    # print("embeddings_to_inject.shape: ", embeddings_to_inject.shape)

    if dummy_rep == True:
        concated_model_input_embeddings = torch.cat(
            (model_input_embeddings, emb_start_token_embedding.to(device), emb_end_token_embedding.to(device)), dim=1
        )

        assert (
            concated_model_input_embeddings.shape[1] != model_input_embeddings.shape[1]
        ), "Concatenated embeddings have the same shape as the original."

        # TODO: Check if this is expected, should the return value be this instead?
        return concated_model_input_embeddings
    else:

        if rep_mapper is not None:
            rep_mapper = rep_mapper.to(device)
            mapped_embeddings_to_inject = rep_mapper(embeddings_to_inject.unsqueeze(0))
            # print("mapped_embeddings_to_inject.shape: ", mapped_embeddings_to_inject.shape)
            # check num of dims and add the batch dim if needed
            if len(mapped_embeddings_to_inject.shape) == 1:
                embeddings_to_inject = mapped_embeddings_to_inject.unsqueeze(0).unsqueeze(0)
            if len(mapped_embeddings_to_inject.shape) == 2:
                embeddings_to_inject = mapped_embeddings_to_inject.unsqueeze(0)
            else:
                embeddings_to_inject = mapped_embeddings_to_inject
            # print("with rep_mapper, embeddings_to_inject.shape: ", embeddings_to_inject.shape)
        else:
            # Chunk/pad
            action_embedding_length = embeddings_to_inject.shape[0]
            model_embedding_dim = model_input_embeddings.shape[2]
            num_chunks = math.ceil(action_embedding_length / model_embedding_dim)

            action_embedding_chunks = [
                embeddings_to_inject[i * model_embedding_dim : i * model_embedding_dim + model_embedding_dim]
                for i in range(num_chunks)
            ]

            # Pad the last chunk if necessary.
            last_chunk_length = action_embedding_chunks[-1].shape[0]
            to_pad = model_embedding_dim - last_chunk_length
            action_embedding_chunks[-1] = F.pad(action_embedding_chunks[-1], (0, to_pad), "constant", 0)

            embeddings_to_inject = torch.cat(action_embedding_chunks, dim=0)

            embeddings_to_inject = embeddings_to_inject.unsqueeze(0)
            embeddings_to_inject = embeddings_to_inject.unsqueeze(0) # torch.Size([1, 1, 4096])
            print("original, embeddings_to_inject.shape: ", embeddings_to_inject.shape)

        if normalize_injection:
            embeddings_to_inject = normalize_emb_mean_var(embeddings_to_inject, target_mean, target_var)

        if backtranslate_rep:
            if embedding_matrix is None:
                embedding_matrix = model.model.embed_tokens.weight
            nearest_token_ids = find_nearest_token_ids(embeddings_to_inject.to(device=embedding_matrix.device, dtype=embedding_matrix.dtype), embedding_matrix)
            embeddings_to_inject = model.model.embed_tokens(nearest_token_ids)

        # print("inject_embeddings, model_input_embeddings.shape: ", model_input_embeddings.shape)
        concated_model_input_embeddings = torch.cat(
            (
                model_input_embeddings,
                emb_start_token_embedding.to(device),
                embeddings_to_inject.to(device),
                emb_end_token_embedding.to(device),
            ),
            dim=1,
        )

        assert (
            concated_model_input_embeddings.shape[1] != model_input_embeddings.shape[1]
        ), "Concatenated embeddings have the same shape as the original."

        # TODO: Check if this is expected, should the return value be this instead?
        return concated_model_input_embeddings


def inject_embeddings_training(
    model_input_embeddings,
    mapped_embeddings_to_inject,
    emb_start_token_embedding,
    emb_end_token_embedding,
    normalize_injection=False,
    target_mean=None,
    target_var=None,
    dummy_rep=False,
    backtranslate_rep=False,
    model=None,
    embedding_matrix=None,
):

    # assert len(mapped_embeddings_to_inject.shape) == 1, "Injected embeddings must be 1D."
    # print("mapped_embeddings_to_inject.shape: ", mapped_embeddings_to_inject.shape)

    device = model_input_embeddings.device
    mapped_embeddings_to_inject = mapped_embeddings_to_inject.to(device)

    if dummy_rep == True:
        concated_model_input_embeddings = torch.cat(
            (model_input_embeddings, emb_start_token_embedding.to(device), emb_end_token_embedding.to(device)), dim=1
        )

        assert (
            concated_model_input_embeddings.shape[1] != model_input_embeddings.shape[1]
        ), "Concatenated embeddings have the same shape as the original."

        # TODO: Check if this is expected, should the return value be this instead?
        return concated_model_input_embeddings
    else:

        if len(mapped_embeddings_to_inject.shape) == 1:
            embeddings_to_inject = mapped_embeddings_to_inject.unsqueeze(0).unsqueeze(0)
            # print("Shape after unsqueezing twice for 1D input:", embeddings_to_inject.shape)


        elif len(mapped_embeddings_to_inject.shape) == 2:
            embeddings_to_inject = mapped_embeddings_to_inject.unsqueeze(0)
        else:
            embeddings_to_inject = mapped_embeddings_to_inject

        if normalize_injection:
            embeddings_to_inject = normalize_emb_mean_var(embeddings_to_inject, target_mean, target_var)

        if backtranslate_rep:
            if embedding_matrix is None:
                embedding_matrix = model.model.embed_tokens.weight
            nearest_token_ids = find_nearest_token_ids(embeddings_to_inject.to(device=embedding_matrix.device, dtype=embedding_matrix.dtype), embedding_matrix)
            embeddings_to_inject = model.model.embed_tokens(nearest_token_ids)

        # print("inject_embeddings, model_input_embeddings.shape: ", model_input_embeddings.shape)
        concated_model_input_embeddings = torch.cat(
            (
                model_input_embeddings,
                emb_start_token_embedding.to(device),
                embeddings_to_inject.to(device),
                emb_end_token_embedding.to(device),
            ),
            dim=1,
        )

        assert (
            concated_model_input_embeddings.shape[1] != model_input_embeddings.shape[1]
        ), "Concatenated embeddings have the same shape as the original."

        # TODO: Check if this is expected, should the return value be this instead?
        return concated_model_input_embeddings


def inject_embeddings_backup(
    model_input_embeddings,
    embeddings_to_inject,
    emb_start_token_embedding,
    emb_end_token_embedding,
    normalize_injection=False,
    target_mean=None,
    target_var=None,
    dummy_rep=False,
    backtranslate_rep=False,
    model=None,
    embedding_matrix=None,
):
    """
    Concatenates embeddings to the model input embeddings which are passed as tokens via model_input.
    Chunks/pads the action embeddings to fit the model embeddings.
    """
    assert len(embeddings_to_inject.shape) == 1, "Injected embeddings must be 1D."

    device = model_input_embeddings.device

    if dummy_rep == True:
        concated_model_input_embeddings = torch.cat(
            (model_input_embeddings, emb_start_token_embedding.to(device), emb_end_token_embedding.to(device)), dim=1
        )

        assert (
            concated_model_input_embeddings.shape[1] != model_input_embeddings.shape[1]
        ), "Concatenated embeddings have the same shape as the original."

        # TODO: Check if this is expected, should the return value be this instead?
        return concated_model_input_embeddings
    else:
        # print("embeddings_to_inject.shape A: ", embeddings_to_inject.shape)

        # Chunk/pad
        action_embedding_length = embeddings_to_inject.shape[0]
        model_embedding_dim = model_input_embeddings.shape[2]
        num_chunks = math.ceil(action_embedding_length / model_embedding_dim)

        action_embedding_chunks = [
            embeddings_to_inject[i * model_embedding_dim : i * model_embedding_dim + model_embedding_dim]
            for i in range(num_chunks)
        ]

        # Pad the last chunk if necessary.
        last_chunk_length = action_embedding_chunks[-1].shape[0]
        to_pad = model_embedding_dim - last_chunk_length
        action_embedding_chunks[-1] = F.pad(action_embedding_chunks[-1], (0, to_pad), "constant", 0)

        embeddings_to_inject = torch.cat(action_embedding_chunks, dim=0)

        # print("embeddings_to_inject.shape B: ", embeddings_to_inject.shape)

        # Reshape action embeddings to fit model embeddings.
        embeddings_to_inject = embeddings_to_inject.unsqueeze(0)
        embeddings_to_inject = embeddings_to_inject.unsqueeze(0)

        # print("embeddings_to_inject.shape C: ", embeddings_to_inject.shape)
        if normalize_injection:
            embeddings_to_inject = normalize_emb_mean_var(embeddings_to_inject, target_mean, target_var)
            # print("embeddings_to_inject.shape D: ", embeddings_to_inject.shape)
            # print("embeddings_to_inject.mean(-1, keepdim=True) D: ", embeddings_to_inject.mean(-1, keepdim=True))
            # print("embeddings_to_inject.var(-1, keepdim=True) D: ", embeddings_to_inject.var(-1, keepdim=True))

        if backtranslate_rep:
            if embedding_matrix is None:
                embedding_matrix = model.model.embed_tokens.weight
            # convert embeddings to most similar token ids based on cos sim and then back to the embeddings
            # print("embeddings_to_inject.device: ", embeddings_to_inject.device)
            # print("embedding_matrix.device: ", embedding_matrix.device)
            # nearest_token_ids = find_nearest_token_id_tensor(embeddings_to_inject.to(device), embedding_matrix.to(device))
            # print("nearest_token_ids: ", nearest_token_ids)
            nearest_token_ids = find_nearest_token_ids(embeddings_to_inject.to(device=embedding_matrix.device, dtype=embedding_matrix.dtype), embedding_matrix)
            # print("nearest_token_ids_2: ", nearest_token_ids_2)
            # print("backtranslate_rep, nearest_token_ids.shape: ", nearest_token_ids.shape)
            embeddings_to_inject = model.model.embed_tokens(nearest_token_ids)
            # print("backtranslate_rep, embeddings_to_inject.shape: ", embeddings_to_inject.shape)

        concated_model_input_embeddings = torch.cat(
            (
                model_input_embeddings,
                emb_start_token_embedding.to(device),
                embeddings_to_inject.to(device),
                emb_end_token_embedding.to(device),
            ),
            dim=1,
        )

        assert (
            concated_model_input_embeddings.shape[1] != model_input_embeddings.shape[1]
        ), "Concatenated embeddings have the same shape as the original."

        # TODO: Check if this is expected, should the return value be this instead?
        return concated_model_input_embeddings

    # bug:
    # return model_input_embeddings


def concatenate_text_to_embeddings(embeddings, text, model, tokenizer):
    """
    Concatenates text to the model input embeddings which are passed as tokens via model_input.
    """
    text_model_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(embeddings.device)
    
    try:
        text_embeddings = model.model.embed_tokens(
                text_model_ids
            ).to(embeddings.device)
    except:
        text_embeddings = model.base_model.model.model.embed_tokens(
                text_model_ids
            ).to(embeddings.device)
        
        # print("text_embeddings.device: ", text_embeddings.device)
        
    concated_embeddings = torch.cat(
        (
            embeddings,
            text_embeddings,

        ),
        dim=1,
    )

    # if True:
    #     print("text_model_ids A: ", text_model_ids)
    #     decoded_back = tokenizer.batch_decode(text_model_ids, skip_special_tokens=True)
    #     print("decoded_back: ", decoded_back)

    assert (
        concated_embeddings.shape[1] != embeddings.shape[1]
    ), "Concatenated embeddings have the same shape as the original."

    return concated_embeddings



import re

def extract_qa_choice_from_response(response: str) -> str:
    """
    Extracts the first answer choice (A/B/C/D) found in the response text,
    optionally preceded by 'Answer:' or other text. Ignores case and extra spacing.

    Args:
        response (str): The full generated response text.

    Returns:
        str: The extracted answer choice ('A', 'B', 'C', or 'D'), or empty string if not found.
    """
    pattern = r"(?:Answer:\s*)?([ABCD])\b"
    match = re.search(pattern, response, flags=re.IGNORECASE)

    if match:
        return match.group(1).upper()
    else:
        return ""



def extract_longest_caption_from_response(response: str) -> str:
    """
    Extracts all caption blocks between ---BEGIN DESCRIPTION--- and ---END DESCRIPTION---,
    and returns the longest one (after removing the 'Description:' prefix).
    
    Args:
        response (str): The full generated response text.

    Returns:
        str: The longest extracted caption, or empty string if none found.
    """

    pattern = r"---BEGIN DESCRIPTION---\s*(?:Description:\s*)?(.*?)\s*---END DESCRIPTION---"
    match = re.search(pattern, response, re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        return ""




def export_dialogue(dialogue, output_path, check_filename_length=False):
    user_turns = [dialogue[i] for i in range(len(dialogue)) if i % 2 == 0]
    llm_turns = [dialogue[i] for i in range(len(dialogue)) if i % 2 == 1]
    llm_turns = [llm_turn.replace("\n", "\n\t\t") for llm_turn in llm_turns]

    # Pad user turns or llm turns if necessary to match the longer one.
    if len(user_turns) > len(llm_turns):
        llm_turns += [""] * (len(user_turns) - len(llm_turns))
    elif len(llm_turns) > len(user_turns):
        user_turns += [""] * (len(llm_turns) - len(user_turns))

    dialogue = [f"{user_turn}\n\t\t{llm_turn}" for user_turn, llm_turn in zip(user_turns, llm_turns)]

    if check_filename_length:
        output_path = shorten_filename(output_path, max_length=250, truncate_phase="_trunc")

    # print("dialogue: ", dialogue)

    parent_dir = os.path.dirname(output_path)  # Extract parent directory
    os.makedirs(parent_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(dialogue))


def round_sig_figs(x: float, precision: int):
    """
    Rounds a number to number of significant figures
    """
    vectorized_round = np.vectorize(lambda x, d: round(x, d))
    return vectorized_round(x, -np.floor(np.log10(abs(x))).astype(int) + (precision - 1))

def flatten_mixed_list(data):
    """
    Flattens a list containing a mix of NumPy arrays, lists, and floats into a single list of floats.

    Args:
        data: The list to be flattened.

    Returns:
        A flattened list of floats.
    """

    flattened = []
    for item in data:
        if isinstance(item, np.ndarray):
            flattened.extend(item.tolist())
        elif isinstance(item, list):
            flattened.extend(flatten_mixed_list(item))  # Recursively flatten nested lists
        elif isinstance(item, float):
            flattened.append(item)

        elif isinstance(item, torch.Tensor):
            flattened.append(item.detach().cpu().numpy().tolist()) # 转换 Tensor 为 NumPy 数组,再转换为列表
        else:
            raise TypeError(f"Unsupported data type: {type(item)}")
    return flattened

def calculate_metrics(predictions, labels):
    """Calculates RMSE, Pearson, and Spearman correlations between two lists.

    Args:
        predictions: A list of predicted values.
        labels: A list of true (ground truth) values.

    Returns:
        A dictionary containing the RMSE, Pearson correlation, and Spearman correlation.
    """

    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have the same length.")
    
    predictions = np.array(predictions)
    labels = np.array(labels)

    rmse = np.sqrt(mean_squared_error(labels, predictions))
    pearson_corr, pearson_pvalue = pearsonr(labels, predictions)
    spearman_corr, spearman_pvalue = spearmanr(labels, predictions)

    return {
        "rmse": rmse,
        "pearson_correlation": pearson_corr,
        "pearson_pvalue": pearson_pvalue,
        "spearman_correlation": spearman_corr,
        "spearman_pvalue": spearman_pvalue
    }

def calculate_binned_metrics(y_labels, y_predictions, num_bins, cut_offs_for_digitalization, bin_to_float_map_middle=None, bin_to_float_map_median=None, clip_bin_nums=True):
    binned_metrics = {}
    # calculate binned accuracy: accuracy within each bin
    # first, convert the y_labels into bin numbers using cut_offs
    y_labels_binned = np.digitize(y_labels, cut_offs_for_digitalization)
    # y_predictions is already binned, so we can use it directly, just convert them to int
    # print("y_labels_binned: ", y_labels_binned)
    # print("y_predictions: ", y_predictions)
    if isinstance(y_predictions, list):
        y_predictions = np.array(y_predictions)
    y_predictions_binned = y_predictions.astype(int) 
    # print("y_predictions_binned: ", y_predictions_binned)
    if clip_bin_nums:
        # y_labels_binned = np.clip(y_labels_binned, 1, num_bins)
        y_predictions_binned = np.clip(y_predictions_binned, 1, num_bins)

    if bin_to_float_map_middle is not None:
        # print("bin_to_float_map_middle: ", bin_to_float_map_middle)
        y_predictions_bin_to_float_middle = np.array([bin_to_float_map_middle[bin_num-1] for bin_num in y_predictions_binned])
        binned_metrics['bin_to_float_middle_metric'] = calculate_metrics(y_predictions_bin_to_float_middle, y_labels)

    if bin_to_float_map_median is not None:
        # print("bin_to_float_map_median: ", bin_to_float_map_median)
        y_predictions_bin_to_float_median = np.array([bin_to_float_map_median[bin_num-1] for bin_num in y_predictions_binned])
        binned_metrics['bin_to_float_median_metric'] = calculate_metrics(y_predictions_bin_to_float_median, y_labels)


    # compute the accuracy by comparing the binned y_labels and y_predictions
    binned_accuracy = {}
    for i in range(1, num_bins+1):
        bin_mask = y_labels_binned == i
        bin_accuracy = np.mean(y_labels_binned[bin_mask] == y_predictions_binned[bin_mask])
        binned_accuracy[i] = bin_accuracy

    binned_metrics["binned_accuracy_by_bin"] = binned_accuracy

    # compute total accuracy
    total_accuracy = np.mean(y_labels_binned == y_predictions_binned)
    binned_metrics["binned_accuracy"] = total_accuracy

    return binned_metrics


def shorten_filename(filepath, max_length=255, truncate_phase=None):
    """Shortens a filename if it exceeds the max_length, preserving extension.

    Args:
        filepath (str): The full path to the file.
        max_length (int, optional): The maximum allowed filename length. Defaults to 255.

    Returns:
        str: The original filepath if it's short enough, otherwise a shortened version.
    """
    
    # Basic checks to avoid errors
    if not isinstance(filepath, str) or not filepath:
        raise ValueError("Invalid filepath provided")

    directory, filename = os.path.split(filepath)
    base_name, extension = os.path.splitext(filename)

    # Check if shortening is needed
    if len(filename) <= max_length:
        return filepath

    # Shorten while preserving extension
    chars_to_remove = len(filename) - max_length + len(extension) 
    if truncate_phase is not None:    
        shortened_name = base_name[:-chars_to_remove] + truncate_phase + extension  
    else:
        shortened_name = base_name[:-chars_to_remove] + extension  

    # Construct the full path with the shortened filename
    shortened_filepath = os.path.join(directory, shortened_name)

    return shortened_filepath

def save_json_to_file(data, filename="data.json", indent=4):
    """Saves a Python object (e.g., dictionary, list) as a JSON file.

    Args:
        data: The Python object to be saved as JSON.
        filename (optional): The name of the file to create (defaults to "data.json").
        indent (optional): The indentation level for formatting the JSON (defaults to 4).
    """

    with open(filename, "w") as json_file:
        json.dump(data, json_file, indent=indent)

def process_ESOL_results(prediction_label_pairs, total_num_tests, example_labels, example_smiles, test_smiles):
    total_num_preds = prediction_label_pairs.shape[0]

    # Compute the RMSE.
    rmse = np.sqrt(np.mean((prediction_label_pairs[:, 0] - prediction_label_pairs[:, 1]) ** 2))

    # ### Copying Examples
    # Compute the total number of preds that match preds in examples.
    total_copy_count = 0
    # Compute the normalized number of times each pred is copied from the examples.
    pred_wise_copy_count = {label: 0 for label in example_labels}
    # Compute whether the copied example is the example with the closest label.
    closest_to_pred_count = 0
    # Compute whether the copied example is the example with the closest SMILES.
    closest_to_smile_count = 0

    # Prepare for Tanimoto distance
    rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator()
    molecules = [MolFromSmiles(smile) for smile in example_smiles]
    example_fgrps = [rdkit_gen.GetFingerprint(mol) for mol in molecules]

    for test_smile, test_pred, test_label in zip(
        test_smiles, prediction_label_pairs[:, 0], prediction_label_pairs[:, 1]
    ):
        if test_pred in example_labels:
            pred_wise_copy_count[test_pred] += 1
            total_copy_count += 1
            if test_pred == example_labels[np.argmin(np.abs(example_labels - test_label))]:
                closest_to_pred_count += 1

            test_fgrp = rdkit_gen.GetFingerprint(MolFromSmiles(test_smile))
            tanimoto_distances = [TanimotoSimilarity(test_fgrp, example_fgrp) for example_fgrp in example_fgrps]
            if test_pred == example_labels[np.argmax(tanimoto_distances)]:
                closest_to_smile_count += 1

    if total_copy_count == 0:
        normalized_pred_wise_copy_count = {label: None for label in example_labels}
        normalized_closest_to_label_count = None
        normalized_closest_to_smile_count = None
    else:
        normalized_pred_wise_copy_count = {
            label: count / total_copy_count for label, count in pred_wise_copy_count.items()
        }
        normalized_closest_to_label_count = closest_to_pred_count / total_copy_count
        normalized_closest_to_smile_count = closest_to_smile_count / total_copy_count

    # Compute the number of failed tests.
    failed_test_count = total_num_tests - len(prediction_label_pairs)
    failed_test_ratio = failed_test_count / total_num_tests

    results = {
        "rmse": rmse,
        "matching examples ratio": total_copy_count / total_num_preds,
        "normalized matching examples by pred": normalized_pred_wise_copy_count,
        "normalized closest to label ratio": normalized_closest_to_label_count,
        "normalized closest to smile ratio": normalized_closest_to_smile_count,
        "failed test ratio": failed_test_ratio,
    }

    return results


def update_results_csv(results):
    """
    Updates the results.csv file with the results of the current experiment.

    Args:
        results (dict): The results of the current experiment.
    """
    results = pd.DataFrame([results])
    if "acc" in results.columns:
        results_name = "binned_results.csv"
    else:
        results_name = "base_results.csv"
    try:
        cur_results_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "./../logs/", results_name), index_col=0)
        cur_results_df = cur_results_df._append(results)
    except (EmptyDataError, FileNotFoundError):
        cur_results_df = results
    cur_results_df.to_csv(os.path.join(os.path.dirname(__file__), "./../logs/", results_name))


class LineBreakStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        # HACK: Llama2 tokenizer adds SPIECE_UNDERLINE token despite setting add_special_tokens=False. This is a workaround.
        # stop_token_ids is 0d after squeeze for 1 token, 1d for multiple tokens
        stop_token_ids = tokenizer("\n", return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze()
        if len(stop_token_ids.shape) != 0:
            stop_token_ids = stop_token_ids[1:]
            self.stop_token_ids = [torch.LongTensor(x).to("cuda:0") for x in stop_token_ids]
        else:
            self.stop_token_ids = [torch.LongTensor(stop_token_ids).to("cuda:0")]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in self.stop_token_ids:
            if torch.equal(input_ids[0][-1], stop_ids):
                return True
        return False


def get_stopping_criteria(tokenizer):
    return StoppingCriteriaList([LineBreakStoppingCriteria(tokenizer)])


def load_ESOL():
    save_path = os.path.join(os.path.dirname(__file__), "./../datasets/delaney")
    _, datasets, _ = dc.utils.load_dataset_from_disk(save_path)
    train, valid, test = datasets

    train_ids = train.ids.reshape(-1, 1)
    train_y = train.y

    valid_ids = valid.ids.reshape(-1, 1)
    valid_y = valid.y

    test_ids = test.ids.reshape(-1, 1)
    test_y = test.y

    return train_ids, train_y, valid_ids, valid_y, test_ids, test_y


def load_DTI(name='BindingDB_Ki', split_method=None, split_features=None, max_smiles_length=None, max_protein_length=None, max_len_on_test_only=False, seed=42):
    data = DTI(name = name)
    data.convert_to_log(form = 'binding')
    
    splits = data.get_split(method=split_method, column_name=split_features, seed=seed)

    train_df, valid_df, test_df = splits['train'], splits['valid'], splits['test']

    # filter out rows with smiles or protein sequences that are too long
    if max_smiles_length is not None:
        # print("A train_df: ", train_df)
        # print("A remove from train_df: ", train_df[train_df['Drug'].str.len() > max_smiles_length])
        if not max_len_on_test_only:
            train_df = train_df[train_df['Drug'].str.len() <= max_smiles_length]
            # print("B train_df: ", train_df)
            valid_df = valid_df[valid_df['Drug'].str.len() <= max_smiles_length]
        test_df = test_df[test_df['Drug'].str.len() <= max_smiles_length]
    
    if max_protein_length is not None:
        # print("A train_df prot: ", train_df)
        # print("A remove from train_df prot: ", train_df[train_df['Target'].str.len() > max_protein_length])
        if not max_len_on_test_only:
            train_df = train_df[train_df['Target'].str.len() <= max_protein_length]
            # print("B train_df prot: ", train_df)
            valid_df = valid_df[valid_df['Target'].str.len() <= max_protein_length]
        test_df = test_df[test_df['Target'].str.len() <= max_protein_length]

    train_inputs, train_y = train_df[['Drug', 'Target']], train_df[['Y']]
    valid_inputs, valid_y = valid_df[['Drug', 'Target']], valid_df[['Y']]
    test_inputs, test_y = test_df[['Drug', 'Target']], test_df[['Y']]

    return train_inputs, train_y, valid_inputs, valid_y, test_inputs, test_y


def load_DTI_with_mode(name='BindingDB_Ki',mode = 'max_affinity', split_method=None, split_features=None, max_smiles_length=None, max_protein_length=None, max_len_on_test_only=False, seed=42):
    data = DTI(name = name)
    data.convert_to_log(form = 'binding')

    data.harmonize_affinities(mode = mode)
# data.harmonize_affinities(mode = 'mean')
    
    splits = data.get_split(method=split_method, column_name=split_features, seed=seed)

    train_df, valid_df, test_df = splits['train'], splits['valid'], splits['test']

    # filter out rows with smiles or protein sequences that are too long
    if max_smiles_length is not None:
        # print("A train_df: ", train_df)
        # print("A remove from train_df: ", train_df[train_df['Drug'].str.len() > max_smiles_length])
        if not max_len_on_test_only:
            train_df = train_df[train_df['Drug'].str.len() <= max_smiles_length]
            # print("B train_df: ", train_df)
            valid_df = valid_df[valid_df['Drug'].str.len() <= max_smiles_length]
        test_df = test_df[test_df['Drug'].str.len() <= max_smiles_length]
    
    if max_protein_length is not None:
        # print("A train_df prot: ", train_df)
        # print("A remove from train_df prot: ", train_df[train_df['Target'].str.len() > max_protein_length])
        if not max_len_on_test_only:
            train_df = train_df[train_df['Target'].str.len() <= max_protein_length]
            # print("B train_df prot: ", train_df)
            valid_df = valid_df[valid_df['Target'].str.len() <= max_protein_length]
        test_df = test_df[test_df['Target'].str.len() <= max_protein_length]

    train_inputs, train_y = train_df[['Drug', 'Target']], train_df[['Y']]
    valid_inputs, valid_y = valid_df[['Drug', 'Target']], valid_df[['Y']]
    test_inputs, test_y = test_df[['Drug', 'Target']], test_df[['Y']]

    return train_inputs, train_y, valid_inputs, valid_y, test_inputs, test_y


def load_dataset(dataset_name, use_cache=True):
    if dataset_name == "delaney":
        if use_cache:
            _, datasets, _ = dc.utils.load_dataset_from_disk(
                os.path.join(os.path.dirname(__file__), "../datasets/delaney/")
            )
        else:
            _, datasets, _ = dc.molnet.load_delaney(
                splitter="scaffold", save_dir=os.path.join(os.path.dirname(__file__), "../datasets/delaney/")
            )

        train, valid, test = datasets

        train_ids = train.ids.reshape(-1)
        train_y = train.y.reshape(-1)
        valid_ids = valid.ids.reshape(-1)
        valid_y = valid.y.reshape(-1)
        test_ids = test.ids.reshape(-1)
        test_y = test.y.reshape(-1)

    elif dataset_name in ["BBB_Martins", "Caco2_Wang", "Lipophilicity_AstraZeneca","Clearance_Hepatocyte_AZ","Clearance_Microsome_AZ"]:
        data = ADME(name=dataset_name)
        splits = data.get_split()

        train = splits["train"]
        train_ids, train_y = train["Drug"].values, train["Y"].values

        valid = splits["valid"]
        valid_ids, valid_y = valid["Drug"].values, valid["Y"].values

        test = splits["test"]
        test_ids, test_y = test["Drug"].values, test["Y"].values

    return train_ids, train_y, valid_ids, valid_y, test_ids, test_y

# Function to find the nearest token ID for each embedding
def find_nearest_token_id(embedding, embedding_matrix):
    similarities = torch.nn.functional.cosine_similarity(embedding.unsqueeze(0), embedding_matrix)
    nearest_token_id = torch.argmax(similarities).item()
    return nearest_token_id

def find_nearest_token_ids(embeddings, embedding_matrix):
    """
    Find the nearest token ids in the embedding matrix for the given embeddings based on the cosine similarity.
    """
    embeddings_norms = torch.linalg.vector_norm(embeddings, dim=2).unsqueeze(-1)
    # print("embeddings_norms.shape: ", embeddings_norms.shape)
    embedding_matrix_norms = torch.linalg.vector_norm(embedding_matrix, dim=1).unsqueeze(0)
    # print("embedding_matrix_norms.shape: ", embedding_matrix_norms.shape)

    # print("embeddings.dtype: ", embeddings.dtype)
    # print("embedding_matrix.dtype: ", embedding_matrix.dtype)
    a_dot_b = torch.matmul(embeddings, embedding_matrix.T)

    cos_similarity = a_dot_b / (embeddings_norms * embedding_matrix_norms)

    nearest_token_ids = torch.argmax(cos_similarity, dim=2)

    return nearest_token_ids

def find_nearest_token_id_tensor(embedding, embedding_matrix):
    # Check the dimensionality of the input embedding
    if embedding.dim() == 1:
        # Single embedding vector; add sequence and batch dimensions
        embedding = embedding.unsqueeze(0).unsqueeze(0)
    elif embedding.dim() == 2:
        # Batch of embeddings; add a sequence dimension
        embedding = embedding.unsqueeze(1)
    elif embedding.dim() != 3:
        raise ValueError(
            "Embedding must be 1-D, 2-D, or 3-D, where the last two dimensions are sequence length and embedding dimension."
        )

    # print("embedding.shape: ", embedding.shape)
    # print("embedding_matrix.shape: ", embedding_matrix.shape)

    # Expand the embedding_matrix to match the batch and length dimensions of batched_embeddings
    embedding_matrix_expanded = embedding_matrix.unsqueeze(0).unsqueeze(
        0
    )  # Shape: (1, 1, num_embedding, embedding_dim)

    # Expand the batched_embeddings to have the num_embedding dimension
    embedding_expanded = embedding.unsqueeze(2)  # Shape: (batch, length, 1, embedding_dim)

    # Compute the cosine similarity along the last dimension
    similarities = torch.nn.functional.cosine_similarity(embedding_expanded, embedding_matrix_expanded, dim=-1)
    # print("similarities2.shape: ", similarities.shape)

    nearest_token_ids = torch.argmax(similarities, dim=-1)
    # print("nearest_token_ids.shape: ", nearest_token_ids.shape)

    # # Compute cosine similarity; each row in the embedding_matrix is treated as a separate token embedding
    # similarities = torch.nn.functional.cosine_similarity(embedding, embedding_matrix, dim= -1)
    # print("similarities.shape: ", similarities.shape)

    # # Find the index of the highest similarity
    # nearest_token_ids = torch.argmax(similarities, dim=-1, keepdim=True)
    # print("nearest_token_ids.shape: ", nearest_token_ids.shape)

    return nearest_token_ids

def monitor_gpu_util():
    # Monitor GPU memory usage
    for gpu in GPUtil.getGPUs():
        print(f'GPU {gpu.id}: Memory used: {gpu.memoryUsed} MB, Total: {gpu.memoryTotal} MB')

    for i in range(torch.cuda.device_count()):
        device = torch.device(f'cuda:{i}')
        torch.cuda.set_device(device)

        # ... your deep learning code for device i ...

        gpu_mem_alloc = torch.cuda.memory_allocated(i) / 1024**2  # Megabytes allocated
        gpu_mem_reserved = torch.cuda.memory_reserved(i) / 1024**2  # Megabytes reserved
        print(f'GPU {i} memory allocated: {gpu_mem_alloc:.2f} MB')
        print(f'GPU {i} memory reserved: {gpu_mem_reserved:.2f} MB')
        

def get_batch_test_example(test_smiles,test_smile_ys, round_to_int, n_sig_figs,n_full_shots, n_representation_shots,
                           use_rep,mol_fm,model, tokenizer,text_buffer,prompt_template_version,
                           llm_prompt_text_icl_template,user_prompt_embeddings_template,load_HF_llm,
                           predicted_property,mol_rep_prompt,use_dummy_rep,
                           rep_start_token_embedding, rep_end_token_embedding, embedding_matrix,
                           mol_rep_mapper):


    if isinstance(test_smiles, list): 
        test_smiles = np.array(test_smiles)  
    if isinstance(test_smile_ys, list):
        test_smile_ys = np.array(test_smile_ys)


    # for test_smile,test_smile_y in zip(test_smiles,test_smile_ys):
    if round_to_int:
        # print("train_y A: ", train_y)
        rounded_train_y = np.round(test_smile_ys).astype(int)
        # print("rounded_train_y A: ", rounded_train_y)
    elif n_sig_figs != -1:

        # Rounds a number to number of significant figures
        rounded_train_y = round_sig_figs(test_smile_ys, n_sig_figs)
    else:
        rounded_train_y = test_smile_ys
    # START of setting up ICL instructions with shot samples
    # get unimol representations of smiles
    selected_example_input_features_label_pairs = np.concatenate([test_smiles, rounded_train_y], axis=1)
    # selected_example_input_features_label_pairs = np.column_stack([test_smiles, rounded_train_y])
    

    example_features, example_labels = (
        selected_example_input_features_label_pairs[:, :-1],
        selected_example_input_features_label_pairs[:, -1],
    )
    # print("A example_features: ", example_features)
    # print("example_features.shape:", example_features.shape)
    if example_features.shape[-1] == 1:
        example_smiles = np.squeeze(example_features, axis=-1)
    # example_smile = example_features
    if use_rep:
        # print("example_smiles: ", example_smiles)
        # example_smiles: reshaphe the example_features to 1D, string type
        

        if len(example_smiles) > 0:
            unimol_reps = mol_fm.get_unimol_rep_tensor(example_smiles)
            # print("B unimol_rep after get_unimol_rep_tensor: ", unimol_reps.shape)
            # print("unimol_rep.shape: ", unimol_rep.shape)
            # unimol_reps.shape: torch.Size([20, 512])


        # print("len(example_smiles): ", len(example_smiles))
        # len(example_smiles):  20

        # make protein_reps a list of dummy values 0
        protein_reps = [0] * len(example_smiles)

    # print("example_features: ", example_features)
    # print("example_labels: ", example_labels)


    if isinstance(example_features, pd.DataFrame):
        example_features = example_features.to_numpy()
    if isinstance(example_labels, pd.DataFrame):
        example_labels = example_labels.to_numpy()


    for example_i, (example_feature, unimol_rep, label) in enumerate(zip(example_features, unimol_reps, example_labels)):
        # monitor_gpu_util()
        example_smile = example_feature
        display_label = label
        # display_label = example_labels
            
        # Segment 1/3: Text before injection.
        if prompt_template_version == 3:

            pre_injection_text = []
            pre_injection_text.extend([
                    f"\n\nQuestion: What is the {predicted_property} of the molecule?"
                ])

            if use_rep:
                pre_injection_text.append(f"{mol_rep_prompt}")

            pre_injection_text = "\n".join(pre_injection_text)
            text_buffer = f"{text_buffer}{pre_injection_text}"
            llm_prompt_text_icl_template.append(text_buffer)
            if load_HF_llm:
                user_prompt_embeddings_template = concatenate_text_to_embeddings(
                    user_prompt_embeddings_template, text_buffer, model, tokenizer
                )
            text_buffer = ""

            if prompt_template_version == 3:
                answer_name = "Answer"

            # Make use_rep work for ligand-protein DTI task
            if use_rep:
                # Segment 2/3: Embedding injection.

                if not use_dummy_rep: # use_dummy_rep=False

                    if load_HF_llm:
                            user_prompt_embeddings_template = inject_embeddings(
                                user_prompt_embeddings_template,
                                unimol_rep,
                                rep_start_token_embedding,
                                rep_end_token_embedding,

                                dummy_rep=False,
                                backtranslate_rep=False,
                                model=model,
                                embedding_matrix=embedding_matrix,
                                rep_mapper=mol_rep_mapper,
                            )


                    # Segment 3/3: Label after injection.
                #     post_injection_text = [
                #         f"\n{answer_name}: {display_label}",
                #     ]
                # else:

            post_injection_text = [
                    f"\n{answer_name}: {display_label}",
                ]

            post_injection_text = "\n".join(post_injection_text)
            text_buffer = f"{text_buffer}{post_injection_text}"

            if load_HF_llm:
                user_prompt_embeddings_template = concatenate_text_to_embeddings(
                    user_prompt_embeddings_template, text_buffer, model, tokenizer
                )
            text_buffer = ""
            # llm_prompt_text_icl_template = "".join(llm_prompt_text_icl_template)

    print("Test examples set up")
    return user_prompt_embeddings_template,text_buffer,llm_prompt_text_icl_template



def get_batch_test_example_with_others(test_smiles,test_smile_ys, round_to_int, n_sig_figs,n_full_shots, n_representation_shots,
                           use_rep,mol_fm,model, tokenizer,text_buffer,prompt_template_version,
                           llm_prompt_text_icl_template,user_prompt_embeddings_template,load_HF_llm,
                           predicted_property,mol_rep_prompt,use_dummy_rep,
                           rep_start_token_embedding, rep_end_token_embedding, embedding_matrix,
                           mol_rep_mapper):


    if isinstance(test_smiles, list): 
        test_smiles = np.array(test_smiles) 
    if isinstance(test_smile_ys, list): 
        test_smile_ys = np.array(test_smile_ys)


    # for test_smile,test_smile_y in zip(test_smiles,test_smile_ys):
    if round_to_int:
        # print("train_y A: ", train_y)
        rounded_train_y = np.round(test_smile_ys).astype(int)
        # print("rounded_train_y A: ", rounded_train_y)
    elif n_sig_figs != -1:

        # Rounds a number to number of significant figures
        rounded_train_y = round_sig_figs(test_smile_ys, n_sig_figs)
    else:
        rounded_train_y = test_smile_ys
    # START of setting up ICL instructions with shot samples
    # get unimol representations of smiles
    selected_example_input_features_label_pairs = np.concatenate([test_smiles, rounded_train_y], axis=1)
    # selected_example_input_features_label_pairs = np.column_stack([test_smiles, rounded_train_y])
    

    example_features, example_labels = (
        selected_example_input_features_label_pairs[:, :-1],
        selected_example_input_features_label_pairs[:, -1],
    )
    # print("A example_features: ", example_features)
    # print("example_features.shape:", example_features.shape)
    if example_features.shape[-1] == 1:
        example_smiles = np.squeeze(example_features, axis=-1)
    # example_smile = example_features
    if use_rep:
        print("example_smiles: ", example_smiles)
        # example_smiles: reshaphe the example_features to 1D, string type
        

        if len(example_smiles) > 0:
            unimol_reps = mol_fm.get_unimol_rep_tensor(example_smiles)
            print("B unimol_rep after get_unimol_rep_tensor: ", unimol_reps.shape)
            # print("unimol_rep.shape: ", unimol_rep.shape)
            # unimol_reps.shape: torch.Size([20, 512])


        # print("len(example_smiles): ", len(example_smiles))
        # len(example_smiles):  20

        # make protein_reps a list of dummy values 0
        protein_reps = [0] * len(example_smiles)

    # print("example_features: ", example_features)
    # print("example_labels: ", example_labels)


    if isinstance(example_features, pd.DataFrame):
        example_features = example_features.to_numpy()
    if isinstance(example_labels, pd.DataFrame):
        example_labels = example_labels.to_numpy()


    for example_i, (example_feature, unimol_rep, label) in enumerate(zip(example_features, unimol_reps, example_labels)):
        # monitor_gpu_util()
        example_smile = example_feature
        display_label = label
        # display_label = example_labels
            
        # Segment 1/3: Text before injection.
        if prompt_template_version == 3:

            pre_injection_text = []
            pre_injection_text.extend([
                    f"\n\nQuestion: What is the {predicted_property} of the molecule?"
                ])

            if use_rep:
                pre_injection_text.append(f"{mol_rep_prompt}")

            pre_injection_text = "\n".join(pre_injection_text)
            text_buffer = f"{text_buffer}{pre_injection_text}"
            llm_prompt_text_icl_template.append(text_buffer)
            if load_HF_llm:
                user_prompt_embeddings_template = concatenate_text_to_embeddings(
                    user_prompt_embeddings_template, text_buffer, model, tokenizer
                )
            text_buffer = ""

            if prompt_template_version == 3:
                answer_name = "Answer"

            # Make use_rep work for ligand-protein DTI task
            if use_rep:
                # Segment 2/3: Embedding injection.

                if not use_dummy_rep: # use_dummy_rep=False

                    if load_HF_llm:
                            user_prompt_embeddings_template = inject_embeddings(
                                user_prompt_embeddings_template,
                                unimol_rep,
                                rep_start_token_embedding,
                                rep_end_token_embedding,

                                dummy_rep=False,
                                backtranslate_rep=False,
                                model=model,
                                embedding_matrix=embedding_matrix,
                                rep_mapper=mol_rep_mapper,
                            )


                    # Segment 3/3: Label after injection.
                #     post_injection_text = [
                #         f"\n{answer_name}: {display_label}",
                #     ]
                # else:

            post_injection_text = [
                    f"\n{answer_name}: {display_label}",
                ]

            post_injection_text = "\n".join(post_injection_text)
            text_buffer = f"{text_buffer}{post_injection_text}"

            if load_HF_llm:
                user_prompt_embeddings_template = concatenate_text_to_embeddings(
                    user_prompt_embeddings_template, text_buffer, model, tokenizer
                )
            text_buffer = ""
            # llm_prompt_text_icl_template = "".join(llm_prompt_text_icl_template)

    print("Test examples set up")
    return user_prompt_embeddings_template,text_buffer,llm_prompt_text_icl_template

__all__ = [
    "extract_answer_floats",
    "fixed_sig_fig_str_tensor",
    "get_action",
    "replace_action_result",
    "normalize_emb_mean_var",
    "inject_embeddings",
    "inject_embeddings_training",
    "concatenate_text_to_embeddings",
    "export_dialogue",
    "round_sig_figs",
    "flatten_mixed_list",
    "calculate_metrics",
    "calculate_binned_metrics",
    "save_json_to_file",
    "process_ESOL_results",
    "update_results_csv",
    "get_stopping_criteria",
    "load_ESOL",
    "load_DTI",
    "find_nearest_token_id",
    "find_nearest_token_ids",
    "find_nearest_token_id_tensor",
    "monitor_gpu_util",
    "get_batch_test_example",
    "calculate_similarity_difference_metrics",
    "apply_alignment",
    # "apply_alignment2",
    "compute_alignment_params",
    "set_random_seed",
    "load_Stability",
    "load_Fluorescence",
    "data_split_loading_ADME",
    "print_label_distribution",
    "string_to_tensor",
    "load_DTI_with_mode",
    "preprocess_smiles_data",
    "load_Beta_Lactamase",
    "load_ppi_affinity",
    "stratified_sample_test",
    "pad_and_concatenate",
    "data_split_loading_Tox",
    "zero_pad_tensor",
    "compute_alignment_params_weighted",
    "kde_resample",
    "extract_longest_caption_from_response",
    "extract_qa_choice_from_response",

]
