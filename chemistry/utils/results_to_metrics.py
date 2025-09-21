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
# import deepchem as dc
import json
# import GPUtil

# from tdc.multi_pred import DTI


torch.set_default_dtype(torch.float16)

import re

def extract_answer_floats(text, num_answers=None):
    """Extracts floating-point numbers after 'Answer *:' from LLM-generated text.

    Args:
        text: The string containing the LLM-generated responses.

    Returns:
        A list of the extracted floating-point numbers, or an empty list if none are found.
    """

    # # Regular expression pattern to match "Answer *:" followed by a float
    # pattern = r"Answer\s*\d+:\s*(-?\d+\.?\d*)"

    # # Find all matches in the text
    # matches = re.findall(pattern, text)
    # print(matches)

    # # Convert the matched strings to floats (and handle potential errors)
    # float_answers = []
    # for i, match in enumerate(matches):
    #     try:
    #         float_answers.append(float(match))
    #     except ValueError:
    #         # Skip values that can't be converted to floats
    #         print(f"Warning: Could not convert '{match}' from {i}th match to a float.")
    #         pass
    float_answers = []
    indices = []
    i=0
    while i < len(text):
        j = text.find('\nAnswer: ', i)
        if j == -1:
            break
        indices.append(j)
        i = j + len('\nAnswer: ')
    for i in range(len(indices)):
        start = indices[i]+len('\nAnswer: ')
        slice = text[start:]
        str_float = slice.partition('\n')[0]
        float_answers.append(float(str_float))
    if num_answers is not None:
        float_answers = float_answers[:num_answers]

    return float_answers

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

def results_to_metrics(results_filepath):
    metrics_dict = {}
    with open(results_filepath) as results_file:
        results = json.load(results_file)
    for smiles in results.keys():
        generated_text = results[smiles]["generated"]
        labels = results[smiles]["labels"]
        print(labels)
        predictions = extract_answer_floats(generated_text,num_answers=None)
        print(predictions)
        metrics_dict[smiles] = calculate_metrics(predictions,labels)
    return metrics_dict

metrics_dict_example = results_to_metrics('/home/gridsan/dsubramanian/transformers-llmxfm/chemistry/logs/logs/results/results.json')
print(metrics_dict_example)






