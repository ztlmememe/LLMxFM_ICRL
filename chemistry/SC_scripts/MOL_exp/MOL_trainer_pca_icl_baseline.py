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

torch.set_default_dtype(torch.float16)

task_type_dict = {
    "BindingDB_Ki": "DTI",
    "BindingDB_IC50": "DTI",
    "KIBA": "DTI",
    "DAVIS": "DTI",

    "Fluorescence": "ESM",
    "Stability": "ESM",
    "Beta_Lactamase": "ESM",

    "Caco2_wang":"MOL",
    "ESOL": "MOL",
    "Solubility_AqSolDB": "MOL",
    "Half_Life_Obach": "MOL",
    "Clearance_Hepatocyte_AZ": "MOL",
    "HydrationFreeEnergy_FreeSolv": "MOL",
    "Lipophilicity_AstraZeneca": "MOL",
    "LD50_Zhu": "MOL",
}





def run(
    # llama_num_b_params=8,
    llama_num_b_params=70,
    n_full_shots=0,
    n_representation_shots=30,
    # num_tests=5,
    num_tests=-1,
    sampling_type="stratified",
    experiment_name="",
    n_sig_figs=-1,
    round_to_int=False,
    binned_regression=False,
    # bin_to_float_map_type="median", # "middle"
    num_bins=5, # 5
    bin_name_type='integer', # 5
    display_answer_choices=False,
    random_seed=42,
    instruction_prompt_name="ESOL_prompt_msr_1_guided_AC",
    batch_size=4,
    # batch_size=5,
    model_name="/home/gridsan/achan/experiments/LLMxFM/llama3/Meta-Llama-3-70B-Instruct-HF",
    llama_version=3,
    temperature=None,  # default value is 1.0
    predicted_property="Solubility",
    max_gen_length=10,
    use_ack_text=False,
    use_rep=True,
    use_react=False,
    rep_start_token="[REP]",
    rep_end_token="[/REP]",
    prompt_template_version=3,
    load_in_4bit=True,
    model_type="llama",
    batch_query_test=False,
    batch_query_size=3,
    avoid_loading_unimol_model=False,
    avoid_loading_protein_model=False,
    first_n_embeddings=-1,
    normalize_fm_rep=False,
    debug_dummy_rep=False,
    backtranslate_rep=False,
    task_name = 'BindingDB_Ki',
    split_method='random', # 'cold_split'
    split_features=None, # ['Drug', 'Target']
    # llm_quantization=True
    esm_model_name="facebook/esm2_t30_150M_UR50D",
    device_map="auto",
    only_use_exact_answer_num_match_for_eval=False,
    max_num_regen_count=3, # max number of generation retries
    # skip_input_feature_strings=False,
    use_icl_string=True,
    append_answer_format_at_end=False,
    shuffle_icl_examples=False,
    max_smiles_length=None, # 100
    max_protein_length=None, # 1000
    max_len_on_test_only=False,
    rep_dim_reduction=None,
    pca_n_components=None,
    save_train_rep_dict=True,
    num_train_rep_dim_reduction=3000,
    use_dummy_rep=False, # to study effect of phases used to include FM rep
    insert_rep_as_string=False,
    text_rep_sig_figs=3, # number of sig figs for stringified rep
    use_llm_api=False,
    llm_api_model="Gemini1.5", # "Gemini1.5", "gpt-4o", "gpt-4o-2024-05-13"
    skip_HF_model_gen=False,
    system_message_in_prompt=False,
    # omit_input_feature_strings=False, # ablate the contribution of reps to model performance
    input_feature_position="above",
    mol_rep_prompt="Use this molecular representation to answer the question: ",
    protein_rep_prompt="Use this protein representation to answer the question: ",
    skip_api_call_for_debug=False,
    arch_fm_llm_rep_enc=None,
    load_pairs = False,
    num_trains = 1000,
    file_name="", # for saving the results


):
    """
    Executes the test suite.
    """

    torch.set_grad_enabled(False)

    set_random_seed(random_seed)

    # Dataset Preparation
    if task_name in task_type_dict:
        task_type = task_type_dict[task_name] # MOL

    # General Parameters
    dialogue_output_base_path = os.path.join(os.path.dirname(__file__), f"./../logs/{task_type}/{task_name}/{file_name}")
    experiment_base_name = (
        f"{task_name}_{n_representation_shots}_shots_{batch_query_size}_batch_query"
    )
    experiment_name = f"{experiment_base_name}_{experiment_name}"
    base_data_path = os.path.join(os.path.dirname(__file__), "./../datasets/")


    if task_name == "Caco2_wang" or task_name == "Solubility_AqSolDB" or task_name == "Half_Life_Obach" or task_name == "Clearance_Hepatocyte_AZ" or task_name == "HydrationFreeEnergy_FreeSolv" or task_name == "Lipophilicity_AstraZeneca":
        data = data_split_loading_ADME(dataset_name=task_name, split_type='random', seed=0, split_fracs=[0.7, 0.1, 0.2])

        # print(data["train"])
        train_inputs, train_y, valid_inputs, valid_y, test_inputs, test_y = data["train"]["SMILES"], data["train"]["Labels"], data["valid"]["SMILES"], data["valid"]["Labels"], data["test"]["SMILES"], data["test"]["Labels"]

    elif task_name == "ESOL":
        print("Loading ESOL dataset")
        train_inputs, train_y, valid_inputs, valid_y, test_inputs, test_y = load_ESOL()

    elif task_name == "LD50_Zhu":
    
        data = data_split_loading_Tox(dataset_name=task_name, split_type='random', seed=0, split_fracs=[0.7, 0.1, 0.2])

        train_inputs, train_y, valid_inputs, valid_y, test_inputs, test_y = data["train"]["SMILES"], data["train"]["Labels"], data["valid"]["SMILES"], data["valid"]["Labels"], data["test"]["SMILES"], data["test"]["Labels"]


        
    if isinstance(train_inputs, pd.DataFrame):
        train_inputs = train_inputs.to_numpy()
        valid_inputs = valid_inputs.to_numpy()
        test_inputs = test_inputs.to_numpy()
    
    elif isinstance(train_inputs, np.ndarray):
        train_inputs = train_inputs
        valid_inputs = valid_inputs
        test_inputs = test_inputs
    else:
        try:
            train_inputs = train_inputs.to_numpy()
            valid_inputs = valid_inputs.to_numpy()
            test_inputs = test_inputs.to_numpy()
        except:
            raise ValueError("train and test inputs must be a pandas DataFrame or numpy array")

    if isinstance(train_y, pd.DataFrame):
        train_y = train_y.to_numpy()
        valid_y = valid_y.to_numpy()
        test_y = test_y.to_numpy()
    elif isinstance(train_y, np.ndarray):
        train_y = train_y
        valid_y = valid_y
        test_y = test_y
    else:
        try:
            train_y = train_y.to_numpy()
            valid_y = valid_y.to_numpy()
            test_y = test_y.to_numpy()
        except:
            raise ValueError("train and test y must be a pandas DataFrame or numpy array")

    # FM Preparation
    if use_rep:
        mol_fm = unimol_clf(avoid_loading_model=avoid_loading_unimol_model,
                            rep_cache_path=f'{base_data_path}/smile_to_rep_store.pkl')
    else:
        print("Not loading FMs")
        mol_fm = unimol_clf(avoid_loading_model=True)


    if task_name == "Solubility_AqSolDB" or task_name == "HydrationFreeEnergy_FreeSolv" and use_rep:
        train_inputs, train_y, valid_inputs, valid_y, test_inputs, test_y = preprocess_smiles_data(mol_fm, 
            train_inputs, train_y, valid_inputs, valid_y, test_inputs, test_y, base_data_path = base_data_path,skip_check=True)

    print("task_name: ", task_name)

    # print("train_inputs.dtype: ", train_inputs.dtype)
    # print("train_y.dtype: ", train_y.dtype)

    print("train_inputs.shape: ", train_inputs.shape)
    print("train_y.shape: ", train_y.shape)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if load_pairs:
        train_fm_reps_tensor = torch.load(f"{base_data_path}/{task_name}_fm_reps_tensor_train.pt")
        print("Load train fm_reps_tensor.shape: ", train_fm_reps_tensor.shape)

    else:

        fm_reps_tensor = []
        fm_inputs = []

        print("Generating FM representations for training data")

        max_train_reps = num_trains
        illegal_smiles_file = "illegal_smiles.txt"
        
        # with torch.no_grad():
        for i, train_input in tqdm(enumerate(train_inputs), total=len(train_inputs), desc="train rep"):

            if len(fm_reps_tensor) >= max_train_reps:
                break
            
            # Process the input (SMILES, optionally protein sequence)

                # check if train_input is list or np.array 
            if isinstance(train_input, list) or isinstance(train_input, np.ndarray):
                train_smile = train_input[0]
            else:
                train_smile = train_input

            try:
                unimol_rep = mol_fm.get_unimol_rep_tensor(train_smile)

                fm_inputs.append(train_smile)
                fm_reps_tensor.append(unimol_rep)

            except:
                print(f"Error: {train_smile} is an illegal SMILES string.")
                continue

        # store the unimol and protein representations and create a dataloader for training the FM representation mapper
        print("Checkpoint C, fm_reps_tensor[0].shape: ", fm_reps_tensor[0].shape)
        # Checkpoint C, fm_reps_tensor[0].shape:  torch.Size([1, 640])
        train_fm_reps_tensor = torch.cat(fm_reps_tensor, dim=0)


    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # LLM Preparation
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    load_HF_llm = not (use_llm_api and skip_HF_model_gen)
    # print("use_llm_api: ", use_llm_api)
    # print("skip_HF_model_gen: ", skip_HF_model_gen)
    print("load_HF_llm: ", load_HF_llm)

    if model_type == "c4ai-command-r-plus":
        embedding_dim = 12288
    elif model_type == "qwen2":
        if "72B" in model_name:
            embedding_dim = 8192
        else:
            embedding_dim = 3584
        print("embedding_dim: ", embedding_dim)
    elif llama_num_b_params == 70:
        embedding_dim = 8192
    elif llama_num_b_params == 13:
        embedding_dim = 5120
    elif llama_num_b_params == 7 or llama_num_b_params == 8:
        embedding_dim = 4096
    max_input_length = 4096

    with torch.no_grad():
        

        if instruction_prompt_name != None:
            instruction_path = os.path.join(os.path.dirname(__file__), f"./../prompts/{instruction_prompt_name}.txt")
            with open(instruction_path, "r") as f:
                system_message = f.read()

        if load_HF_llm:
            if load_in_4bit:
                    model = LlamaForCausalLM.from_pretrained(
                        model_name, device_map=device_map, load_in_4bit=load_in_4bit, torch_dtype=torch.bfloat16,
                        # use_auth_token=access_token,
                    )
            else:
                model = LlamaForCausalLM.from_pretrained(
                    model_name, device_map=device_map, torch_dtype=torch.bfloat16,
                    # use_auth_token=access_token,
                )  # explicit just in case
            tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
            
            embedding_matrix = model.model.embed_tokens.weight
            llm_embs = embedding_matrix.detach().cpu().to(dtype=torch.float32).to(torch.float16).numpy()
            print("llm_embs.shape: ", llm_embs.shape)
            llm_emb_dim_size = llm_embs.shape[-1]
            del llm_embs

            non_zero_embeddings_mean_mean = None
            non_zero_embeddings_var_mean = None

            # delete this func if OOM or use OT method
            if normalize_fm_rep:
                # get non-zero embeddings
                non_zero_embeddings = embedding_matrix[~torch.all(torch.abs(embedding_matrix) < 1e-10, dim=1)]
                if first_n_embeddings != -1:
                    non_zero_embeddings_to_get_stats = non_zero_embeddings[:first_n_embeddings]
                else:
                    non_zero_embeddings_to_get_stats = non_zero_embeddings

                # get stats of non-zero embeddings
                non_zero_embeddings_mean_mean = (
                    non_zero_embeddings_to_get_stats.mean(-1, keepdim=True).mean().detach().item()
                )
                non_zero_embeddings_var_mean = (
                    non_zero_embeddings_to_get_stats.var(-1, keepdim=True).mean().detach().item()
                )
                print("non_zero_embeddings_mean_mean: ", non_zero_embeddings_mean_mean)
                print("non_zero_embeddings_var_mean: ", non_zero_embeddings_var_mean)
                del non_zero_embeddings
                del non_zero_embeddings_to_get_stats

                gc.collect()
                torch.cuda.empty_cache()

            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id

            # with record_function("token_prep"):
            rep_start_token_embedding = model.model.embed_tokens(
                tokenizer.encode(rep_start_token, return_tensors="pt", add_special_tokens=False).to(
                    model.model.embed_tokens.weight.device
                )
            )
            rep_end_token_embedding = model.model.embed_tokens(
                tokenizer.encode(rep_end_token, return_tensors="pt", add_special_tokens=False).to(
                    model.model.embed_tokens.weight.device
                )
            )

            pad_token_embedding = model.model.embed_tokens(
                tokenizer.encode(tokenizer.pad_token, return_tensors="pt", add_special_tokens=False).to(
                    model.model.embed_tokens.weight.device
                )
            )


        # Example Selection
        if round_to_int:
            # print("train_y A: ", train_y)
            rounded_train_y = np.round(train_y).astype(int)
        elif n_sig_figs != -1:
            # Rounds a number to number of significant figures  
            rounded_train_y = round_sig_figs(train_y, n_sig_figs)
        else:
            rounded_train_y = train_y

        # print("train_inputs: ", train_inputs)
        # print("train_inputs.dtype: ", train_inputs.dtype)

        # print("rounded_train_y: ", rounded_train_y)
        if isinstance(train_inputs, pd.DataFrame):
            train_inputs_np = train_inputs.to_numpy()
        elif isinstance(train_inputs, np.ndarray):
            train_inputs_np = train_inputs
        elif isinstance(train_y, object):
            train_inputs_np = train_inputs.values
        else:
            raise ValueError("train_inputs must be a pandas DataFrame or numpy array")
        

        # fit PCA to unimol representations
        mol_array = train_fm_reps_tensor.cpu().numpy()

        mol_pca = PCA(n_components=pca_n_components, random_state=random_seed)  # Reduce to 50 dimensions
        train_mol_reduced_reps = mol_pca.fit_transform(mol_array)
        print("train_mol_reduced_reps.shape: ", train_mol_reduced_reps.shape)


        if binned_regression:
            pass
        
        # print("train_inputs_np.shape: ", train_inputs_np.shape)
        # print("rounded_train_y.shape: ", rounded_train_y.shape)
        if train_inputs_np.ndim == 1:
            train_inputs_np = train_inputs_np.reshape(-1, 1)  
        
        if rounded_train_y.ndim == 1:
            rounded_train_y = rounded_train_y.reshape(-1, 1)




        example_smiles_label_pairs = np.concatenate([train_inputs_np, rounded_train_y], axis=1)

        total_num_examples = n_full_shots + n_representation_shots

        # Select Examples by Stratified Sampling
        if sampling_type == "stratified":
            example_smiles_label_pairs = example_smiles_label_pairs[example_smiles_label_pairs[:, -1].argsort()]
            gap_size = len(example_smiles_label_pairs) // (total_num_examples + 1)
            selected_example_input_features_label_pairs = example_smiles_label_pairs[
                [i * gap_size for i in range(1, total_num_examples + 1)], :
            ]
            # print("A selected_example_input_features_label_pairs: ", selected_example_input_features_label_pairs)
            if shuffle_icl_examples:
                rng = np.random.default_rng(seed=random_seed)

                permutation = rng.permutation(len(selected_example_input_features_label_pairs))

                selected_example_input_features_label_pairs = selected_example_input_features_label_pairs[permutation]

        example_features, example_labels = (
            selected_example_input_features_label_pairs[:, :-1],
            selected_example_input_features_label_pairs[:, -1],
        )

        # print("example_features: ", example_features)
        if example_features.shape[-1] == 1:
            example_smiles = np.squeeze(example_features, axis=-1)

        # example_features:  [['GSGNAKAPRKTVTVAKNTVAERGFKKRAGRYFDEEAHEPHVEK']



        text_buffer = ""
        llm_prompt_text_icl_template = []

        if load_HF_llm:
            user_prompt_embeddings_template = torch.empty((1, 0, embedding_dim)).to(model.model.embed_tokens.weight.device)
        
        if not use_llm_api:
            if instruction_prompt_name != None:
                instruction_path = os.path.join(os.path.dirname(__file__), f"./../prompts/{instruction_prompt_name}.txt")
                with open(instruction_path, "r") as f:
                    system_message = f.read()

                if model_type == "c4ai-command-r-plus":
                    text_buffer = (
                        f"<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{system_message}<|END_OF_TURN_TOKEN|>"
                    )
                elif model_type == "qwen2":
                    text_buffer = (
                        f"<|im_start|>system\n{system_message}<|im_end|>\n"
                    )
                elif llama_version == 3:
                    text_buffer = (
                        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>"
                    )
                elif llama_version == 2:
                    text_buffer = f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>"

            if model_type == "c4ai-command-r-plus":
                text_buffer = f"{text_buffer}<|START_OF_TURN_TOKEN|><|USER_TOKEN|>"
            elif model_type == "qwen2":
                text_buffer = f"{text_buffer}\n<|im_start|>user\n"
            elif llama_version == 3:
                text_buffer = f"{text_buffer}<|start_header_id|>user<|end_header_id|>"

        # START of setting up ICL instructions with shot samples
        if use_rep:
            if len(example_smiles) > 0:
                unimol_reps = mol_fm.get_unimol_rep_tensor(example_smiles)

                if rep_dim_reduction == "PCA":
                    # print("example unimol_reps: ", unimol_reps)
                    # print("example unimol_reps shape: ", unimol_reps.shape)
                    example_mol_reduced_reps = mol_pca.transform(unimol_reps.cpu().numpy())
                    # print("example example_mol_reduced_reps: ", example_mol_reduced_reps)
                    print("example_mol_reduced_reps.shape: ", example_mol_reduced_reps.shape)

                    unimol_reps = torch.tensor(example_mol_reduced_reps).to(unimol_reps.device)

        else:
            unimol_reps = len(example_smiles) * [None]


                        

        if isinstance(example_features, pd.DataFrame):
            example_features = example_features.to_numpy()
        if isinstance(example_labels, pd.DataFrame):
            example_labels = example_labels.to_numpy()

        # Set up instructions before examples
        if len(text_buffer) > 0:
            llm_prompt_text_icl_template.append(text_buffer)
            if load_HF_llm:
                user_prompt_embeddings_template = concatenate_text_to_embeddings(
                    user_prompt_embeddings_template, text_buffer, model, tokenizer
                )
        text_buffer = ""


        # Monitor GPU memory usage
        # print("torch.cuda.memory_summary(): ", torch.cuda.memory_summary())
        print("GPU monitor A")
        # monitor_gpu_util()
        # non_zero_embeddings_mean_mean:  -1.3887882232666016e-05
        # non_zero_embeddings_var_mean:  8.249282836914062e-05


        for example_i, (example_feature, unimol_rep, label) in enumerate(zip(example_features, unimol_reps, example_labels)):
            
            if isinstance(example_feature, list) and len(example_feature) == 1:
                example_smile = example_feature[0]

            elif isinstance(example_feature , np.ndarray) and len(example_feature) == 1:
                example_smile = example_feature[0]

            else:
                example_smile = example_feature


            # print("example_smile: ", example_smile)
            # print("example_smile[0]: ", example_smile[0])   


            if binned_regression and bin_name_type == 'integer':
                pass
            else:
                display_label = label

            # Segment 1/3: Text before injection.
            if prompt_template_version == 3:
                if prompt_template_version == 3:
                            answer_name = "Answer"
                else:
                    answer_name = predicted_property


                if task_type == "MOL":
                    # pre_injection_text = []
                    if use_rep:
                        pre_injection_text = []
                        if use_icl_string:
                            pre_injection_text.extend([
                                f"\n\nDrug SMILES: < {example_smile} >",
                                f"Given the SMILES sequence of the drug molecule {input_feature_position}, answer the following question using the specified format.",
                                f"Question: What is the {predicted_property} of the drug molecule {input_feature_position}?",
                            ])

                            
                            pre_injection_text.append(f"{mol_rep_prompt}")
                            
                            pre_injection_text = "\n".join(pre_injection_text)

                            text_buffer = f"{text_buffer}{pre_injection_text}"
                            llm_prompt_text_icl_template.append(text_buffer)
                            if load_HF_llm:
                                user_prompt_embeddings_template = concatenate_text_to_embeddings(
                                    user_prompt_embeddings_template, text_buffer, model, tokenizer
                                )
                            text_buffer = ""


                            pre_injection_text = []
                                
                            if insert_rep_as_string:
                                # concatenate rep as string to text
                                # print("unimol_rep A: ", unimol_rep)
                                str_smile_rep = fixed_sig_fig_str_tensor(unimol_rep, text_rep_sig_figs)
                                # print("str_smile_rep A: ", str_smile_rep)
                                llm_prompt_text_icl_template.append(str_smile_rep)
                                if load_HF_llm:
                                    user_prompt_embeddings_template = concatenate_text_to_embeddings(
                                        user_prompt_embeddings_template, str_smile_rep, model, tokenizer
                                    )
                            else:
                                if load_HF_llm:
                                    user_prompt_embeddings_template = inject_embeddings(
                                        user_prompt_embeddings_template,
                                        unimol_rep,
                                        rep_start_token_embedding,
                                        rep_end_token_embedding,
                                        normalize_fm_rep,
                                        non_zero_embeddings_mean_mean,
                                        non_zero_embeddings_var_mean,
                                        dummy_rep=debug_dummy_rep,
                                        backtranslate_rep=backtranslate_rep,
                                        model=model,
                                        embedding_matrix=embedding_matrix,
                                    )

                        else:
                            pre_injection_text.extend([
                                f"\n\nQuestion: What is the {predicted_property} of the drug molecule?"
                            ])

                            # pre_injection_text.extend([
                            #     f"\n\nGiven the representations of the drug molecule, answer the following question using the specified format.",
                            #     f"Question: What is the {predicted_property} of the drug molecule?"
                            # ])

                            pre_injection_text.append(f"{mol_rep_prompt}")
                            
                            pre_injection_text = "\n".join(pre_injection_text)

                            text_buffer = f"{text_buffer}{pre_injection_text}"
                            llm_prompt_text_icl_template.append(text_buffer)
                            if load_HF_llm:
                                user_prompt_embeddings_template = concatenate_text_to_embeddings(
                                    user_prompt_embeddings_template, text_buffer, model, tokenizer
                                )
                            text_buffer = ""

                            if insert_rep_as_string:
                                str_smile_rep = fixed_sig_fig_str_tensor(unimol_rep, text_rep_sig_figs)
                                llm_prompt_text_icl_template.append(str_smile_rep)
                                if load_HF_llm:
                                    user_prompt_embeddings_template = concatenate_text_to_embeddings(
                                        user_prompt_embeddings_template, str_smile_rep, model, tokenizer
                                    )
                            else:
                                if load_HF_llm:
                                    user_prompt_embeddings_template = inject_embeddings(
                                        user_prompt_embeddings_template,
                                        unimol_rep,
                                        rep_start_token_embedding,
                                        rep_end_token_embedding,
                                        normalize_fm_rep,
                                        non_zero_embeddings_mean_mean,
                                        non_zero_embeddings_var_mean,
                                        dummy_rep=debug_dummy_rep,
                                        backtranslate_rep=backtranslate_rep,
                                        model=model,
                                        embedding_matrix=embedding_matrix,
                                    )

                    else:

                        pre_injection_text = []
                        if use_icl_string:

                            pre_injection_text.extend([
                                f"\n\nDrug SMILES: < {example_smile} >",
                                f"Given the SMILES sequence of the drug molecule {input_feature_position}, answer the following question using the specified format.",
                                f"Question: What is the {predicted_property} of the drug molecule {input_feature_position}?",
                            ])

                            
                            pre_injection_text = "\n".join(pre_injection_text)
                            text_buffer = f"{text_buffer}{pre_injection_text}"
                            llm_prompt_text_icl_template.append(text_buffer)
                            if load_HF_llm:
                                user_prompt_embeddings_template = concatenate_text_to_embeddings(
                                    user_prompt_embeddings_template, text_buffer, model, tokenizer
                                )
                            text_buffer = ""


            post_injection_text = [
                    f"\n{answer_name}: {display_label}",
                ]

            post_injection_text = "\n".join(post_injection_text)
            text_buffer = f"{text_buffer}{post_injection_text}"
            llm_prompt_text_icl_template.append(text_buffer)
            if load_HF_llm:
                user_prompt_embeddings_template = concatenate_text_to_embeddings(
                    user_prompt_embeddings_template, text_buffer, model, tokenizer
                )
            text_buffer = ""

        # if torch.isnan(user_prompt_embeddings_template).any() or torch.isinf(user_prompt_embeddings_template).any():
        #     print("Error: user_prompt_embeddings_template after contains invalid values.")

        llm_prompt_text_icl_template = "".join(llm_prompt_text_icl_template)
        print("GPU monitor B")

        # Inference
       
        output_dir_name = f"{timestamp}_{experiment_name}"

        # prediction_label_pairs = []
        # batch_queue = []
        batched_model_input_embeddings = []
        batched_user_prompts = []

        dialogue_output_path = os.path.join(dialogue_output_base_path, output_dir_name)
        # os.makedirs(os.path.join(dialogue_output_path, "trajectories"), exist_ok=True)
        os.makedirs(os.path.join(dialogue_output_path, "new_trajectories"), exist_ok=True)


        query_batch_idx = 0 # track index of batch query samples within the query
        qn_idx = 1

        if isinstance(test_inputs, pd.DataFrame):
            test_inputs = test_inputs.to_numpy()
        if isinstance(test_y, pd.DataFrame):
            test_y = test_y.to_numpy()

        batch_eval_y_labels = []
        batch_test_ids = []
        eval_y_predictions = []
        eval_y_labels = []
        eval_test_ids = []
        not_eval_test_ids = []


        if torch.isnan(user_prompt_embeddings_template).any() or torch.isinf(user_prompt_embeddings_template).any():
            print("Error: user_prompt_embeddings_template before test inject contains invalid values.")

        for test_i, (test_input, y) in tqdm(enumerate(zip(test_inputs, test_y)), total=len(test_inputs)):


            if test_i == num_tests:
                # print("BREAK due to test_i == num_tests")
                break

            at_final_test_sample = (test_i == len(test_inputs) - 1 or test_i == num_tests - 1)
            
            # collect labels for evaluation
            batch_eval_y_labels.append(y)
            batch_test_ids.append(test_i)
            
            if isinstance(test_input, list) and len(test_input) == 1:
                test_smile = test_input[0]

            elif isinstance(test_input , np.ndarray) and len(test_input) == 1:
                test_smile = test_input[0]
            else:
                test_smile = test_input

            # Initialize user prompt and model input embeddings
            if batch_query_test != True or (batch_query_test == True and query_batch_idx == 0):
                if load_HF_llm:
                    model_input_embeddings = user_prompt_embeddings_template.clone().to(
                        model.model.embed_tokens.weight.device
                    )
                user_prompt = llm_prompt_text_icl_template
                final_prompt_text_to_append = []


            if batch_query_test == True:
                qn_idx = query_batch_idx + 1


            if task_type == "MOL":
                if use_rep:
                    

                    test_text = []
                    # eg. input_feature_position="above", predicted_property="Binding Affinity Ki" predicted_property="Solubility",
                    if use_icl_string:
                        test_text.extend([
                            f"\n\nDrug SMILES: < {test_smile} >",
                            f"Given the SMILES sequence of the drug molecule {input_feature_position}, answer the following question using the specified format.",
                            f"Question {qn_idx}: What is the {predicted_property} of the drug molecule {input_feature_position}?",
                        ])


                        test_text.append(f"{mol_rep_prompt}")
                        
                        test_text = "\n".join(test_text)
                        user_prompt = f"{user_prompt}{test_text}"
                        if load_HF_llm:
                            model_input_embeddings = concatenate_text_to_embeddings(
                                model_input_embeddings, test_text, model, tokenizer
                            )

                        unimol_rep = mol_fm.get_unimol_rep_tensor(test_smile)[0].to(
                        model.model.embed_tokens.weight.device
                    )

                        if rep_dim_reduction == "PCA":

                            if len(unimol_rep.shape) == 1:
                                unimol_rep_np = unimol_rep.unsqueeze(0).cpu().numpy()
                            else:
                                unimol_rep_np = unimol_rep.cpu().numpy()
                            test_mol_reduced_rep = mol_pca.transform(unimol_rep_np)

                            if len(unimol_rep.shape) == 1:
                                unimol_rep = torch.tensor(test_mol_reduced_rep[0]).to(unimol_rep.device)
                            else:
                                unimol_rep = torch.tensor(test_mol_reduced_rep).to(unimol_rep.device)

                        if insert_rep_as_string:

                                str_smile_rep = fixed_sig_fig_str_tensor(unimol_rep, text_rep_sig_figs)
                                user_prompt = f"{user_prompt}{str_smile_rep}"
                                if load_HF_llm:
                                    model_input_embeddings = concatenate_text_to_embeddings(
                                        model_input_embeddings, str_smile_rep, model, tokenizer
                                    )
                        else:
                                if load_HF_llm:
                                    model_input_embeddings = inject_embeddings(
                                        model_input_embeddings,
                                        unimol_rep,
                                        rep_start_token_embedding,
                                        rep_end_token_embedding,
                                        normalize_fm_rep,
                                        non_zero_embeddings_mean_mean,
                                        non_zero_embeddings_var_mean,
                                        dummy_rep=debug_dummy_rep,
                                        backtranslate_rep=backtranslate_rep,
                                        model=model,
                                        embedding_matrix=embedding_matrix,
                                    )

                    
                    else:
                        test_text.extend([
                            f"\n\nQuestion {qn_idx}: What is the {predicted_property} of the drug molecule?"
                        ])


                        # test_text.extend([
                        #     f"\n\nGiven the representations of the drug molecule, answer the following question using the specified format.",
                        #     f"Question {qn_idx}: What is the {predicted_property} of the drug molecule?"
                        # ])

                        test_text.append(f"{mol_rep_prompt}")

                        test_text = "\n".join(test_text)
                        user_prompt = f"{user_prompt}{test_text}"
                        if load_HF_llm:
                            model_input_embeddings = concatenate_text_to_embeddings(
                                model_input_embeddings, test_text, model, tokenizer
                            )

                        unimol_rep = mol_fm.get_unimol_rep_tensor(test_smile)[0].to(
                        model.model.embed_tokens.weight.device
                    )

                        if rep_dim_reduction == "PCA":

                            if len(unimol_rep.shape) == 1:
                                unimol_rep_np = unimol_rep.unsqueeze(0).cpu().numpy()
                            else:
                                unimol_rep_np = unimol_rep.cpu().numpy()
                            test_mol_reduced_rep = mol_pca.transform(unimol_rep_np)

                            if len(unimol_rep.shape) == 1:
                                unimol_rep = torch.tensor(test_mol_reduced_rep[0]).to(unimol_rep.device)
                            else:
                                unimol_rep = torch.tensor(test_mol_reduced_rep).to(unimol_rep.device)

                        if insert_rep_as_string:

                                str_smile_rep = fixed_sig_fig_str_tensor(unimol_rep, text_rep_sig_figs)
                                user_prompt = f"{user_prompt}{str_smile_rep}"
                                if load_HF_llm:
                                    model_input_embeddings = concatenate_text_to_embeddings(
                                        model_input_embeddings, str_smile_rep, model, tokenizer
                                    )
                        else:
                                if load_HF_llm:
                                    model_input_embeddings = inject_embeddings(
                                        model_input_embeddings,
                                        unimol_rep,
                                        rep_start_token_embedding,
                                        rep_end_token_embedding,
                                        normalize_fm_rep,
                                        non_zero_embeddings_mean_mean,
                                        non_zero_embeddings_var_mean,
                                        dummy_rep=debug_dummy_rep,
                                        backtranslate_rep=backtranslate_rep,
                                        model=model,
                                        embedding_matrix=embedding_matrix,
                                    )



                else:
                    if use_icl_string:    
                        test_text = []


                        test_text.extend([
                            f"\n\nDrug SMILES: < {test_smile} >",
                            f"Given the SMILES sequence of the drug molecule {input_feature_position}, answer the following question using the specified format.",
                            f"Question {qn_idx}: What is the {predicted_property} of the drug molecule {input_feature_position}?",
                        ])


                        test_text = "\n".join(test_text)
                        user_prompt = f"{user_prompt}{test_text}"
                        if load_HF_llm:
                            model_input_embeddings = concatenate_text_to_embeddings(
                                model_input_embeddings, test_text, model, tokenizer
                            )


            # Create example answer format 

            if prompt_template_version == 3:
                if batch_query_test == True:
                    qn_idx = query_batch_idx + 1
                    test_text = [
                        f"\nPlease respond with the following format for each question:",
                        f"---BEGIN FORMAT TEMPLATE FOR QUESTION {qn_idx}---",  # for batch query, use: ---BEGIN FORMAT TEMPLATE FOR QUESTION {qn_idx}---
                        f"Answer {qn_idx}: [Your Answer Here for Question {qn_idx}]",  # for batch query, use: Answer Choice {qn_idx}: [Your Answer Choice Here for Question {qn_idx}]
                        # f"Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1 for Question]", # for batch query, use: Confidence Score {qn_idx}: [Your Numerical Prediction Confidence Score Here From 0 To 1 for Question {qn_idx}]
                        f"---END FORMAT TEMPLATE FOR QUESTION {qn_idx}---",  # for batch query, use: ---END FORMAT TEMPLATE FOR QUESTION {qn_idx}---
                    ]
                    query_batch_idx += 1
                else:
                    test_text = [
                        f"\nPlease respond with the following format for each question:",
                        f"---BEGIN FORMAT TEMPLATE FOR QUESTION---",  # for batch query, use: ---BEGIN FORMAT TEMPLATE FOR QUESTION {qn_idx}---
                        f"Answer: [Your Answer Here for Question]",  # for batch query, use: Answer Choice {qn_idx}: [Your Answer Choice Here for Question {qn_idx}]
                        # f"Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1 for Question]", # for batch query, use: Confidence Score {qn_idx}: [Your Numerical Prediction Confidence Score Here From 0 To 1 for Question {qn_idx}]
                        f"---END FORMAT TEMPLATE FOR QUESTION---",  # for batch query, use: ---END FORMAT TEMPLATE FOR QUESTION {qn_idx}---
                    ]
                

            test_text = "\n".join(test_text)


            # Append llm start of turn token after the last test example in the batch
            if batch_query_test != True or (
                batch_query_test == True
                and (qn_idx == batch_query_size or at_final_test_sample)
            ):  

                if not use_llm_api:
                    if model_type == "c4ai-command-r-plus":
                        test_text = f"{test_text}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
                    elif model_type == "qwen2":
                        test_text = f"{test_text}<|im_end|>\n<|im_start|>assistant\n"
                    elif llama_version == 3:
                        test_text = f"{test_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    else:
                        test_text = f"{test_text}[/INST]"

                # print(test_i, "E test_text: ", test_text)

            user_prompt = f"{user_prompt}{test_text}"
            # print("912 test_text: ", test_text)
            # print("912 model_input_embeddings: ", model_input_embeddings)
            if len(test_text) > 0 and load_HF_llm:
                model_input_embeddings = concatenate_text_to_embeddings(
                    model_input_embeddings, test_text, model, tokenizer
                )

            # add user prompt to the batch
            if batch_query_test != True or (
                batch_query_test == True
                and (qn_idx == batch_query_size or at_final_test_sample)
            ):
                if batch_query_test == True and qn_idx == batch_query_size:
                    query_batch_idx = 0

                if load_HF_llm:
                    batched_model_input_embeddings.append(model_input_embeddings)
                    
                    if torch.isnan(model_input_embeddings).any() or torch.isinf(model_input_embeddings).any():
                            print("Error: model_input_embeddings in batch contains invalid values.")

                else:
                    batched_model_input_embeddings.append(None)
                
                if system_message_in_prompt:
                    user_prompt = system_message + "\n" + user_prompt
                batched_user_prompts.append(user_prompt)


            if len(batched_model_input_embeddings) > 0 and (len(batched_model_input_embeddings) % batch_size == 0 or at_final_test_sample):

                if load_HF_llm:
                    max_length_in_batch = max([x.shape[1] for x in batched_model_input_embeddings])
                    attention_mask = torch.empty(0, max_length_in_batch).to(model.model.embed_tokens.weight.device)
                    for j, x in enumerate(batched_model_input_embeddings):

                        padding = pad_token_embedding.repeat(1, max_length_in_batch - x.shape[1], 1)
                        batched_model_input_embeddings[j] = torch.cat(
                            [padding, x], dim=1
                        )  # TODO: Check this, pad tokens should be padded at the end, not start of the input sequence

                        attention_mask = torch.cat(
                            [
                                attention_mask,
                                torch.cat(
                                    [torch.zeros(1, max_length_in_batch - x.shape[1]), torch.ones(1, x.shape[1])],
                                    dim=1,  # TODO: Check this, pad tokens should be padded at the end, not start of the input sequence, reverse attention_mask's order for pad_tokens and input_tokens
                                ).to(model.model.embed_tokens.weight.device),
                            ],
                            dim=0,
                        )

                    batched_model_input_embeddings = torch.cat(batched_model_input_embeddings, dim=0)

                # regenerate output if evaluation failed
                test_batch_gen_eval_success = False
                regen_count = 0
                while test_batch_gen_eval_success == False:
                    if regen_count > max_num_regen_count:
                        print("Max number of regenerations reached. Exiting test retry for batch_test_ids: ", batch_test_ids)
                        break

                    if regen_count > 0:
                        print(regen_count, " Retrying test batch generation for batch_test_ids: ", batch_test_ids)


                    decoded_input_texts = []
                    if not skip_HF_model_gen:
                        with torch.inference_mode():
                            batched_model_input_embeddings = batched_model_input_embeddings.to(model.lm_head.weight.dtype)

                            print("GPU monitor C ", test_i)
                            # monitor_gpu_util()

                            # if torch.isnan(batched_model_input_embeddings).any() or torch.isinf(batched_model_input_embeddings).any():
                            #     print("Error: batched_model_input_embeddings contains invalid values.")

                            # batched_model_input_embeddings = torch.nan_to_num(
                            #     batched_model_input_embeddings, nan=0.0, posinf=1.0, neginf=-1.0
                            # )


                            # if torch.isnan(attention_mask).any() or torch.isinf(attention_mask).any() or (attention_mask < 0).any():
                            #     print("Error: attention_mask contains invalid values.")

                            
                            if temperature != None:
                                batched_output_ids = model.generate(
                                    inputs_embeds=batched_model_input_embeddings,
                                    # max_length=max_input_length,
                                    max_length=batched_model_input_embeddings.shape[1] + max_gen_length,
                                    pad_token_id=tokenizer.pad_token_id,
                                    # stopping_criteria=stopping_criteria,
                                    attention_mask=attention_mask,
                                    temperature=temperature,
                                    
                                )
                            else:
                                batched_output_ids = model.generate(
                                    inputs_embeds=batched_model_input_embeddings,
                                    # max_length=max_input_length,
                                    max_length=batched_model_input_embeddings.shape[1] + max_gen_length,
                                    pad_token_id=tokenizer.pad_token_id,
                                    # stopping_criteria=stopping_criteria,
                                    attention_mask=attention_mask,
                                    
                                    
                                )

                            print("GPU monitor D ", test_i)
                            # monitor_gpu_util()
                            

                            batched_output_text = tokenizer.batch_decode(
                                batched_output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                            )
                            del batched_output_ids
                            # print("llama3.1 batched_output_text: ", batched_output_text)

                            # logging.info("Decoding embeddings")
                            # print("batched_model_input_embeddings: ", batched_model_input_embeddings)
                            decoded_input_token_ids = find_nearest_token_ids(batched_model_input_embeddings, embedding_matrix)
                            # print("decoded_input_token_ids: ", decoded_input_token_ids)
                            decoded_final_input_texts = tokenizer.batch_decode(decoded_input_token_ids)
                            # print("decoded_final_input_texts: ", decoded_final_input_texts)
                            # print("decoded_final_input_texts[0]: ", decoded_final_input_texts[0])
                            # # print("llm_prompt_text_icl_template: ", llm_prompt_text_icl_template)
                            # print("user_prompt: ", user_prompt)


                    print("GPU monitor E ", test_i)
                    # monitor_gpu_util()
                    
                    if skip_HF_model_gen:
                        decoded_input_texts.extend(batched_user_prompts)
                        batched_user_prompts = []
                    else:
                        decoded_input_texts.extend(decoded_final_input_texts)

                    # print("decoded_input_texts B: ", decoded_input_texts)

                    # Clear memory
                    if "unimol_rep" in locals():
                        del unimol_rep

                    gc.collect()
                    torch.cuda.empty_cache()

                    # set test_batch_gen_eval_success to True by default, and set to False if any test sample fails
                    test_batch_gen_eval_success = True
                    # print(test_i, " batched_output_text: ", batched_output_text)
                    # print(test_i, " batched_output_text len: ", len(batched_output_text))

                    # print(test_i, " batch_eval_y_labels: ", batch_eval_y_labels)
                    # print(test_i, " batch_eval_y_labels len: ", len(batch_eval_y_labels))

                    for test_processing_i, (generated, decoded_input_text) in enumerate(
                        tqdm(zip(batched_output_text, decoded_input_texts), desc="Processing batch", leave=False)
                    ):  
                        # print("test_processing_i: ", test_processing_i)
                        # collect predictions for evaluation
                        float_answers = extract_answer_floats(generated)
                        print(test_i, " b float_answers: ", float_answers)
                        print(test_i, " b float_answers[:len(batch_eval_y_labels)]: ", float_answers[:len(batch_eval_y_labels)])
                        print(test_i, " b batch_eval_y_labels: ", batch_eval_y_labels)
                        print(test_i, " b batch_test_ids: ", batch_test_ids)

                        # check that the number of predictions is at least larger the number of labels
                        test_sample_gen_eval_success = False
                        if len(float_answers) >= len(batch_eval_y_labels):
                            if len(float_answers) == len(batch_eval_y_labels):
                                eval_y_predictions = eval_y_predictions + float_answers
                                eval_y_labels = eval_y_labels + batch_eval_y_labels
                                eval_test_ids = eval_test_ids + batch_test_ids
                                test_sample_gen_eval_success = True
                            elif not only_use_exact_answer_num_match_for_eval:
                                eval_y_predictions = eval_y_predictions + float_answers[:len(batch_eval_y_labels)]
                                eval_y_labels = eval_y_labels + batch_eval_y_labels
                                eval_test_ids = eval_test_ids + batch_test_ids
                                test_sample_gen_eval_success = True
                        
                        # if one test sample fails, the whole batch fails
                        if test_sample_gen_eval_success == False:
                            test_batch_gen_eval_success = False

                        # cur_dialogue = [cur_dialogue, generated]
                        cur_new_dialogue = [decoded_input_text, generated]

                        if task_type == "DTI":
                            # cur_test_smile = batch_queue[-1][-1][0]
                            # cur_test_smile = batch_queue[-1][-1][1]
                            output_dialogue_filename = f"{timestamp}_{experiment_name}_testind{test_i}.txt"

                        else:
                            # cur_test_smile = batch_queue[-1][-1]
                            output_dialogue_filename = f"{timestamp}_{experiment_name}_testind{test_i}.txt"

                        # print("dialogue_output_path 2: ", dialogue_output_path)
                        # print("output_dialogue_filename 2: ", output_dialogue_filename)
                        # print("cur_new_dialogue: ", cur_new_dialogue)
                        export_dialogue(
                            cur_new_dialogue,
                            os.path.join(
                                dialogue_output_path,
                                "new_trajectories",
                                output_dialogue_filename
                            ), 
                            check_filename_length=True,
                        )

                    # batch_queue = []
                    if test_batch_gen_eval_success == False:
                        regen_count += 1

                # after generation loop, reset batch eval lists
                if test_batch_gen_eval_success == False:
                    not_eval_test_ids = not_eval_test_ids + batch_test_ids
                
                # reset batch eval lists after generation loop
                batch_eval_y_labels = []
                batch_test_ids = []

                if 'model_input_embeddings' in locals():
                    del model_input_embeddings
                del batched_model_input_embeddings
                batched_model_input_embeddings = []

    
        flattened_eval_y_labels = flatten_mixed_list(eval_y_labels)
        flattened_eval_y_predictions = flatten_mixed_list(eval_y_predictions)            

        eval_metrics = calculate_metrics(flattened_eval_y_labels, flattened_eval_y_predictions)
        
        eval_metrics["eval_y_labels"] = flattened_eval_y_labels
        eval_metrics["eval_y_predictions"] = flattened_eval_y_predictions

        eval_metrics["eval_test_ids"] = eval_test_ids
        eval_metrics["not_eval_test_ids"] = not_eval_test_ids
        eval_metrics["num_eval_test_ids"] = len(eval_test_ids)
        eval_metrics["num_not_eval_test_ids"] = len(not_eval_test_ids)
        eval_metrics["random_seed"] = random_seed
        eval_metrics["instruction_prompt_name"] = instruction_prompt_name

        print("eval_metrics: ", eval_metrics)

        save_json_to_file(eval_metrics, filename=os.path.join(dialogue_output_path, "results_eval_metrics.json"))
        print("Saved eval_metrics to file")

