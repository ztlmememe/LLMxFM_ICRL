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
import evaluate

torch.set_default_dtype(torch.float16)

task_type_dict = {
    "Property": "QA",
    "Source": "QA",
    "Structure": "QA",
    "Usage": "QA",
}
def load_train_test_csv_from_dir(folder_path):
    train_path = os.path.join(folder_path, "train.csv")
    test_path = os.path.join(folder_path, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train.csv not found in {folder_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"test.csv not found in {folder_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Loaded train.csv ({len(train_df)} samples)")
    print(f"Loaded test.csv ({len(test_df)} samples)")

    return train_df, test_df

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
    # rep_start_token="[REP]",
    # rep_end_token="[/REP]",
    rep_start_token="(",
    rep_end_token=")",
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
        task_type = task_type_dict[task_name] # Caption

    # General Parameters
    dialogue_output_base_path = os.path.join(os.path.dirname(__file__), f"./../logs/{task_type}/{task_name}/{file_name}")
    experiment_base_name = (
        f"{task_name}_{n_representation_shots}_shots_{batch_query_size}_batch_query"
    )
    experiment_name = f"{experiment_base_name}_{experiment_name}"
    base_data_path = os.path.join(os.path.dirname(__file__), "./../datasets/")

    base_data_name = "/home/users/ntu/tianle00/scratch/llmxfm-0.0/chemistry/datasets/moleculeqa/TXT"
    folder = os.path.join(base_data_name, task_name)
    if task_type == "QA":

        # train_inputs, train_y, test_inputs, test_y = 

        # data_folder = "/home/users/ntu/tianle00/scratch/llmxfm-0.0/chemistry/datasets/moleculeqa/TXT/Property"
        

        train_df, test_df = load_train_test_csv_from_dir(folder)


        

        train_inputs = train_df["SMILES"]
        train_questions = train_df["Question"]
        train_y = train_df["Answer"]

        test_inputs = test_df["SMILES"]
        test_questions = test_df["Question"]
        test_y = test_df["Answer"]


        print("train_inputs.shape: ", train_inputs.shape)
        print("train_questions.shape: ", train_questions.shape)
        print("train_y.shape: ", train_y.shape)
        print("test_inputs.shape: ", test_inputs.shape)
        print("test_questions.shape: ", test_questions.shape)
        print("test_y.shape: ", test_y.shape)

        # print example SMILES and questions
        # print("Example train SMILES: ", train_inputs.iloc[0])
        # print("Example train question: ", train_questions.iloc[0])
        # print("Example train answer: ", train_y.iloc[0])

        # train_inputs, train_y, valid_inputs, valid_y, test_inputs, test_y = data["train"]["SMILES"], data["train"]["Labels"], data["valid"]["SMILES"], data["valid"]["Labels"], data["test"]["SMILES"], data["test"]["Labels"]

    
        
    if isinstance(train_inputs, pd.DataFrame):
        train_inputs = train_inputs.to_numpy()
        # valid_inputs = valid_inputs.to_numpy()
        train_questions = train_questions.to_numpy()
        test_inputs = test_inputs.to_numpy()
    
    elif isinstance(train_inputs, np.ndarray):
        train_inputs = train_inputs
        # valid_inputs = valid_inputs
        train_questions = train_questions.to_numpy()
        test_inputs = test_inputs
    else:
        try:
            train_inputs = train_inputs.to_numpy()
            # valid_inputs = valid_inputs.to_numpy()
            train_questions = train_questions.to_numpy()
            test_inputs = test_inputs.to_numpy()
        except:
            raise ValueError("train and test inputs must be a pandas DataFrame or numpy array")

    if isinstance(train_y, pd.DataFrame):
        train_y = train_y.to_numpy()
        # valid_y = valid_y.to_numpy()
        test_y = test_y.to_numpy()
    elif isinstance(train_y, np.ndarray):
        train_y = train_y
        # valid_y = valid_y
        test_y = test_y
    else:
        try:
            train_y = train_y.to_numpy()
            # valid_y = valid_y.to_numpy()
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
    elif '70B' in model_name:
        embedding_dim = 8192
    elif llama_num_b_params == 13:
        embedding_dim = 5120
    elif '8B' in model_name:
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
        rounded_train_y = train_y

        if isinstance(train_inputs, pd.DataFrame):
            train_inputs_np = train_inputs.to_numpy()
        elif isinstance(train_inputs, np.ndarray):
            train_inputs_np = train_inputs
        else:
            raise ValueError("train_inputs must be a pandas DataFrame or numpy array")
        
        # OT process -- molecule rep
        
        # print("train_inputs_np.shape: ", train_inputs_np.shape)
        # print("rounded_train_y.shape: ", rounded_train_y.shape)
        if train_inputs_np.ndim == 1:
            train_inputs_np = train_inputs_np.reshape(-1, 1)  
        
        if rounded_train_y.ndim == 1:
            rounded_train_y = rounded_train_y.reshape(-1, 1)

        if train_questions.ndim == 1:
            train_questions = train_questions.reshape(-1, 1)

        # print("train_inputs_np.shape: ", train_inputs_np.shape)
        # print("rounded_train_y.shape: ", rounded_train_y.shape)
        # print("train_questions.shape: ", train_questions.shape)

        
        example_smiles_label_pairs = np.concatenate([train_inputs_np, train_questions, rounded_train_y], axis=1)
        from collections import defaultdict

        # 构建类别到样本的映射
        label_to_examples = defaultdict(list)
        for row in example_smiles_label_pairs:
            label = row[-1]  # 假设最后一列是标签（A/B/C/D）
            label_to_examples[label].append(row)

        # 计算每类应采样数量
        total_num_examples = n_full_shots + n_representation_shots
        num_classes = len(label_to_examples)
        num_per_class = total_num_examples // num_classes

        # 为每个类别采样
        rng = np.random.default_rng(seed=random_seed)
        selected = []

        for label, examples in label_to_examples.items():
            examples = np.array(examples)
            if len(examples) < num_per_class:
                print(f"Warning: not enough examples for label {label}, only {len(examples)} available.")
                sampled = examples
            else:
                idx = rng.permutation(len(examples))[:num_per_class]
                sampled = examples[idx]
            selected.append(sampled)

        # 合并所有类别的样本
        selected_example_input_features_label_pairs = np.concatenate(selected, axis=0)

        # 打乱示例顺序（保持样本对齐）
        rng = np.random.default_rng(seed=random_seed)  # 可设定种子以保证复现性
        rng.shuffle(selected_example_input_features_label_pairs)  # 就地打乱每一行顺序




        # 拆分出 SMILES、Question、Label 三部分
        example_features = selected_example_input_features_label_pairs[:, 0]
        example_questions = selected_example_input_features_label_pairs[:, 1:-1]
        example_labels = selected_example_input_features_label_pairs[:, -1]

        if example_features.shape[-1] == 1:
            example_smiles = np.squeeze(example_features, axis=-1)
        
        else:
            example_smiles = example_features


        # 打印确认
        print("example_features.shape: ", example_features.shape)
        print("example_questions.shape: ", example_questions.shape)
        print("example_labels.shape: ", example_labels.shape)
        

        



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
        mol_reps = len(example_smiles) * [None]


        print("sample done")

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


        for example_i, (example_feature, example_question, mol_rep, label) in enumerate(zip(example_features, example_questions,
                                                                          mol_reps, example_labels)):
            
            if isinstance(example_feature, list) and len(example_feature) == 1:
                example_smile = example_feature[0]

            elif isinstance(example_feature , np.ndarray) and len(example_feature) == 1:
                example_smile = example_feature[0]

            else:
                example_smile = example_feature
            # print("example_smile: ", example_smile)
            # print("example_smile[0]: ", example_smile[0])   

            display_label = label

            # Segment 1/3: Text before injection.
            if prompt_template_version == 3:
                if prompt_template_version == 3:
                            answer_name = "Answer"
                else:
                    answer_name = predicted_property


                if task_type == "QA":
                    # pre_injection_text = []
                    if use_rep:
                        pre_injection_text = []
                        if use_icl_string:
                            pre_injection_text.extend([
                                f"\n\nMolecule SMILES: < {example_smile} >",
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
                # example_question
                    f"\n{example_question[0]}",
                    f"{answer_name}: {display_label}",
                ]

            post_injection_text = "\n".join(post_injection_text)
            text_buffer = f"{text_buffer}{post_injection_text}"
            llm_prompt_text_icl_template.append(text_buffer)
            if load_HF_llm:
                user_prompt_embeddings_template = concatenate_text_to_embeddings(
                    user_prompt_embeddings_template, text_buffer, model, tokenizer
                )
            text_buffer = ""

        if torch.isnan(user_prompt_embeddings_template).any() or torch.isinf(user_prompt_embeddings_template).any():
            print("Error: user_prompt_embeddings_template after contains invalid values.")

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

        all_predictions = []
        all_references = []


        if torch.isnan(user_prompt_embeddings_template).any() or torch.isinf(user_prompt_embeddings_template).any():
            print("Error: user_prompt_embeddings_template before test inject contains invalid values.")

        for test_i, (test_input, test_question,y) in tqdm(enumerate(zip(test_inputs, test_questions,
                                                          test_y)), total=len(test_inputs)):

            # print("test_input: ", test_input)
            # print("y: ", y)


            # all_predictions.append(generated.strip())
            
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


            if task_type == "QA":
                if use_rep:
                    test_text = []
                    # eg. input_feature_position="above", predicted_property="Binding Affinity Ki" predicted_property="Solubility",
                    if use_icl_string:
                        test_text.extend([
                            f"\n\nMolecule SMILES: < {test_smile} >",
                        ])

                        # test_text.append(f"{mol_rep_prompt}")
                        
                        test_text = "\n".join(test_text)
                        user_prompt = f"{user_prompt}{test_text}"
                        if load_HF_llm:
                            model_input_embeddings = concatenate_text_to_embeddings(
                                model_input_embeddings, test_text, model, tokenizer
                            )


            # Create example answer format 

            batch_query_test = False

            test_text = [
                f"\n{test_question}",
                f"Please respond with the following format:",
                # f"\nDescription:",
                f"---BEGIN ANSWER---",
                f"Answer: [Your choice for the given question]",
                f"---END ANSWER---",
            ]


            # test_text = [
                # f"\nDescription:",
            # ]


            

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

                            if torch.isnan(batched_model_input_embeddings).any() or torch.isinf(batched_model_input_embeddings).any():
                                print("Error: batched_model_input_embeddings contains invalid values.")

                            # std_val = batched_model_input_embeddings.std()
                            # print(f"batched_model_input_embeddings values are nearly constant. std={std_val.item()}")


                            # batched_model_input_embeddings = torch.nan_to_num(
                            #     batched_model_input_embeddings, nan=0.0, posinf=1.0, neginf=-1.0
                            # )


                            # if torch.isnan(attention_mask).any() or torch.isinf(attention_mask).any() or (attention_mask < 0).any():
                            #     print("Error: attention_mask contains invalid values.")

                            # print("batched_model_input_embeddings.shape: ", batched_model_input_embeddings.shape)
                            # print("attention_mask.shape: ", attention_mask.shape)

                            decoded_input_token_ids = find_nearest_token_ids(batched_model_input_embeddings, embedding_matrix)
                            # print("decoded_input_token_ids: ", decoded_input_token_ids)
                            decoded_final_input_texts = tokenizer.batch_decode(decoded_input_token_ids)

                            # print("decoded_final_input_texts: ", decoded_final_input_texts)

                            # # print(torch.isnan(batched_model_input_embeddings).any(), torch.isinf(batched_model_input_embeddings).any(), batched_model_input_embeddings.min())
                            # print(tokenizer.decode([114062]))


                            if temperature != None:
                                batched_output_ids = model.generate(
                                    inputs_embeds=batched_model_input_embeddings,
                                    # max_length=max_input_length,
                                    # max_length=batched_model_input_embeddings.shape[1] + max_gen_length,
                                    pad_token_id=tokenizer.pad_token_id,
                                    # stopping_criteria=stopping_criteria,
                                    attention_mask=attention_mask,
                                    temperature=temperature,

                                    max_new_tokens=128,
                                    # temperature=1.2, 
                                    do_sample=True,
                                    
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

                            


                    print("GPU monitor E ", test_i)
                    # monitor_gpu_util()
                    
                    if skip_HF_model_gen:
                        decoded_input_texts.extend(batched_user_prompts)
                        batched_user_prompts = []
                    else:
                        decoded_input_texts.extend(decoded_final_input_texts)

                    # print("batched_output_text[0]: ", batched_output_text[0])

                    pred = extract_qa_choice_from_response(batched_output_text[0])
                    # print("Longest Caption:\n", pred)

                    if pred is not None and len(pred) > 0:

                        # batched_output_text = [x.strip() for x in batched_output_text]
                        # print("batched_output_text: ", batched_output_text)
                        all_predictions.append(pred.strip())
                        all_references.append(y[0].strip())
                        test_batch_gen_eval_success = True
                    else:
                        test_batch_gen_eval_success = False
                        regen_count += 1
                        


                if not test_batch_gen_eval_success:
                    not_eval_test_ids = not_eval_test_ids + batch_test_ids

                batch_eval_y_labels = []
                batch_test_ids = []

                if 'model_input_embeddings' in locals():
                    del model_input_embeddings
                del batched_model_input_embeddings
                batched_model_input_embeddings = []


                # Clear memory
                if "mol_rep" in locals():
                    del mol_rep

                gc.collect()
                torch.cuda.empty_cache()
                

                cur_new_dialogue = [decoded_input_texts[0],pred]

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

                
    
        assert len(all_predictions) == len(all_references), "Mismatch in prediction and reference lengths"

        # 计算准确率
        correct = sum(p == r for p, r in zip(all_predictions, all_references))
        accuracy = correct / len(all_predictions) if all_predictions else 0.0

        metrics = {"Accuracy": accuracy}

        # 打印结果
        print("\nEvaluation Metrics:")
        for key, val in metrics.items():
            print(f"{key}: {val:.2%}")  # 百分比格式输出，如 85.33%

        # 保存到文件
        save_json_to_file(metrics, filename=os.path.join(dialogue_output_path, "results_eval_metrics.json"))
        print("Saved eval_metrics to file")


