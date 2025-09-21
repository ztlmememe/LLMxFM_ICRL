import gc
import random
from tqdm import tqdm
import datetime
import torch
from chemistry.utils.audio_utils import AudioRepr, data_split_loading_audio, preprocess_audio_data
from chemistry.utils.utils import *
from chemistry.utils.LMM_api_utils import *
import traceback
from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM, BitsAndBytesConfig
import os
from huggingface_hub import login
import numpy as np
import pandas as pd
import json
import deepchem as dc
from sklearn.decomposition import PCA
import pickle
import joblib
from chemistry.utils.models import MLP_Mapper, MLP_Mapper_withoutbn, MLP_Linear
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd

torch.set_default_dtype(torch.float16)

task_type_dict = {
    "ESC50": "AUDIO",
    "VGGSound": "AUDIO",
}


def run(
    llama_num_b_params=70,
    n_full_shots=0,
    n_representation_shots=30,
    num_tests=-1,
    sampling_type="stratified",
    experiment_name="",
    n_sig_figs=-1,
    round_to_int=False,
    binned_regression=False,
    num_bins=5,
    bin_name_type='integer',
    display_answer_choices=False,
    random_seed=42,
    instruction_prompt_name="AUDIO_prompt_guided",
    batch_size=4,
    model_name="meta-llama/Meta-Llama-3-70B-Instruct",
    llama_version=3,
    temperature=None,
    predicted_property="Sound Category",
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
    avoid_loading_audio_model=False,
    first_n_embeddings=-1,
    normalize_fm_rep=False,
    debug_dummy_rep=False,
    backtranslate_rep=False,
    task_name='ESC50',
    split_method='random',
    split_features=None,
    audio_model_name="facebook/wav2vec2-base-960h",
    device_map="auto",
    only_use_exact_answer_num_match_for_eval=False,
    max_num_regen_count=3,
    use_icl_string=True,
    append_answer_format_at_end=False,
    shuffle_icl_examples=False,
    max_audio_length=None,
    max_len_on_test_only=False,
    rep_dim_reduction=None,
    pca_n_components=None,
    save_train_rep_dict=True,
    num_train_rep_dim_reduction=3000,
    use_dummy_rep=False,
    insert_rep_as_string=False,
    text_rep_sig_figs=3,
    use_llm_api=False,
    llm_api_model="gpt-4o",
    skip_HF_model_gen=False,
    system_message_in_prompt=False,
    input_feature_position="above",
    audio_rep_prompt="Use this audio representation to answer the question: ",
    skip_api_call_for_debug=False,
    arch_fm_llm_rep_enc=None,
    load_pairs=False,
    num_trains=1000,
    file_name="",
):
    """
    Executes the test suite for audio classification using LLMs.
    """

    torch.set_grad_enabled(False)
    set_random_seed(random_seed)

    # Dataset Preparation
    if task_name in task_type_dict:
        task_type = task_type_dict[task_name]

    # General Parameters
    dialogue_output_base_path = os.path.join(os.path.dirname(__file__), f"./../logs/{task_type}/{task_name}/{file_name}")
    experiment_base_name = (
        f"{task_name}_{n_representation_shots}_shots_{batch_query_size}_batch_query"
    )
    experiment_name = f"{experiment_base_name}_{experiment_name}"
    base_data_path = os.path.join(os.path.dirname(__file__), "./../datasets/")

    # Load data based on task
    if task_name == "ESC50" or task_name == "VGGSound":
        data = data_split_loading_audio(dataset_name=task_name, split_type=split_method, seed=random_seed, split_fracs=[0.7, 0.1, 0.2])
        
        train_inputs, train_y, valid_inputs, valid_y, test_inputs, test_y = data["train"]["AUDIO"], data["train"]["Labels"], data["valid"]["AUDIO"], data["valid"]["Labels"], data["test"]["AUDIO"], data["test"]["Labels"]
        train_srs, valid_srs, test_srs = data["train"]["SampleRates"], data["valid"]["SampleRates"], data["test"]["SampleRates"]
    else:
        raise ValueError(f"Unsupported task name: {task_name}")

    if isinstance(train_inputs, pd.DataFrame):
        train_inputs = train_inputs.to_numpy()
        valid_inputs = valid_inputs.to_numpy()
        test_inputs = test_inputs.to_numpy()
    
    if isinstance(train_y, pd.DataFrame):
        train_y = train_y.to_numpy()
        valid_y = valid_y.to_numpy()
        test_y = test_y.to_numpy()
    
    # Feature Model Preparation
    if use_rep:
        audio_fm = AudioRepr(model_name=audio_model_name, avoid_loading_model=avoid_loading_audio_model,
                          rep_cache_path=f'{base_data_path}/{task_name}_to_rep_store.pkl')
    else:
        print("Not loading audio feature models")
        audio_fm = AudioRepr(model_name=audio_model_name, avoid_loading_model=True)

    # Preprocess data if needed
    train_inputs, train_y, valid_inputs, valid_y, test_inputs, test_y = preprocess_audio_data(
        audio_fm, train_inputs, train_y, valid_inputs, valid_y, test_inputs, test_y, 
        base_data_path=base_data_path, skip_check=True
    )

    print("task_name:", task_name)
    print("train_inputs shape:", len(train_inputs))
    print("train_y shape:", len(train_y))

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    train_inputs, train_y = test_inputs, test_y 

    # load_pairs = False

    if load_pairs:
        train_fm_reps_tensor = torch.load(f"{base_data_path}/{task_name}_fm_reps_tensor_train.pt")
        print("Load train fm_reps_tensor.shape:", train_fm_reps_tensor.shape)
    else:
        fm_reps_tensor = []
        fm_inputs = []
        fm_srs = []

        # check training data and label
        print("train_inputs[0]:", train_inputs[0])
        print("max:", np.max(train_inputs[0]))
        print("min:", np.min(train_inputs[0]))
        print("mean:", np.mean(train_inputs[0]))
        print("std:", np.std(train_inputs[0]))
        print("train_y[0]:", train_y[0])

        print("Generating audio feature representations for training data")

        max_train_reps = num_trains
        
        for i, (train_input, sr) in tqdm(enumerate(zip(train_inputs, train_srs)), total=len(train_inputs), desc="train rep"):
            if len(fm_reps_tensor) >= max_train_reps:
                break
            
            try:
                # Create a unique ID for caching
                audio_id = f"{task_name}_train_{i}"
                
                # Get representation using AudioRepr
                audio_rep = audio_fm.get_wav2vec_rep_tensor(train_input, sr, audio_id)

                print("audio_rep.shape:", audio_rep.shape)
                
                # fm_inputs.append(train_input)
                # fm_srs.append(sr)
                fm_reps_tensor.append(audio_rep)
            except Exception as e:
                print(f"Error processing audio sample {i}: {e}")
                continue

        # Stack the audio representations
        train_fm_reps_tensor = torch.cat(fm_reps_tensor, dim=0)
        
        # Save the representations if requested
        if save_train_rep_dict:
            torch.save(train_fm_reps_tensor, f"{base_data_path}/{task_name}_fm_reps_tensor_train.pt")
            print(f"Saved {train_fm_reps_tensor.shape[0]} audio representations to file")

    # print("train_fm_reps_tensor.shape:", train_fm_reps_tensor.shape)
    # make sure 2 dim
    if train_fm_reps_tensor.ndim == 3:
        train_fm_reps_tensor = train_fm_reps_tensor.squeeze(1)
    # print("train_fm_reps_tensor.shape:", train_fm_reps_tensor.shape)



    # LLM Preparation
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    load_HF_llm = not (use_llm_api and skip_HF_model_gen)
    print("load_HF_llm:", load_HF_llm)

    # Set embedding dimensions based on model
    if model_type == "c4ai-command-r-plus":
        embedding_dim = 12288
    elif model_type == "qwen2":
        if "72B" in model_name:
            embedding_dim = 8192
        else:
            embedding_dim = 3584
    elif llama_num_b_params == 70:
        embedding_dim = 8192
    elif llama_num_b_params == 13:
        embedding_dim = 5120
    elif llama_num_b_params == 7 or llama_num_b_params == 8:
        embedding_dim = 4096
    
    max_input_length = 4096

    with torch.no_grad():
        # Load instruction prompt
        if instruction_prompt_name is not None:
            instruction_path = os.path.join(os.path.dirname(__file__), f"./../prompts/{instruction_prompt_name}.txt")
            with open(instruction_path, "r") as f:
                system_message = f.read()

        # Load LLM
        if load_HF_llm:
            if load_in_4bit:
                model = LlamaForCausalLM.from_pretrained(
                    model_name, device_map=device_map, load_in_4bit=load_in_4bit, torch_dtype=torch.bfloat16,
                )
            else:
                model = LlamaForCausalLM.from_pretrained(
                    model_name, device_map=device_map, torch_dtype=torch.bfloat16,
                )
                
            tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
            
            embedding_matrix = model.model.embed_tokens.weight
            llm_embs = embedding_matrix.detach().cpu().to(dtype=torch.float32).to(torch.float16).numpy()
            print("llm_embs.shape:", llm_embs.shape)
            llm_emb_dim_size = llm_embs.shape[-1]
            del llm_embs

            non_zero_embeddings_mean_mean = None
            non_zero_embeddings_var_mean = None

            # Calculate embedding statistics if normalization is requested
            if normalize_fm_rep:
                non_zero_embeddings = embedding_matrix[~torch.all(torch.abs(embedding_matrix) < 1e-10, dim=1)]
                if first_n_embeddings != -1:
                    non_zero_embeddings_to_get_stats = non_zero_embeddings[:first_n_embeddings]
                else:
                    non_zero_embeddings_to_get_stats = non_zero_embeddings

                non_zero_embeddings_mean_mean = (
                    non_zero_embeddings_to_get_stats.mean(-1, keepdim=True).mean().detach().item()
                )
                non_zero_embeddings_var_mean = (
                    non_zero_embeddings_to_get_stats.var(-1, keepdim=True).mean().detach().item()
                )
                print("non_zero_embeddings_mean_mean:", non_zero_embeddings_mean_mean)
                print("non_zero_embeddings_var_mean:", non_zero_embeddings_var_mean)
                del non_zero_embeddings
                del non_zero_embeddings_to_get_stats

                gc.collect()
                torch.cuda.empty_cache()

            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id

            # Prepare token embeddings
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

        # Process labels for example selection
        rounded_train_y = train_y

        # Prepare input format
        if isinstance(train_inputs, pd.DataFrame):
            train_inputs_np = train_inputs.to_numpy()
        elif isinstance(train_inputs, np.ndarray):
            train_inputs_np = train_inputs
        else:
            train_inputs_np = np.array(train_inputs)
        
        # OT process for audio representation
        audio_dim_size = train_fm_reps_tensor.shape[-1]

        train_audio_tensor = train_fm_reps_tensor.to(
            model.model.embed_tokens.weight.device
        )

        print("Using MLP_Linear for audio representation") 
        
        audio_projector = MLP_Linear(ninput=audio_dim_size, noutput=llm_emb_dim_size, layers=arch_fm_llm_rep_enc).to(
                        model.model.embed_tokens.weight.device
                    )
        
        # Apply PCA for dimensionality reduction
        audio_array = train_audio_tensor.cpu().numpy()

        audio_pca = PCA(n_components=pca_n_components, random_state=random_seed)
        audio_pca.fit(audio_array)
        train_audio_reduced_reps = audio_pca.fit_transform(audio_array)

        unique_text_model_ids = set()

        for k, reduced_tensor in enumerate(train_audio_reduced_reps):
            str_audio_rep = fixed_sig_fig_str_tensor(reduced_tensor, text_rep_sig_figs)
            text_model_ids = tokenizer.encode(str_audio_rep, return_tensors="pt", add_special_tokens=False).to(
                model.model.embed_tokens.weight.device
            )
            text_model_ids_tuple = tuple(text_model_ids.tolist()[0])

            new_elements = [elem for elem in text_model_ids_tuple if elem not in unique_text_model_ids]
            unique_text_model_ids.update(new_elements)
    
        unique_text_model_ids_tensor = torch.tensor(
                list(unique_text_model_ids), device=model.model.embed_tokens.weight.device
            ).unsqueeze(0)
            
        text_embeddings = model.model.embed_tokens(unique_text_model_ids_tensor).to(
            model.model.embed_tokens.weight.device
        )

        pca_train_tensor = text_embeddings.squeeze(0)

        mapped_train_tensors = []

        for k, train_tensor in enumerate(train_audio_tensor):
            mapper_output = audio_projector(train_tensor.unsqueeze(0)).to(
                model.model.embed_tokens.weight.device
            )
            mapped_train_tensors.append(mapper_output)

        mapped_train_tensor = torch.cat(mapped_train_tensors, dim=0).squeeze(0)
        scales_audio, shifts_audio = compute_alignment_params(mapped_train_tensor, pca_train_tensor)

        print("Audio alignment parameters computed")

        # Prepare example pairs for few-shot learning
        if train_inputs_np.ndim == 1:
            train_inputs_np = train_inputs_np.reshape(-1, 1)  
        
        if isinstance(rounded_train_y, np.ndarray) and rounded_train_y.ndim == 1:
            rounded_train_y = rounded_train_y.reshape(-1, 1)

        # Create audio-label pairs for few-shot examples
        example_audio_label_pairs = []
        for i in range(min(len(train_inputs_np), len(rounded_train_y))):
            example_audio_label_pairs.append((train_inputs_np[i], train_srs[i], rounded_train_y[i]))
        
        example_audio_label_pairs = np.array(example_audio_label_pairs, dtype=object)

        total_num_examples = n_full_shots + n_representation_shots

        # Select Examples for Few-Shot Learning
        if sampling_type == "stratified":
            # Sort by label for stratified sampling
            example_audio_label_pairs = sorted(example_audio_label_pairs, key=lambda x: x[2])
            
            # Select examples evenly distributed across labels
            gap_size = len(example_audio_label_pairs) // (total_num_examples + 1)
            selected_examples = [example_audio_label_pairs[i * gap_size] for i in range(1, total_num_examples + 1)]
            
            if shuffle_icl_examples:
                random.shuffle(selected_examples)

        
        example_features = []
        example_srs = []
        example_labels = []
        
        for ex in selected_examples:
            example_features.append(ex[0])
            example_srs.append(ex[1])
            example_labels.append(ex[2])

        text_buffer = ""
        llm_prompt_text_icl_template = []

        if load_HF_llm:
            user_prompt_embeddings_template = torch.empty((1, 0, embedding_dim)).to(model.model.embed_tokens.weight.device)
        
        # Setup prompt format
        if not use_llm_api:
            if instruction_prompt_name is not None:
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

        # Setup ICL instructions with shot samples
        if use_rep:
            if len(example_features) > 0:
                audio_reps = []
                for j, (example_audio, sr) in enumerate(zip(example_features, example_srs)):
                    audio_id = f"{task_name}_example_{j}"
                    audio_rep = audio_fm.get_wav2vec_rep_tensor(example_audio, sr, audio_id).to(
                        model.model.embed_tokens.weight.device
                    )

                    mapper_output = audio_projector(audio_rep).to(
                        model.model.embed_tokens.weight.device
                    )
                    mapper_output_aligned = apply_alignment(mapper_output, scales_audio, shifts_audio)
                    audio_reps.append(mapper_output_aligned)

                    if torch.isnan(mapper_output_aligned).any() or torch.isinf(mapper_output_aligned).any():
                        print("Error: example mapper_output_aligned contains invalid values.")
            else:
                audio_reps = []

        # Set up instructions before examples
        if len(text_buffer) > 0:
            llm_prompt_text_icl_template.append(text_buffer)
            if load_HF_llm:
                user_prompt_embeddings_template = concatenate_text_to_embeddings(
                    user_prompt_embeddings_template, text_buffer, model, tokenizer
                )
            text_buffer = ""

        print("GPU monitor A")

        # Build few-shot examples for in-context learning
        for example_i, (example_audio, audio_rep, label) in enumerate(zip(example_features, audio_reps, example_labels)):
            display_label = label if not binned_regression or bin_name_type != 'integer' else label

            # Format the example prompt
            if prompt_template_version == 3:
                answer_name = "Answer"
                
                # Prepare text to display before injecting representation
                if task_type == "AUDIO":
                    if use_rep:
                        pre_injection_text = []
                        if use_icl_string:
                            pre_injection_text.extend([
                                f"\n\nAudio Sample {example_i+1}",
                                f"Given the audio sample {input_feature_position}, answer the following question using the specified format.",
                                f"Question: What is the {predicted_property} of the audio sample {input_feature_position}?",
                            ])
                            
                            pre_injection_text.append(f"{audio_rep_prompt}")
                            
                            pre_injection_text = "\n".join(pre_injection_text)
                            text_buffer = f"{text_buffer}{pre_injection_text}"
                            llm_prompt_text_icl_template.append(text_buffer)
                            
                            if load_HF_llm:
                                user_prompt_embeddings_template = concatenate_text_to_embeddings(
                                    user_prompt_embeddings_template, text_buffer, model, tokenizer
                                )
                            text_buffer = ""
                                
                            example_audio_rep = torch.squeeze(audio_rep, dim=0)

                            if load_HF_llm:
                                user_prompt_embeddings_template = inject_embeddings_training(
                                    user_prompt_embeddings_template,
                                    example_audio_rep,
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
                            pre_injection_text = [
                                # f"\n\nQuestion: What is the {predicted_property} of the audio sample?",
                                f"\n\n{audio_rep_prompt}"
                            ]
                            
                            pre_injection_text = "\n".join(pre_injection_text)
                            text_buffer = f"{text_buffer}{pre_injection_text}"
                            llm_prompt_text_icl_template.append(text_buffer)
                            
                            if load_HF_llm:
                                user_prompt_embeddings_template = concatenate_text_to_embeddings(
                                    user_prompt_embeddings_template, text_buffer, model, tokenizer
                                )
                            text_buffer = ""

                            example_audio_rep = torch.squeeze(audio_rep, dim=0)

                            if load_HF_llm:
                                user_prompt_embeddings_template = inject_embeddings_training(
                                    user_prompt_embeddings_template,
                                    example_audio_rep,
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

            # Add the example answer
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

        llm_prompt_text_icl_template = "".join(llm_prompt_text_icl_template)
        print("GPU monitor B")

        # Inference
        output_dir_name = f"{timestamp}_{experiment_name}"

        batched_model_input_embeddings = []
        batched_user_prompts = []

        dialogue_output_path = os.path.join(dialogue_output_base_path, output_dir_name)
        os.makedirs(os.path.join(dialogue_output_path, "new_trajectories"), exist_ok=True)

        query_batch_idx = 0  # track index of batch query samples within the query
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

        for test_i, (test_input, test_sr, y) in tqdm(enumerate(zip(test_inputs, test_srs, test_y)), total=len(test_inputs)):
            if test_i == num_tests:
                break

            at_final_test_sample = (test_i == len(test_inputs) - 1 or test_i == num_tests - 1)
            
            # Collect labels for evaluation
            batch_eval_y_labels.append(y)
            batch_test_ids.append(test_i)
            
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

            # Process audio input for the test sample
            if task_type == "AUDIO":
                if use_rep:
                    audio_id = f"{task_name}_test_{test_i}"
                    test_text = []
                    
                    if use_icl_string:
                        test_text.extend([
                            f"\n\nAudio Sample for Testing",
                            f"Given the audio sample {input_feature_position}, answer the following question using the specified format.",
                            f"Question {qn_idx}: What is the {predicted_property} of the audio sample {input_feature_position}?",
                        ])

                        test_text.append(f"{audio_rep_prompt}")
                        
                        test_text = "\n".join(test_text)
                        user_prompt = f"{user_prompt}{test_text}"
                        if load_HF_llm:
                            model_input_embeddings = concatenate_text_to_embeddings(
                                model_input_embeddings, test_text, model, tokenizer
                            )

                        audio_rep = audio_fm.get_wav2vec_rep_tensor(test_input, test_sr, audio_id).to(
                            model.model.embed_tokens.weight.device
                        )
                        mapper_output = audio_projector(audio_rep).to(
                            model.model.embed_tokens.weight.device
                        )
                        mapper_output_aligned = apply_alignment(mapper_output, scales_audio, shifts_audio)
                        test_audio_rep = mapper_output_aligned
                        test_audio_rep = torch.squeeze(test_audio_rep, dim=0)

                        if load_HF_llm:
                            model_input_embeddings = inject_embeddings_training(
                                model_input_embeddings,
                                test_audio_rep,
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
                            f"\n\nQuestion {qn_idx}: What is the {predicted_property} of the audio sample?",
                        ])

                        test_text.append(f"{audio_rep_prompt}")

                        test_text = "\n".join(test_text)
                        user_prompt = f"{user_prompt}{test_text}"
                        if load_HF_llm:
                            model_input_embeddings = concatenate_text_to_embeddings(
                                model_input_embeddings, test_text, model, tokenizer
                            )

                        audio_rep = audio_fm.get_wav2vec_rep_tensor(test_input, test_sr, audio_id).to(
                            model.model.embed_tokens.weight.device
                        )
                        mapper_output = audio_projector(audio_rep).to(
                            model.model.embed_tokens.weight.device
                        )
                        mapper_output_aligned = apply_alignment(mapper_output, scales_audio, shifts_audio)
                        test_audio_rep = mapper_output_aligned
                        test_audio_rep = torch.squeeze(test_audio_rep, dim=0)

                        if load_HF_llm:
                            model_input_embeddings = inject_embeddings_training(
                                model_input_embeddings,
                                test_audio_rep,
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

            # Create example answer format
            if prompt_template_version == 3:
                if batch_query_test == True:
                    qn_idx = query_batch_idx + 1
                    test_text = [
                        f"\nPlease respond with the following format for each question:",
                        f"---BEGIN FORMAT TEMPLATE FOR QUESTION {qn_idx}---",
                        f"Answer {qn_idx}: [Your Answer Here for Question {qn_idx}]",
                        f"---END FORMAT TEMPLATE FOR QUESTION {qn_idx}---",
                    ]
                    query_batch_idx += 1
                else:
                    test_text = [
                        f"\nPlease respond with the following format for each question:",
                        f"---BEGIN FORMAT TEMPLATE FOR QUESTION---",
                        f"Answer: [Your Answer Here for Question]",
                        f"---END FORMAT TEMPLATE FOR QUESTION---",
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

            user_prompt = f"{user_prompt}{test_text}"
            if len(test_text) > 0 and load_HF_llm:
                model_input_embeddings = concatenate_text_to_embeddings(
                    model_input_embeddings, test_text, model, tokenizer
                )

            # Add user prompt to the batch
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

            # Process batch when it reaches batch_size or on the last sample
            if len(batched_model_input_embeddings) > 0 and (len(batched_model_input_embeddings) % batch_size == 0 or at_final_test_sample):
                if load_HF_llm:
                    max_length_in_batch = max([x.shape[1] for x in batched_model_input_embeddings])
                    attention_mask = torch.empty(0, max_length_in_batch).to(model.model.embed_tokens.weight.device)
                    for j, x in enumerate(batched_model_input_embeddings):
                        padding = pad_token_embedding.repeat(1, max_length_in_batch - x.shape[1], 1)
                        batched_model_input_embeddings[j] = torch.cat(
                            [padding, x], dim=1
                        )
                        attention_mask = torch.cat(
                            [
                                attention_mask,
                                torch.cat(
                                    [torch.zeros(1, max_length_in_batch - x.shape[1]), torch.ones(1, x.shape[1])],
                                    dim=1
                                ).to(model.model.embed_tokens.weight.device),
                            ],
                            dim=0,
                        )

                    batched_model_input_embeddings = torch.cat(batched_model_input_embeddings, dim=0)

                # Generate outputs, with retry logic if needed
                test_batch_gen_eval_success = False
                regen_count = 0
                while test_batch_gen_eval_success == False:
                    if regen_count > max_num_regen_count:
                        print("Max number of regenerations reached. Exiting test retry for batch_test_ids:", batch_test_ids)
                        break

                    if regen_count > 0:
                        print(f"{regen_count} Retrying test batch generation for batch_test_ids:", batch_test_ids)

                    decoded_input_texts = []
                    if not skip_HF_model_gen:
                        with torch.inference_mode():
                            batched_model_input_embeddings = batched_model_input_embeddings.to(model.lm_head.weight.dtype)

                            print("GPU monitor C", test_i)
                            
                            if temperature is not None:
                                batched_output_ids = model.generate(
                                    inputs_embeds=batched_model_input_embeddings,
                                    max_length=batched_model_input_embeddings.shape[1] + max_gen_length,
                                    pad_token_id=tokenizer.pad_token_id,
                                    attention_mask=attention_mask,
                                    temperature=temperature,
                                )
                            else:
                                batched_output_ids = model.generate(
                                    inputs_embeds=batched_model_input_embeddings,
                                    max_length=batched_model_input_embeddings.shape[1] + max_gen_length,
                                    pad_token_id=tokenizer.pad_token_id,
                                    attention_mask=attention_mask,
                                )

                            print("GPU monitor D", test_i)
                            
                            batched_output_text = tokenizer.batch_decode(
                                batched_output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                            )
                            del batched_output_ids

                            # Decode embeddings for logging
                            decoded_input_token_ids = find_nearest_token_ids(batched_model_input_embeddings, embedding_matrix)
                            decoded_final_input_texts = tokenizer.batch_decode(decoded_input_token_ids)

                    print("GPU monitor E", test_i)
                    
                    if skip_HF_model_gen:
                        decoded_input_texts.extend(batched_user_prompts)
                        batched_user_prompts = []
                    else:
                        decoded_input_texts.extend(decoded_final_input_texts)

                    # Clear memory
                    if "audio_rep" in locals():
                        del audio_rep

                    gc.collect()
                    torch.cuda.empty_cache()

                    # Evaluate outputs
                    test_batch_gen_eval_success = True
                    
                    for test_processing_i, (generated, decoded_input_text) in enumerate(
                        tqdm(zip(batched_output_text, decoded_input_texts), desc="Processing batch", leave=False)
                    ):  
                        # Extract answers from generated text
                        if task_name == "ESC50" or task_name == "VGGSound":
                            # For classification tasks, extract class names instead of floats
                            extracted_answers = extract_answer_strings(generated)
                            print(f"{test_i} extracted answers: {extracted_answers}")
                            print(f"{test_i} batch_eval_y_labels: {batch_eval_y_labels}")
                        else:
                            extracted_answers = extract_answer_floats(generated)
                            print(f"{test_i} extracted answers: {extracted_answers}")
                            print(f"{test_i} batch_eval_y_labels: {batch_eval_y_labels}")

                        # Check that the number of predictions matches the number of labels
                        test_sample_gen_eval_success = False
                        if len(extracted_answers) >= len(batch_eval_y_labels):
                            if len(extracted_answers) == len(batch_eval_y_labels):
                                eval_y_predictions = eval_y_predictions + extracted_answers
                                eval_y_labels = eval_y_labels + batch_eval_y_labels
                                eval_test_ids = eval_test_ids + batch_test_ids
                                test_sample_gen_eval_success = True
                            elif not only_use_exact_answer_num_match_for_eval:
                                eval_y_predictions = eval_y_predictions + extracted_answers[:len(batch_eval_y_labels)]
                                eval_y_labels = eval_y_labels + batch_eval_y_labels
                                eval_test_ids = eval_test_ids + batch_test_ids
                                test_sample_gen_eval_success = True
                        
                        # If one test sample fails, the whole batch fails
                        if test_sample_gen_eval_success == False:
                            test_batch_gen_eval_success = False

                        # Save dialogue
                        cur_new_dialogue = [decoded_input_text, generated]
                        output_dialogue_filename = f"{timestamp}_{experiment_name}_testind{test_i}.txt"

                        export_dialogue(
                            cur_new_dialogue,
                            os.path.join(
                                dialogue_output_path,
                                "new_trajectories",
                                output_dialogue_filename
                            ), 
                            check_filename_length=True,
                        )

                    # Retry generation if needed
                    if test_batch_gen_eval_success == False:
                        regen_count += 1

                # Track samples without successful evaluations
                if test_batch_gen_eval_success == False:
                    not_eval_test_ids = not_eval_test_ids + batch_test_ids
                
                # Reset batch evaluation lists
                batch_eval_y_labels = []
                batch_test_ids = []

                if 'model_input_embeddings' in locals():
                    del model_input_embeddings
                del batched_model_input_embeddings
                batched_model_input_embeddings = []

        # Calculate evaluation metrics
        if task_name == "ESC50" or task_name == "VGGSound":
            # For classification tasks, using accuracy instead of regression metrics
            eval_metrics = calculate_classification_metrics(eval_y_labels, eval_y_predictions)
        else:
            flattened_eval_y_labels = flatten_mixed_list(eval_y_labels)
            flattened_eval_y_predictions = flatten_mixed_list(eval_y_predictions)
            eval_metrics = calculate_metrics(flattened_eval_y_labels, flattened_eval_y_predictions)
            eval_metrics["eval_y_labels"] = flattened_eval_y_labels
            eval_metrics["eval_y_predictions"] = flattened_eval_y_predictions

        # Add additional metrics information
        eval_metrics["eval_test_ids"] = eval_test_ids
        eval_metrics["not_eval_test_ids"] = not_eval_test_ids
        eval_metrics["num_eval_test_ids"] = len(eval_test_ids)
        eval_metrics["num_not_eval_test_ids"] = len(not_eval_test_ids)
        eval_metrics["random_seed"] = random_seed
        eval_metrics["instruction_prompt_name"] = instruction_prompt_name

        print("eval_metrics:", eval_metrics)

        # Save results
        save_json_to_file(eval_metrics, filename=os.path.join(dialogue_output_path, "results_eval_metrics.json"))
        print("Saved eval_metrics to file")

def calculate_classification_metrics(y_true, y_pred):
    """
    Calculate classification metrics for audio classification tasks
    
    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    
    Returns:
    - Dictionary of metrics including accuracy
    """
    # Convert to lists if they're not already
    if not isinstance(y_true, list):
        y_true = y_true.tolist()
    if not isinstance(y_pred, list):
        y_pred = y_pred.tolist()
    
    # Calculate accuracy
    correct = sum(1 for true, pred in zip(y_true, y_pred) if str(true).lower() == str(pred).lower())
    accuracy = correct / len(y_true) if len(y_true) > 0 else 0
    
    # Create metrics dictionary
    metrics = {
        "accuracy": accuracy,
        "num_samples": len(y_true),
        "num_correct": correct
    }
    
    return metrics

def extract_answer_strings(text):
    """
    Extract answer strings from generated text output
    
    Parameters:
    - text: Generated text containing answers
    
    Returns:
    - List of extracted answer strings
    """
    import re
    
    # Look for answers in the format "Answer: [response]" or "Answer X: [response]"
    pattern = r"Answer(?:\s+\d+)?:\s*(.+?)(?:\n|$|\.|---END)"
    
    matches = re.findall(pattern, text)
    
    # Clean up the extracted answers
    cleaned_answers = []
    for match in matches:
        # Remove brackets, quotes, etc.
        cleaned = match.strip().strip('[]"\'')
        # Remove phrases like "Your Answer Here" if they appear
        cleaned = re.sub(r"\[?Your Answer (?:Here|Choice).*?\]?", "", cleaned)
        cleaned = cleaned.strip()
        if cleaned:  # Only add non-empty answers
            cleaned_answers.append(cleaned)
    
    return cleaned_answers