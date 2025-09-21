import gc
import torch
from chemistry.SC_scripts.AU_trainer_pca_OT_icl2 import run as run_base_batch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run audio classification experiments.")
    parser.add_argument("--experiment_type", type=str, choices=["rep_icl", "rep", "rep_icl_w_o_norm", "rep_w_o_norm"], 
                       default="rep_icl", help="Type of experiment to run.")
    parser.add_argument("--experiment_name", type=str, default="audio_experiment", help="Base name of the experiment.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    
    parser.add_argument("--task_name", type=str, choices=["ESC50", "VGGSound"], 
                       default="ESC50", help="Task name.")
    parser.add_argument("--instruction_prompt_name", type=str, 
                       default="AUDIO_prompt_guided", help="Instruction prompt name.")

    parser.add_argument("--n_representation_shots", type=int, default=20, help="Number of shots for the representation.")
    parser.add_argument("--batch_query_size", type=int, default=3, help="Batch query size.")

    parser.add_argument("--pca_n_components", type=int, default=20, help="Number of components to keep for PCA.")
    parser.add_argument("--num_tests", type=int, default=50, help="Number of tests to run.")
    parser.add_argument("--num_trains", type=int, default=1000, help="Number of train sets to run.")

    # Architecture options: "64-64", "128-128", "256-256", "512-512", "1024-1024"
    parser.add_argument("--all_arch", type=str, default="64-64", help="Architecture of the audio LLM representation encoder.")
    parser.add_argument("--model_name", type=str, 
                       default="meta-llama/Meta-Llama-3-70B-Instruct", help="Model name.")
    parser.add_argument("--audio_model_name", type=str, 
                       default="facebook/wav2vec2-base-960h", help="Audio model name.")
    parser.add_argument("--max_gen_length", type=int, default=100, help="Max generation length.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    experiment_type = args.experiment_type
    experiment_name = args.experiment_name
    task_name = args.task_name
    file_name = experiment_name

    if experiment_type == "rep_icl":
        use_icl_string = True
        use_rep = True
        insert_rep_as_string = False
        normalize_fm_rep = True
        experiment_name = f"{experiment_name}_rep_icl"

    elif experiment_type == "rep":
        use_icl_string = False
        use_rep = True
        insert_rep_as_string = False
        normalize_fm_rep = True
        experiment_name = f"{experiment_name}_rep"

    elif experiment_type == "rep_icl_w_o_norm":
        use_icl_string = True
        use_rep = True
        insert_rep_as_string = False
        normalize_fm_rep = False
        experiment_name = f"{experiment_name}_rep_icl_w_o_norm"

    elif experiment_type == "rep_w_o_norm":
        use_icl_string = False
        use_rep = True
        insert_rep_as_string = False
        normalize_fm_rep = False
        experiment_name = f"{experiment_name}_rep_w_o_norm"

    else:
        raise ValueError(f"Invalid experiment_type: {experiment_type}")
    
    print(f"Running experiment: {experiment_name}")

    if task_name == "ESC50":
        predicted_property = "Sound Category"
    elif task_name == "VGGSound":
        predicted_property = "Sound Class"
    else:
        raise ValueError(f"Unsupported task name: {task_name}")

    run_base_batch(
        sampling_type="stratified", 
        random_seed=args.seed,
        
        load_pairs=True,
        
        load_in_4bit=False,
        normalize_fm_rep=normalize_fm_rep,
        
        n_representation_shots=args.n_representation_shots,
        batch_query_size=args.batch_query_size,
        
        experiment_name=experiment_name,
        task_name=task_name,
        instruction_prompt_name=args.instruction_prompt_name,
        
        num_tests=args.num_tests,
        num_trains=args.num_trains,
        
        # Set to True to insert audio's string for in-context learning
        use_icl_string=use_icl_string,
        
        audio_rep_prompt="Audio vector representation: ",
        
        # Insert rep as string if true
        insert_rep_as_string=insert_rep_as_string,
        
        # Set false to use pure ICL without representations
        use_rep=use_rep,
        
        llama_num_b_params=70,
        model_name=args.model_name,
        audio_model_name=args.audio_model_name,
        n_sig_figs=3,
        batch_size=1,
        llama_version=3,
        n_full_shots=0,
        use_react=False,
        use_ack_text=False,
        
        batch_query_test=True,
        max_gen_length=args.max_gen_length,
        backtranslate_rep=False,
        temperature=0.6,  # 0.6 used for llama3 70B on huggingface model demo page
        shuffle_icl_examples=True,
        
        rep_dim_reduction="PCA",
        pca_n_components=args.pca_n_components,
        predicted_property=predicted_property,
        arch_fm_llm_rep_enc=args.all_arch,
        file_name=file_name,
    )