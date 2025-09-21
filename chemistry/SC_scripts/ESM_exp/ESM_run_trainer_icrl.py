import gc
import torch
from chemistry.SC_scripts.ESM_trainer_icrl import run as run_base_batch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run DTI experiments.")
    parser.add_argument("--experiment_type", type=str, choices=["rep_icl", "rep","rep_icl_w_o_norm", "rep_w_o_norm"], default="pca_icl", help="Type of experiment to run.")
    parser.add_argument("--experiment_name", type=str, default="ours", help="Base name of the experiment.")
    parser.add_argument("--seed", type=int, default=18451246, help="Random seed.")
    
    parser.add_argument("--task_name", type=str, choices=["Stability", "Fluorescence","Beta_Lactamase","PPI_Affinity"], default="Stability", help="Task name.")
    parser.add_argument("--instruction_prompt_name", type=str, default="ESOL_prompt_msr_1_guided_AC_simple", help="Instruction prompt name.")

    parser.add_argument("--n_representation_shots", type=int, default=20, help="Number of shots for the representation.")
    parser.add_argument("--batch_query_size", type=int, default=3, help="Batch query size.")

    parser.add_argument("--pca_n_components", type=int, default=20, help="Number of components to keep for PCA.")
    parser.add_argument("--num_tests", type=int, default=90, help="Number of tests to run.")
    parser.add_argument("--num_trains", type=int, default=1000, help="Number of train sets to run.")

    # all_arch # "128-128", "256-256", "512-512", "1024-1024"
    parser.add_argument("--all_arch", type=str, default="64-64", help="Architecture of the FM LLM representation encoder.")

    parser.add_argument("--model_name", type=str, default="/home/chanalvin/cache/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/945c8663693130f8be2ee66210e062158b2a9693", help="Model name.")

    
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
        normalize_fm_rep=True
        experiment_name = f"{experiment_name}_rep_icl"

    elif experiment_type == "rep":
        use_icl_string = False
        use_rep = True
        insert_rep_as_string = False
        normalize_fm_rep=True
        experiment_name = f"{experiment_name}_rep"

    elif experiment_type == "rep_icl_w_o_norm":
        use_icl_string = True
        use_rep = True
        insert_rep_as_string = False
        normalize_fm_rep=False

        experiment_name = f"{experiment_name}_rep_icl_w_o_norm"

    elif experiment_type == "rep_w_o_norm":
        use_icl_string = False
        use_rep = True
        insert_rep_as_string = False
        normalize_fm_rep=False
        experiment_name = f"{experiment_name}_rep_w_o_norm"

    else:
        raise ValueError(f"Invalid experiment_type: {experiment_type}")
    
    print(f"Running experiment: {experiment_name}")

    if task_name == "Stability":
        predicted_property = "Stability"

    elif task_name == "Fluorescence":
        predicted_property = "Fluorescence"

    elif task_name == "Beta_Lactamase":
        predicted_property = "Fitness scores"
    
    elif task_name == "PPI_Affinity":
        predicted_property = "Numerical interaction affinity value"
         

    run_base_batch(sampling_type="stratified", random_seed=args.seed,
                   
                    load_pairs=True,
                    # load_pairs=False,

                    load_in_4bit=False,
                    # load_in_4bit=True,

                    normalize_fm_rep=normalize_fm_rep,
                    # normalize_fm_rep=False,
    
                    n_representation_shots=args.n_representation_shots,
                    batch_query_size=args.batch_query_size,

                    experiment_name=experiment_name,
                    task_name = task_name,
                    instruction_prompt_name=args.instruction_prompt_name,

                    # num_tests=300,
                    num_tests=args.num_tests,
                    num_trains=args.num_trains,
                    
                    # set to True to insert mol/pro's string for icl
                    use_icl_string=use_icl_string,

                    mol_rep_prompt="Molecular vector representation: ",
                    protein_rep_prompt="Protein vector representation: ",

                    # use pca to reduce the dimension of the rep, then insert the reduced rep as a string
                    insert_rep_as_string=insert_rep_as_string,

                    # set false to use pure ICL
                    use_rep=use_rep,

                    llama_num_b_params=70, 
                    model_name = args.model_name,
                    n_sig_figs=3, 
                    batch_size=1,
                    llama_version=3,
                    # llama_num_b_params=8, model_name = "meta-llama/Meta-Llama-3-8B-Instruct",
                    n_full_shots=0, 
                    use_react=False,
                    use_ack_text=False,

                    batch_query_test=True,
                    max_gen_length = 100,
                    backtranslate_rep=False,
                    temperature=0.6, # 0.6 used for llama31 70B on huggingface model demo page, default value is 1.0 from huggingface
                    shuffle_icl_examples=True,
                    max_smiles_length=100, # 100
                    max_protein_length=500, # 1000
                    # max_len_on_test_only=True,
                    max_len_on_test_only=False,

                    rep_dim_reduction="PCA",
                    pca_n_components=args.pca_n_components,
                    predicted_property=predicted_property,
                    arch_fm_llm_rep_enc=args.all_arch,
                    file_name=file_name,

                    )