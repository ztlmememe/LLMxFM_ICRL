"""
This script executes the test suite for each experiment.
"""

import gc
import torch
from chemistry.SC_scripts.DTI_trainer_pca_OT_icl import run as run_base_batch

"""
This script was used for the DTI task and supports two datasets: BindingDB_Ki and DAVIS.
"""
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run DTI experiments.")
    parser.add_argument("--experiment_type", type=str, choices=["pca_icl", "icl", "pca","rep_icl", "rep","rep_icl_w_o_norm", "rep_w_o_norm"], default="pca_icl", help="Type of experiment to run.")
    parser.add_argument("--experiment_name", type=str, default="baseline", help="Base name of the experiment.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[18451246], help="List of random seeds.")
    parser.add_argument("--task_name", type=str, choices=["BindingDB_Ki", "DAVIS","BindingDB_IC50","KIBA"], default="BindingDB_Ki", help="Task name.")
    parser.add_argument("--instruction_prompt_name", type=str, default="ESOL_prompt_msr_1_guided_AC_simple", help="Instruction prompt name.")
    parser.add_argument("--mode", type=str, choices=["default",'max_affinity','mean'], default="default", help="Mode for the affinity values.")
    parser.add_argument("--n_representation_shots", type=int, default=10, help="Number of shots for the representation.")
    parser.add_argument("--batch_query_size", type=int, default=3, help="Batch query size.")
    
    return parser.parse_args()



if __name__ == "__main__":

    # seeds = [18451246]
    

    args = parse_args()

    experiment_type = args.experiment_type
    experiment_name = args.experiment_name
    seeds = args.seeds
    task_name = args.task_name
    mode = args.mode

    file_name = experiment_name
    instruction_prompt_name = args.instruction_prompt_name
    n_representation_shots = args.n_representation_shots
    batch_query_size = args.batch_query_size

    #  # "rep_icl", "rep"
    # # experiment_type = "rep_icl" 
    # experiment_type = "rep" 

    # experiment_name="ot_method"

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

    elif task_name == "BindingDB_Ki":
        predicted_property = "Binding Affinity Ki"

    elif task_name == "BindingDB_IC50":
        predicted_property = "Binding Affinity"

    elif task_name == "KIBA":
        predicted_property = "Binding Affinity"


    for all_arch in ['64-64']: # "128-128", "256-256", "512-512", "1024-1024"
        
        for seed in seeds:
                    
                    print(f"Running experiment with seed: {seed}")
                    

                    run_base_batch(sampling_type="stratified", random_seed=seed, 
                                   
                    mode = mode,
                    
                    n_representation_shots=n_representation_shots,
                    batch_query_size=batch_query_size,

                    load_in_4bit=False,
                    # load_in_4bit=True,

                    experiment_name=experiment_name,
                    task_name = task_name,
                    # task_name = "DAVIS",
                    
                    # used for ESOL experiment
                    # instruction_prompt_name="ESOL_prompt_msr_1_guided_AC_simple", 

                    # used for alvin's experiment
                    # instruction_prompt_name="ESOL_prompt_msr_1_guided_AC_simple_mol_repinstruct6",

                    # used for pure ICL + PCA (+ OT) 
                    instruction_prompt_name=instruction_prompt_name,

                    # used for pure PCA (+ OT) 
                    # instruction_prompt_name="DTI_prompt_mol_pro_rep", 

                    # used for pure ICL
                    # instruction_prompt_name="DTI_prompt_mol_pro", 
                    
                    normalize_fm_rep=normalize_fm_rep,
                    # normalize_fm_rep=False,

                    # num_tests=300,
                    num_tests=90,
                    
                    # set to True to insert mol/pro's string for icl
                    use_icl_string=use_icl_string,

                    mol_rep_prompt="Molecular vector representation: ",
                    protein_rep_prompt="Protein vector representation: ",

                    # use pca to reduce the dimension of the rep, then insert the reduced rep as a string
                    insert_rep_as_string=insert_rep_as_string,

                    # set false to use pure ICL
                    # use_rep=True,
                    use_rep=use_rep,

                    load_pairs=True,
                    # load_pairs=False,

                    llama_num_b_params=70, 
                    model_name = "/home/chanalvin/cache/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/945c8663693130f8be2ee66210e062158b2a9693",
                    

                    

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
                    pca_n_components=20,
                    predicted_property=predicted_property,
                    

                    arch_fm_llm_rep_enc=all_arch,
                    file_name=file_name,

                   )



