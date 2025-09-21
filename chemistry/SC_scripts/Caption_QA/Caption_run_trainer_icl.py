import gc
import torch
from chemistry.SC_scripts.Caption_trainer_icl import run as run_base_batch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run DTI experiments.")
    parser.add_argument("--experiment_type", type=str, choices=["icl"], default="pca_icl", help="Type of experiment to run.")
    parser.add_argument("--experiment_name", type=str, default="ours", help="Base name of the experiment.")
    parser.add_argument("--seed", type=int, default=18451246, help="Random seed.")
    
    parser.add_argument("--task_name", type=str, choices=["ChEBI-20"], default="ChEBI-20", help="Task name.")
    parser.add_argument("--instruction_prompt_name", type=str, default="MoleculeCaption", help="Instruction prompt name.")

    parser.add_argument("--n_representation_shots", type=int, default=150, help="Number of shots for the representation.")
    parser.add_argument("--batch_query_size", type=int, default=1, help="Batch query size.")

    parser.add_argument("--pca_n_components", type=int, default=20, help="Number of components to keep for PCA.")
    parser.add_argument("--num_tests", type=int, default=90, help="Number of tests to run.")
    parser.add_argument("--num_trains", type=int, default=10000, help="Number of train sets to run.")

    # all_arch # "128-128", "256-256", "512-512", "1024-1024"
    parser.add_argument("--all_arch", type=str, default="64-64", help="Architecture of the FM LLM representation encoder.")
    parser.add_argument("--model_name", type=str, default="/home/chanalvin/cache/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/945c8663693130f8be2ee66210e062158b2a9693", help="Model name.")
    parser.add_argument("--max_gen_length", type=int, default=100, help="Max generation length.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation.")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    experiment_type = args.experiment_type
    experiment_name = args.experiment_name
    task_name = args.task_name
    file_name = experiment_name

    if experiment_type == "icl":
        use_icl_string = True
        use_rep = True
        insert_rep_as_string = False
        normalize_fm_rep=True
        experiment_name = f"{experiment_name}_icl"
    
    print(f"Running experiment: {experiment_name}")


    run_base_batch(sampling_type="stratified", random_seed=args.seed,
                   
                    # load_pairs=True,
                    load_pairs=False,

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

                    mol_rep_prompt="Molecular representation: ",
                    protein_rep_prompt="Protein representation: ",

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
                    max_gen_length = args.max_gen_length,
                    backtranslate_rep=False,
                    temperature=args.temperature,
                    shuffle_icl_examples=True,
                    max_smiles_length=100, # 100
                    max_protein_length=500, # 1000
                    # max_len_on_test_only=True,
                    max_len_on_test_only=False,

                    rep_dim_reduction="PCA",
                    pca_n_components=args.pca_n_components,
                    arch_fm_llm_rep_enc=args.all_arch,
                    file_name=file_name,

                    )