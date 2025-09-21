import os
import pickle
from tqdm import tqdm
from datasets import load_dataset
import torch
from chemistry.utils.unimol_utils import unimol_clf
import pandas as pd


import multiprocessing
import torch


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

def build_unimol_rep_cache(datapath="liupf/ChEBI-20-MM", 
                            cache_path="./smile_to_rep_store.pkl",
                            max_samples=None,
                            batch_size=256):

    
    mol_fm = unimol_clf(avoid_loading_model=False, rep_cache_path=cache_path)

    train_df, test_df = load_train_test_csv_from_dir(datapath)
    print(f"Train DataFrame rows example:\n{train_df.head()}")
    print(f"Test DataFrame shape: {test_df.shape}")
    smiles_train_list = train_df["SMILES"]

    smiles_test_list = test_df["SMILES"]



    # combine train and test SMILES
    smiles_list = pd.concat([smiles_train_list, smiles_test_list], ignore_index=True).tolist()


    # transform to numpy array
    if isinstance(smiles_list, pd.DataFrame):
        smiles_list = smiles_list.to_numpy()

    if max_samples:
        smiles_list = smiles_list[:max_samples]


    rep_cache = {}
    errors = []

    # smile_rep_map = pickle.load(cache_path) if os.path.exists(cache_path) else {}

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            smile_rep_map = pickle.load(f)
    else:
        smile_rep_map = {}


    smiles_to_process = []
    for smile in smiles_list:
        if smile not in smile_rep_map:
            smiles_to_process.append(smile)

    print(f"Total SMILES to process: {len(smiles_to_process)}")
    errors = []
    skipped_smiles_total = []

    MAX_SMILES_LENGTH = 800 


    for i in tqdm(range(0, len(smiles_to_process), batch_size), desc="Processing batches"):
        batch = smiles_to_process[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(smiles_to_process) + batch_size - 1) // batch_size} with {len(batch)} SMILES")
        
        try:

            filtered_batch = []
            for smile in batch:
                if len(smile) > MAX_SMILES_LENGTH:
                    print(f" Skipping too long SMILES (len={len(smile)}): {smile[:60]}...")
                    skipped_batch_save_path = '/mnt/ssd/ztl/LLMxFM/chemistry/SC_scripts/skip.txt'


                    with open(skipped_batch_save_path, "w") as f:
                        f.write(smile + "\n")
                else:
                    filtered_batch.append(smile)

            batch = filtered_batch
            

            reps = mol_fm.clf.get_repr(batch)["cls_repr"]

            # reps = run_repr_with_timeout(batch, mol_fm, timeout=300)
            print(f"reps.shape: {reps[0].shape}")

            if reps is None:
                print(" Batch processing failed due to timeout or error.")
                errors.extend(batch)
                skipped_smiles_total.extend(batch)
                continue
            else:
                for smile, rep in zip(batch, reps):
                    smile_rep_map[smile] = torch.tensor(rep)


        except Exception as e:
            print(f"Error in batch {i // batch_size}: {e}")
            errors.extend(batch)
            continue

        
        try:
            tmp_path = cache_path + ".tmp"
            with open(tmp_path, "wb") as f:
                pickle.dump(smile_rep_map, f)
            os.replace(tmp_path, cache_path)
        except Exception as e:
            print(f" Failed to save cache: {e}")

    print(f"Finished. Total saved: {len(smile_rep_map)} SMILES")

    

    if errors:
        print(f"{len(errors)} SMILES failed. First few: {errors[:5]}")








# python -m chemistry.SC_scripts.get_unimol_cache


if __name__ == "__main__":
    build_unimol_rep_cache(
        datapath="/mnt/ssd/ztl/LLMxFM/chemistry/datasets/moleculeqa/TXT/Property",
        cache_path="/mnt/ssd/ztl/LLMxFM/chemistry/datasets/smile_to_rep_store.pkl",
        max_samples=None
    )

    build_unimol_rep_cache(
        datapath="/mnt/ssd/ztl/LLMxFM/chemistry/datasets/moleculeqa/TXT/Source",
        cache_path="/mnt/ssd/ztl/LLMxFM/chemistry/datasets/smile_to_rep_store.pkl",
        max_samples=None
    )

    build_unimol_rep_cache(
        datapath="/mnt/ssd/ztl/LLMxFM/chemistry/datasets/moleculeqa/TXT/Structure",
        cache_path="/mnt/ssd/ztl/LLMxFM/chemistry/datasets/smile_to_rep_store.pkl",
        max_samples=None
    )

    build_unimol_rep_cache(
        datapath="/mnt/ssd/ztl/LLMxFM/chemistry/datasets/moleculeqa/TXT/Usage",
        cache_path="/mnt/ssd/ztl/LLMxFM/chemistry/datasets/smile_to_rep_store.pkl",
        max_samples=None
    )
