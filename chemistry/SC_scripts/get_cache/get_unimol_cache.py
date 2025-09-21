import os
import pickle
from tqdm import tqdm
from datasets import load_dataset
import torch
from chemistry.utils.unimol_utils import unimol_clf
import pandas as pd


import multiprocessing
import torch

def run_repr_with_timeout(batch, mol_fm,timeout=300):
    def worker(queue):
        try:
            reps = mol_fm.clf.get_repr(batch)["cls_repr"]
            queue.put(reps)
        except Exception as e:
            queue.put(e)

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=worker, args=(queue,))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        print("Timeout: get_repr execution exceeded time limit.")
        return None
    else:
        result = queue.get()
        if isinstance(result, Exception):
            print("Error occurred during get_repr:", result)
            return None
        return result




def build_unimol_rep_cache(dataset_name="liupf/ChEBI-20-MM", 
                            cache_path="./smile_to_rep_store.pkl",
                            max_samples=None,
                            batch_size=128):


    mol_fm = unimol_clf(avoid_loading_model=False, rep_cache_path=cache_path)

    dataset_train = load_dataset("liupf/ChEBI-20-MM", split="train")  # full dataset
    smiles_train_list = dataset_train["SMILES"]
    train_inputs = pd.DataFrame(smiles_train_list, columns=["SMILES"])


    dataset_test = load_dataset("liupf/ChEBI-20-MM", split="test")  # full dataset
    smiles_test_list = dataset_test["SMILES"]
    test_inputs = pd.DataFrame(smiles_test_list, columns=["SMILES"])


    # combine train and test SMILES
    smiles_list = pd.concat([train_inputs, test_inputs], ignore_index=True)["SMILES"].tolist()

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
        # print(f"Batch SMILES: {batch[:5]}...") 
        
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

        # 每处理一个 batch 就保存 cache（安全写入）
        try:
            tmp_path = cache_path + ".tmp"
            with open(tmp_path, "wb") as f:
                pickle.dump(smile_rep_map, f)
            os.replace(tmp_path, cache_path)
        except Exception as e:
            print(f" Failed to save cache: {e}")

    print(f"Finished. Total saved: {len(smile_rep_map)} SMILES")

    

    if errors:
        print(f" {len(errors)} SMILES failed. First few: {errors[:5]}")








# python -m chemistry.SC_scripts.get_unimol_cache


if __name__ == "__main__":
    build_unimol_rep_cache(
        dataset_name="liupf/ChEBI-20-MM",
        cache_path="/mnt/ssd/ztl/LLMxFM/chemistry/datasets/smile_to_rep_store.pkl",
        max_samples=None
    )
