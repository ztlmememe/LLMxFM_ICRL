import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from transformers import EsmModel, AutoTokenizer, BitsAndBytesConfig
import io
import joblib
from transformers import BertModel, BertTokenizer
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

import re
def preprocess_protein_sequence(seq: str):

    seq = re.sub(r"[UZOB]", "X", seq)

    return ' '.join(list(seq))


class prot_bert_model:
    def __init__(self, model_name, use_cache=True, avoid_loading_model=False, rep_cache_path="./../aaseq_to_rep_store.pkl", infer_only=True):
        self.use_cache = use_cache

        self.rep_cache_path = os.path.join(os.path.dirname(__file__), rep_cache_path)
        # self.rep_cache_path = full_rep_cache_path
        if self.use_cache and os.path.exists(self.rep_cache_path):
            print(f"Loading aaseq_rep_map from {self.rep_cache_path}")
            # with open(self.rep_cache_path, "rb") as f:
            #     if torch.cuda.is_available():
            #         self.aaseq_rep_map = pickle.load(f)
            #     else:
            #         self.aaseq_rep_map = CPU_Unpickler(f).load()

            try:
                self.aaseq_rep_map = joblib.load(self.rep_cache_path)
            except Exception as e:
                print(f"Error loading aaseq_rep_map from {self.rep_cache_path}: {e}")
                self.aaseq_rep_map = {}
                # print("1A aaseq_rep_map keys: ", self.aaseq_rep_map.keys())
            # check the size of the aaseq_rep_map
            print(f"Loaded aaseq_rep_map with {len(self.aaseq_rep_map)} entries.")
        else:
            self.aaseq_rep_map = {}
            # print("1B aaseq_rep_map keys: ", self.aaseq_rep_map.keys())

        self.avoid_loading_model = avoid_loading_model
        if avoid_loading_model:
            self.fm = None
        else:
            # esm_model_name = "facebook/esm2_t30_150M_UR50D"
            if infer_only:
                # self.fm = EsmModel.from_pretrained(esm_model_name, add_pooling_layer=False).eval()
                self.fm = BertModel.from_pretrained(model_name).eval()
            else:
                self.fm = BertModel.from_pretrained(model_name)
            self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)







    def get_esm_rep_tensor(self, aaseqs, single_input_batch=False, save_cache=True, skip_return=False):
        """
        Get the esm representation of the given aaseqs string.

        If aaseqs is an iterable, then the function returns a stacked tensor of the representations.
        If aaseqs is a string, then the function returns a single tensor of the representation.
        """
        # print("get_esm_rep_tensor, aaseqs: ", aaseqs)
        if isinstance(aaseqs, str):
            aaseqs = [aaseqs]

        # aaseq = preprocess_protein_sequence(aaseq)
        aaseqs = [preprocess_protein_sequence(seq) for seq in aaseqs]  # Preprocess sequences

        # print(f"aaseqs[0]: {aaseqs[0]}")

        # test self.aaseq_rep_map
        # print("self.aaseq_rep_map keys[0]: ", list(self.aaseq_rep_map.keys())[0] if self.aaseq_rep_map else "Empty map")

        # print("get_esm_rep_tensor, aaseqs: ", aaseqs)
        # Identify the aaseqs that are already in the aaseq_rep_map.
        aaseqs_to_find_reps = []
        for aaseq in aaseqs:
            if aaseq not in self.aaseq_rep_map:
                aaseqs_to_find_reps.append(aaseq)

        # print(f"Smiles to find reps: {aaseqs_to_find_reps}")
        # Generate the representations for the aaseqs that are not in the aaseq_rep_map.
        # print("aaseq_rep_map keys: ", self.aaseq_rep_map.keys())
        # print("aaseqs_to_find_reps: ", aaseqs_to_find_reps)
        if len(aaseqs_to_find_reps) > 0:
            if self.fm != None:
                if single_input_batch:
                    pooled_protein_features = []
                    for aaseq in aaseqs_to_find_reps:
                        # print("A aaseq: ", aaseq)

                        protein_input_ids = self.tokenizer([aaseq], return_tensors="pt", add_special_tokens=True, padding=True)
                        # print("A protein_input_ids: ", protein_input_ids)
                        protein_fm_outputs = self.fm(**protein_input_ids)

                        protein_seq_features = protein_fm_outputs[0]


                        pooled_protein_feature = protein_seq_features[0, 0, :] 

                        pooled_protein_features.append(pooled_protein_feature)
                    # print("A len(pooled_protein_features): ", len(pooled_protein_features))
                else:
                    # print("aaseqs_to_find_reps: ", aaseqs_to_find_reps)
                    # sequences = [preprocess_protein_sequence(seq) for seq in sequences]  # Preprocess sequences
                    # aaseqs_to_find_reps = [preprocess_protein_sequence(seq) for seq in aaseqs_to_find_reps]
                    protein_input_ids = self.tokenizer(aaseqs_to_find_reps, return_tensors="pt", add_special_tokens=True, padding=True)
                    # protein_input_ids = tokenizer([protein_seq], return_tensors="pt", add_special_tokens=False)
                    # print("protein_input_ids: ", protein_input_ids)
                    protein_fm_outputs = self.fm(**protein_input_ids)
                    protein_seq_features = protein_fm_outputs[0]  # Assuming the output is at index 0
                    # print("protein_seq_features.shape: ", protein_seq_features.shape)
                    pooled_protein_features = protein_seq_features[:, 0, :] 
                    # print("pooled_protein_features.shape: ", pooled_protein_features.shape)

                # print("aaseqs_to_find_reps: ", aaseqs_to_find_reps)
                # generated_reps = self.fm.get_repr(aaseqs_to_find_reps)
                for aaseq, rep in zip(aaseqs_to_find_reps, pooled_protein_features):
                    self.aaseq_rep_map[aaseq] = torch.Tensor(rep)
                    # self.aaseq_rep_map[aaseq] = torch.Tensor(rep).to("cuda")
            else:
                raise

        if skip_return:
            return
        
        # Collect the representations for the queried aaseqs.
        # check if the rep is tensor, if not, then convert it to tensor.
        if not all(isinstance(self.aaseq_rep_map[aaseq], torch.Tensor) for aaseq in aaseqs):
            # print("Not all aaseq_rep_map values are tensors. Converting them to tensors.")
            reps = [torch.Tensor(self.aaseq_rep_map[aaseq]) for aaseq in aaseqs]
        else:
            reps = [self.aaseq_rep_map[aaseq].cpu() for aaseq in aaseqs]
        stacked_reps = torch.stack(reps)
        # stacked_reps = torch.stack(reps).to("cuda:0")

        # if len(stacked_reps) == 1:
        #     return stacked_reps[0]
        # else:
        return stacked_reps

__all__ = ["prot_bert_model"]
