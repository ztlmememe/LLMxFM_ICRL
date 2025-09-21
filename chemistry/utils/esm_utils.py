import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from transformers import EsmModel, AutoTokenizer, BitsAndBytesConfig
import io
import joblib
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


class esm_model:
    def __init__(self, esm_model_name, use_cache=True, avoid_loading_model=False, rep_cache_path="./../aaseq_to_rep_store.pkl", infer_only=True):
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
                self.fm = EsmModel.from_pretrained(esm_model_name, add_pooling_layer=False).eval()
            else:
                self.fm = EsmModel.from_pretrained(esm_model_name, add_pooling_layer=False)
            self.tokenizer = AutoTokenizer.from_pretrained(esm_model_name)




    def get_esm_rep_tensor(self, aaseqs, single_input_batch=False, save_cache=True, skip_return=False):
        """
        Get the esm representation of the given aaseqs string.

        If aaseqs is an iterable, then the function returns a stacked tensor of the representations.
        If aaseqs is a string, then the function returns a single tensor of the representation.
        """
        # print("get_esm_rep_tensor, aaseqs: ", aaseqs)
        if isinstance(aaseqs, str):
            aaseqs = [aaseqs]

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
                        protein_input_ids = self.tokenizer([aaseq], return_tensors="pt", add_special_tokens=False, padding=True)
                        # print("A protein_input_ids: ", protein_input_ids)
                        protein_fm_outputs = self.fm(**protein_input_ids)

                        # protein_fm_outputs: BaseModelOutputWithPoolingAndCrossAttentions(
                        #         last_hidden_state=sequence_output,
                        #         pooler_output=pooled_output,
                        #         past_key_values=encoder_outputs.past_key_values,
                        #         hidden_states=encoder_outputs.hidden_states,
                        #         attentions=encoder_outputs.attentions,
                        #         cross_attentions=encoder_outputs.cross_attentions,
                        # )
                        protein_seq_features = protein_fm_outputs[0]
                        # print("A protein_seq_features.shape: ", protein_seq_features.shape)
                        #  torch.Size([1, 43, 640])

                        pooled_protein_feature = protein_seq_features[0, 0, :] 
                        # print("A pooled_protein_feature.shape: ", pooled_protein_feature.shape)
                        # torch.Size([640]) cls token

                        pooled_protein_features.append(pooled_protein_feature)
                    # print("A len(pooled_protein_features): ", len(pooled_protein_features))
                else:
                    # print("aaseqs_to_find_reps: ", aaseqs_to_find_reps)
                    protein_input_ids = self.tokenizer(aaseqs_to_find_reps, return_tensors="pt", add_special_tokens=False, padding=True)
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
            
            if save_cache:
                # Save the aaseq_rep_map.
                # pk_file_path = os.path.join(os.path.dirname(__file__), "./../aaseq_to_rep_store.pkl")
                pk_file_path = self.rep_cache_path
                
                # before saving, check if the file is already there. merge the aaseq_rep_map dictionaries if it is.
                if os.path.exists(pk_file_path):
                    with open(pk_file_path, "rb") as f:
                        if torch.cuda.is_available():
                            existing_aaseq_rep_map = pickle.load(f)
                        else:
                            existing_aaseq_rep_map = CPU_Unpickler(f).load()

                    existing_aaseq_rep_map.update(self.aaseq_rep_map)
                    self.aaseq_rep_map = existing_aaseq_rep_map

                try:
                    # Use a temporary file to avoid overwriting the original in case of errors
                    temp_file_path = pk_file_path + ".tmp"
                    with open(temp_file_path, "wb") as f:
                        pickle.dump(self.aaseq_rep_map, f)

                    # If successful, replace the original file with the temporary one
                    os.replace(temp_file_path, pk_file_path) 

                    print("Saved aaseq_to_rep_store to cache.")

                except (pickle.PicklingError, OSError) as e:
                    # Handle potential pickling or file system errors
                    print(f"Error saving aaseq_rep_map: {e}")
                    # You might want to log this error or take other corrective actions
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)  # Clean up the temporary file

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
    

    def compress_tensor(self,tensor, method="mean", compress_ratio=0.2, positions=None):
        """
        Compress the tensor along the sequence dimension using various methods, supporting percentage-based compression.

        Args:
            tensor (torch.Tensor): The input tensor of shape (seq_length, feature_dim).
            method (str): Compression method ("mean", "fixed_positions").
            compress_ratio (float): Compression ratio (e.g., 0.2 means 20% of the original size).
            positions (list): Fixed positions to keep (used for "fixed_positions" method).

        Returns:
            torch.Tensor: The compressed tensor.
        """
        # Track whether the input had a batch dimension and handle it by squeezing
        had_batch_dim = False
        if len(tensor.shape) == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
            had_batch_dim = True
        elif len(tensor.shape) != 2:
            raise ValueError(f"Expected a 2D tensor, but got tensor with shape: {tensor.shape}")

        seq_length, feature_dim = tensor.shape

        # print("compress_tensor, tensor.shape: ", tensor.shape)

        if method == "mean":
            # Ensure compress_ratio is valid
            if not (0 < compress_ratio <= 1):
                raise ValueError("compress_ratio must be in the range (0, 1].")

            # Calculate chunk size based on the compress_ratio
            chunk_size = max(1, int(1 / compress_ratio))  # Each chunk will represent 1/compress_ratio of the sequence

            # Generate indices for chunks, handling uneven last chunk
            indices = torch.arange(seq_length).split(chunk_size)

            # Apply mean pooling over chunks
            compressed_chunks = [tensor[idx].mean(dim=0, keepdim=True) for idx in indices]

            # Concatenate results to form the compressed tensor
            compressed_tensor = torch.cat(compressed_chunks, dim=0)

        elif method == "fixed_positions":
            # Validate positions argument
            if positions is None:
                raise ValueError("`positions` must be specified for the 'fixed_positions' method.")
            if not all(0 <= pos < seq_length for pos in positions):
                raise ValueError("`positions` contains indices out of tensor sequence length.")

            # Select the specified positions
            compressed_tensor = tensor[positions]

        else:
            raise ValueError("Unsupported compression method. Use 'mean' or 'fixed_positions'.")

        # Restore batch dimension if the input had it
        if had_batch_dim:
            compressed_tensor = compressed_tensor.unsqueeze(0)

        return compressed_tensor


    def get_esm_rep_tensor_v2(self, aaseqs, single_input_batch=False, save_cache=True, skip_return=False,
                              compress_method=None, compress_ratio=0.2, compress_positions=None):
        """
        Get the esm representation of the given aaseqs string.

        If aaseqs is an iterable, then the function returns a stacked tensor of the representations.
        If aaseqs is a string, then the function returns a single tensor of the representation.
        """
        # print("get_esm_rep_tensor, aaseqs: ", aaseqs)
        if isinstance(aaseqs, str):
            aaseqs = [aaseqs]

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
                        protein_input_ids = self.tokenizer([aaseq], return_tensors="pt", add_special_tokens=False, padding=True)
                        # print("A protein_input_ids: ", protein_input_ids)
                        protein_fm_outputs = self.fm(**protein_input_ids)

                        protein_seq_features = protein_fm_outputs[0]
                        # print("A protein_seq_features.shape: ", protein_seq_features.shape)
                        #  torch.Size([1, 43, 640])

                        if compress_method:
                            pooled_protein_feature = self.compress_tensor(
                                protein_seq_features[0, 1:, :],
                                method=compress_method,
                                compress_ratio= compress_ratio,
                                positions=compress_positions
                            )
                            # print("A protein_seq_features.shape: ", protein_seq_features.shape)
                            # torch.Size([1, 236, 640]) -> torch.Size([1, 48, 640])
                        else:
                            pooled_protein_feature = protein_seq_features[0, 1:, :] 

                        # pooled_protein_feature = protein_seq_features[0, 1:, :] 
                        # print("A pooled_protein_feature.shape: ", pooled_protein_feature.shape)
                        # V1: torch.Size([640])
                        # V2: torch.Size([285, 640])

                        pooled_protein_features.append(pooled_protein_feature)
                    # print("A len(pooled_protein_features): ", len(pooled_protein_features))

                # print("aaseqs_to_find_reps: ", aaseqs_to_find_reps)
                # generated_reps = self.fm.get_repr(aaseqs_to_find_reps)
                for aaseq, rep in zip(aaseqs_to_find_reps, pooled_protein_features):
                    self.aaseq_rep_map[aaseq] = torch.Tensor(rep)
                    # self.aaseq_rep_map[aaseq] = torch.Tensor(rep).to("cuda")
            else:
                raise
            
            if save_cache:
                # Save the aaseq_rep_map.
                pk_file_path = self.rep_cache_path
                
                # before saving, check if the file is already there. merge the aaseq_rep_map dictionaries if it is.
                if os.path.exists(pk_file_path):
                    with open(pk_file_path, "rb") as f:
                        if torch.cuda.is_available():
                            existing_aaseq_rep_map = pickle.load(f)
                        else:
                            existing_aaseq_rep_map = CPU_Unpickler(f).load()

                    existing_aaseq_rep_map.update(self.aaseq_rep_map)
                    self.aaseq_rep_map = existing_aaseq_rep_map

                try:
                    # Use a temporary file to avoid overwriting the original in case of errors
                    temp_file_path = pk_file_path + ".tmp"
                    with open(temp_file_path, "wb") as f:
                        pickle.dump(self.aaseq_rep_map, f)

                    # If successful, replace the original file with the temporary one
                    os.replace(temp_file_path, pk_file_path) 

                    print("Saved aaseq_to_rep_store to cache.")

                except (pickle.PicklingError, OSError) as e:
                    # Handle potential pickling or file system errors
                    print(f"Error saving aaseq_rep_map: {e}")
                    # You might want to log this error or take other corrective actions
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)  # Clean up the temporary file

        if skip_return:
            return
        
        # Collect the representations for the queried aaseqs.
        reps = [self.aaseq_rep_map[aaseq].cpu() for aaseq in aaseqs]
        stacked_reps = torch.stack(reps)
        # stacked_reps = torch.stack(reps).to("cuda:0")

        # if len(stacked_reps) == 1:
        #     return stacked_reps[0]
        # else:
        return stacked_reps


    def get_esm_shallow_and_deep_rep(self, aaseqs, layer_indices, single_input_batch=False):
        """
        Get the esm representation of the given aaseqs string from specified layers.

        """
        if isinstance(aaseqs, str):
            aaseqs = [aaseqs]

        protein_input_ids = self.tokenizer(
            aaseqs,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True
        )

        if self.fm is not None:
            self.fm.config.output_hidden_states = True
        else:
            raise ValueError("The ESM model is not loaded.")

        if single_input_batch:
            layer_outputs = {f"Layer {idx}": [] for idx in layer_indices}

            for aaseq in aaseqs:
                single_input = self.tokenizer([aaseq], return_tensors="pt", add_special_tokens=False, padding=True)
                outputs = self.fm(**single_input)

                hidden_states = outputs.hidden_states
                for idx in layer_indices:
                    tmp = hidden_states[idx]
                    protein_seq_features = tmp[0]
                    # print("A protein_seq_features.shape: ", protein_seq_features.shape)
                    pooled_protein_feature = protein_seq_features[0,:] 
                    layer_outputs[f"Layer {idx}"].append(pooled_protein_feature)

            for key in layer_outputs:
                layer_outputs[key] = torch.stack(layer_outputs[key])

            return layer_outputs


__all__ = ["esm_model"]
