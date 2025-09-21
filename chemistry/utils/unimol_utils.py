import os
import pickle
import numpy as np
from unimol_tools import UniMolRepr
import torch
from tqdm import tqdm
import io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


class unimol_clf:
    def __init__(self, use_cache=True, avoid_loading_model=False, rep_cache_path="./../smile_to_rep_store.pkl"):
        self.use_cache = use_cache

        self.full_rep_cache_path = os.path.join(os.path.dirname(__file__), rep_cache_path)
        if self.use_cache and os.path.exists(self.full_rep_cache_path):
            with open(self.full_rep_cache_path, "rb") as f:
                if torch.cuda.is_available():
                    self.smile_rep_map = pickle.load(f)
                    # print("load smile_rep_map A, self.smile_rep_map: ", self.smile_rep_map)
                else:
                    self.smile_rep_map = CPU_Unpickler(f).load()
                    # print("load smile_rep_map B, self.smile_rep_map: ", self.smile_rep_map)

                # print("1A smile_rep_map keys: ", self.smile_rep_map.keys())

                print(f"Loaded smile_rep_map from cache path: {self.full_rep_cache_path}")
                print(f"Number of cached representations: {len(self.smile_rep_map)}")
        else:
            self.smile_rep_map = {}
            # print("1B smile_rep_map keys: ", self.smile_rep_map.keys())

        self.avoid_loading_model = avoid_loading_model
        if avoid_loading_model:
            self.clf = None
        else:
            self.clf = UniMolRepr(data_type="molecule", remove_hs=False)

    def get_unimol_rep_string(self, smiles):
        if self.clf != None:
            reps = self.clf.get_repr(smiles)["cls_repr"]
            return [np.array2string(r[:10].round(3)) for r in reps]
        else:
            raise 

    def get_unimol_rep_tensor(self, smiles, save_cache=True, skip_return=False):
        """
        Get the unimol representation of the given smiles string.

        If smiles is an iterable, then the function returns a stacked tensor of the representations.
        If smiles is a string, then the function returns a single tensor of the representation.
        """
        if isinstance(smiles, str):
            smiles = [smiles]

        # print("get_unimol_rep_tensor, smiles: ", smiles)
        # Identify the smiles that are already in the smile_rep_map.
        smiles_to_find_reps = []
        for smile in smiles:
            if smile not in self.smile_rep_map:
                smiles_to_find_reps.append(smile)

        # Generate the representations for the smiles that are not in the smile_rep_map.
        # print("smile_rep_map keys: ", self.smile_rep_map.keys())
        # print("smiles_to_find_reps: ", smiles_to_find_reps)
        if len(smiles_to_find_reps) > 0:
            if self.clf != None:
                # print("smiles_to_find_reps: ", smiles_to_find_reps)
                generated_reps = self.clf.get_repr(smiles_to_find_reps)
                # print("generated_reps: ", generated_reps)
                for smile, rep in zip(smiles_to_find_reps, generated_reps["cls_repr"]):
                    self.smile_rep_map[smile] = torch.Tensor(rep)
                    # self.smile_rep_map[smile] = torch.Tensor(rep).to("cuda")
            else:
                raise Exception("The model has not been loaded. Please set avoid_loading_model to True.")

            if save_cache:
                # Save the smile_rep_map.
                # pk_file_path = os.path.join(os.path.dirname(__file__), "./../smile_to_rep_store.pkl")

                # before saving, check if the file is already there. merge the smile_rep_map dictionaries if it is.
                if os.path.exists(self.full_rep_cache_path):
                    with open(self.full_rep_cache_path, "rb") as f:
                        if torch.cuda.is_available():
                            existing_smile_rep_map = pickle.load(f)
                        else:
                            existing_smile_rep_map = CPU_Unpickler(f).load()

                    existing_smile_rep_map.update(self.smile_rep_map)
                    self.smile_rep_map = existing_smile_rep_map

                try:
                    # Use a temporary file to avoid overwriting the original in case of errors
                    temp_file_path = self.full_rep_cache_path + ".tmp"
                    with open(temp_file_path, "wb") as f:
                        pickle.dump(self.smile_rep_map, f)

                    # If successful, replace the original file with the temporary one
                    os.replace(temp_file_path, self.full_rep_cache_path) 

                    print("Saved smile_rep_map to cache.")

                except (pickle.PicklingError, OSError) as e:
                    # Handle potential pickling or file system errors
                    print(f"Error saving smile_rep_map {temp_file_path}: {e}")
                    # You might want to log this error or take other corrective actions
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)  # Clean up the temporary file

            # with open(os.path.join(os.path.dirname(__file__), "./../smile_to_rep_store.pkl"), "wb") as f:
            #     pickle.dump(self.smile_rep_map, f)

        if skip_return:
            return
        
        # Collect the representations for the queried smiles.
        reps = [self.smile_rep_map[smile].cpu() for smile in smiles]
        # reps = [self.smile_rep_map[smile] for smile in smiles]
        # print("reps: ", reps)
        # print shape of each rep
        # for i, rep in enumerate(reps):
        #     print(i, " len(rep.shape): ", len(rep.shape), " rep.shape: ", rep.shape)

        stacked_reps = torch.stack(reps)
        # stacked_reps = torch.stack(reps).to("cuda:0")

        # if len(stacked_reps) == 1:
        #     return stacked_reps[0]
        # else:
        return stacked_reps
        # return stacked_reps

    def get_unimol_rep_tensor_v2(self, smiles, save_cache=True, skip_return=False):
        """
        Get the unimol representation of the given smiles string.

        If smiles is an iterable, then the function returns a stacked tensor of the representations.
        If smiles is a string, then the function returns a single tensor of the representation.
        """
        if isinstance(smiles, str):
            smiles = [smiles]

        # print("get_unimol_rep_tensor, smiles: ", smiles)
        # Identify the smiles that are already in the smile_rep_map.
        smiles_to_find_reps = []
        for smile in smiles:
            if smile not in self.smile_rep_map:
                smiles_to_find_reps.append(smile)

        # Generate the representations for the smiles that are not in the smile_rep_map.
        # print("smile_rep_map keys: ", self.smile_rep_map.keys())
        # print("smiles_to_find_reps: ", smiles_to_find_reps)
        if len(smiles_to_find_reps) > 0:
            if self.clf != None:
                # print("smiles_to_find_reps: ", smiles_to_find_reps)
                generated_reps = self.clf.get_repr_v2(smiles_to_find_reps)
                print("generated_reps.shape: ", generated_reps.shape)


                        #                 if compress_method:
                        #     pooled_protein_feature = self.compress_tensor(
                        #         protein_seq_features[0, 1:, :],
                        #         method=compress_method,
                        #         compress_ratio= compress_ratio,
                        #         positions=compress_positions
                        #     )
                        #     # print("A protein_seq_features.shape: ", protein_seq_features.shape)
                        #     # torch.Size([1, 236, 640]) -> torch.Size([1, 48, 640])
                        # else:
                        #     pooled_protein_feature = protein_seq_features[0, 1:, :] 
                # print("generated_reps: ", generated_reps)
                for smile, rep in zip(smiles_to_find_reps, generated_reps["cls_repr"]):
                    self.smile_rep_map[smile] = torch.Tensor(rep)
                    # self.smile_rep_map[smile] = torch.Tensor(rep).to("cuda")
            else:
                raise Exception("The model has not been loaded. Please set avoid_loading_model to True.")

            if save_cache:
                # Save the smile_rep_map.
                # pk_file_path = os.path.join(os.path.dirname(__file__), "./../smile_to_rep_store.pkl")

                # before saving, check if the file is already there. merge the smile_rep_map dictionaries if it is.
                if os.path.exists(self.full_rep_cache_path):
                    with open(self.full_rep_cache_path, "rb") as f:
                        if torch.cuda.is_available():
                            existing_smile_rep_map = pickle.load(f)
                        else:
                            existing_smile_rep_map = CPU_Unpickler(f).load()

                    existing_smile_rep_map.update(self.smile_rep_map)
                    self.smile_rep_map = existing_smile_rep_map

                try:
                    # Use a temporary file to avoid overwriting the original in case of errors
                    temp_file_path = self.full_rep_cache_path + ".tmp"
                    with open(temp_file_path, "wb") as f:
                        pickle.dump(self.smile_rep_map, f)

                    # If successful, replace the original file with the temporary one
                    os.replace(temp_file_path, self.full_rep_cache_path) 

                    print("Saved smile_rep_map to cache.")

                except (pickle.PicklingError, OSError) as e:
                    # Handle potential pickling or file system errors
                    print(f"Error saving smile_rep_map {temp_file_path}: {e}")
                    # You might want to log this error or take other corrective actions
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)  # Clean up the temporary file

            # with open(os.path.join(os.path.dirname(__file__), "./../smile_to_rep_store.pkl"), "wb") as f:
            #     pickle.dump(self.smile_rep_map, f)

        if skip_return:
            return
        
        # Collect the representations for the queried smiles.
        reps = [self.smile_rep_map[smile].cpu() for smile in smiles]
        # reps = [self.smile_rep_map[smile] for smile in smiles]
        # print("reps: ", reps)
        # print shape of each rep
        # for i, rep in enumerate(reps):
        #     print(i, " len(rep.shape): ", len(rep.shape), " rep.shape: ", rep.shape)

        stacked_reps = torch.stack(reps)
        # stacked_reps = torch.stack(reps).to("cuda:0")

        # if len(stacked_reps) == 1:
        #     return stacked_reps[0]
        # else:
        return stacked_reps
        # return stacked_reps

    def get_unimol_shallow_and_deep_rep(self, smiles, layer_indices=None, save_cache=True, skip_return=False):
        if isinstance(smiles, str):
            smiles = [smiles]

        smiles_to_find_reps = []
        for smile in smiles:
            smiles_to_find_reps.append(smile)

        # print(f"Smiles to find reps: {smiles_to_find_reps}")

        if len(smiles_to_find_reps) > 0:
            if self.clf is not None:
                layer_outputs = {idx: [] for idx in layer_indices} if layer_indices else {}

                # def create_hook_fn(layer_idx):
                #     def hook_fn(module, input, output):
                #         print(f"Layer {layer_idx} output type: {type(output)}")
                #         if isinstance(output, tuple):
                #             print(f"Layer {layer_idx} output contains {len(output)} elements")
                #             for i, item in enumerate(output):
                #                 print(f"Output[{i}] shape: {item.shape}")
                #         elif isinstance(output, torch.Tensor):
                #             print(f"Layer {layer_idx} output shape: {output.shape}")
                #         layer_outputs[layer_idx].append(output.detach().cpu())
                #     return hook_fn

                # Layer 1 output type: <class 'tuple'>
                # Layer 1 output contains 3 elements
                # Output[0] shape: torch.Size([1, 44, 512])
                # Output[1] shape: torch.Size([64, 44, 44])
                # Output[2] shape: torch.Size([64, 44, 44])
                                        
                def create_hook_fn(layer_idx):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            tmp_1 = output[0].detach().cpu()
                            tmp_2 = tmp_1[0]
                            pooled_mol_feature = tmp_2[0, :] 
                            # print(f"Layer {layer_idx} output shape: {pooled_mol_feature.shape}")
                            layer_outputs[layer_idx].append(pooled_mol_feature)
                        else:
                            tmp = output.detach().cpu()
                            pooled_mol_feature = tmp[0, 0, :] 
                            layer_outputs[layer_idx].append(pooled_mol_feature)
                    return hook_fn


                hooks = []
                # print(f"Layer indices: {layer_indices}")
                if layer_indices is not None:
                    # print(f"Layer indices: {layer_indices}")
                    for idx in layer_indices:
                        target_layer = self.clf.model.encoder.layers[idx]
                        # print(f"Hooking layer {target_layer}")
                        # Hooking layer TransformerEncoderLayer(
                        #   (self_attn): SelfMultiheadAttention(
                        #     (in_proj): Linear(in_features=512, out_features=1536, bias=True)
                        #     (out_proj): Linear(in_features=512, out_features=512, bias=True)
                        #   )
                        #   (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                        #   (fc1): Linear(in_features=512, out_features=2048, bias=True)
                        #   (fc2): Linear(in_features=2048, out_features=512, bias=True)
                        #   (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                        # )
                        hooks.append(target_layer.register_forward_hook(create_hook_fn(idx)))

                generated_reps = self.clf.get_repr(smiles_to_find_reps)['cls_repr']

                # print(f"Generated reps: {len(generated_reps[0])}") 512

                for hook in hooks:
                    hook.remove()

                for smile in smiles_to_find_reps:
                    self.smile_rep_map[smile] = {
                        idx: torch.stack(layer_outputs[idx]) for idx in layer_indices
                    }
            else:
                raise Exception("The model has not been loaded. Please set avoid_loading_model to True.")

        if skip_return:
            return

        if layer_indices is not None:
            reps = [
                {idx: self.smile_rep_map[smile][idx] for idx in layer_indices} for smile in smiles
            ]
            return {idx: torch.stack([rep[idx] for rep in reps]).squeeze(dim=1) for idx in layer_indices}
        else:
            reps = [self.smile_rep_map[smile].cpu() for smile in smiles]
            return torch.stack(reps).squeeze(dim=1)


__all__ = ["unimol_clf"]
