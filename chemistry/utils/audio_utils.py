import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
import io
from transformers import AutoProcessor, AutoModel
from datasets import load_dataset
import torch.nn.functional as F
import random
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import librosa

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def load_and_prepare_esc50():
    """
    Load and prepare the ESC-50 dataset
    
    Returns:
    - Dictionary containing training, validation, and testing splits
    """
    dataset = load_dataset("ashraq/esc50")
    
    # Create train/val/test splits
    train_ds = dataset["train"].filter(lambda x: x["fold"] < 4)
    valid_ds = dataset["train"].filter(lambda x: x["fold"] == 4)
    test_ds = dataset["train"].filter(lambda x: x["fold"] == 5)
    
    # Extract relevant data
    train_audio = train_ds["audio"]
    train_labels = train_ds["target"]
    valid_audio = valid_ds["audio"]
    valid_labels = valid_ds["target"]
    test_audio = test_ds["audio"]
    test_labels = test_ds["target"]
    
    # Convert to numpy arrays if needed
    train_inputs = [item["array"] for item in train_audio]
    train_srs = [item["sampling_rate"] for item in train_audio]
    valid_inputs = [item["array"] for item in valid_audio]
    valid_srs = [item["sampling_rate"] for item in valid_audio]
    test_inputs = [item["array"] for item in test_audio]
    test_srs = [item["sampling_rate"] for item in test_audio]
    
    return {
        "train": {"inputs": train_inputs, "sampling_rates": train_srs, "labels": train_labels},
        "valid": {"inputs": valid_inputs, "sampling_rates": valid_srs, "labels": valid_labels},
        "test": {"inputs": test_inputs, "sampling_rates": test_srs, "labels": test_labels}
    }

def load_and_prepare_vggsound(max_samples_per_class=100):
    """
    Load and prepare the VGGSound dataset with a limit on samples per class
    
    Parameters:
    - max_samples_per_class: Maximum samples to include per class to keep dataset manageable
    
    Returns:
    - Dictionary containing training, validation, and testing splits
    """
    try:
        dataset = load_dataset("Loie/VGGSound")
        
        # Count samples per class
        class_counts = {}
        filtered_dataset = {"train": [], "test": []}
        
        # Filter training set
        for sample in dataset["train"]:
            label = sample["label"]
            if label not in class_counts:
                class_counts[label] = 0
            
            if class_counts[label] < max_samples_per_class:
                filtered_dataset["train"].append(sample)
                class_counts[label] += 1
        
        # Reset counts for test set
        class_counts = {}
        
        # Filter test set
        for sample in dataset["test"]:
            label = sample["label"]
            if label not in class_counts:
                class_counts[label] = 0
            
            if class_counts[label] < max_samples_per_class:
                filtered_dataset["test"].append(sample)
                class_counts[label] += 1
        
        # Create train/val split from filtered training data
        train_size = int(0.8 * len(filtered_dataset["train"]))
        train_ds = filtered_dataset["train"][:train_size]
        valid_ds = filtered_dataset["train"][train_size:]
        test_ds = filtered_dataset["test"]
        # shuffle the data

        # random.shuffle(train_ds)
        # random.shuffle(valid_ds)
        # random.shuffle(test_ds)
        
        # Extract relevant data
        train_audio = [item["audio"]["array"] for item in train_ds]
        train_srs = [item["audio"]["sampling_rate"] for item in train_ds]
        train_labels = [item["label"] for item in train_ds]
        
        valid_audio = [item["audio"]["array"] for item in valid_ds]
        valid_srs = [item["audio"]["sampling_rate"] for item in valid_ds]
        valid_labels = [item["label"] for item in valid_ds]
        
        test_audio = [item["audio"]["array"] for item in test_ds]
        test_srs = [item["audio"]["sampling_rate"] for item in test_ds]
        test_labels = [item["label"] for item in test_ds]
        
        return {
            "train": {"inputs": train_audio, "sampling_rates": train_srs, "labels": train_labels},
            "valid": {"inputs": valid_audio, "sampling_rates": valid_srs, "labels": valid_labels},
            "test": {"inputs": test_audio, "sampling_rates": test_srs, "labels": test_labels}
        }
    except Exception as e:
        print(f"Error loading VGGSound dataset: {e}")
        return None

class AudioRepr:

    def __init__(self, model_name="facebook/wav2vec2-base-960h", use_cache=True, avoid_loading_model=False, rep_cache_path="./../audio_to_rep_store.pkl"):
        self.use_cache = use_cache
        self.model_name = model_name

        self.full_rep_cache_path = os.path.join(os.path.dirname(__file__), rep_cache_path)
        if self.use_cache and os.path.exists(self.full_rep_cache_path):
            with open(self.full_rep_cache_path, "rb") as f:
                if torch.cuda.is_available():
                    self.audio_rep_map = pickle.load(f)
                else:
                    self.audio_rep_map = CPU_Unpickler(f).load()
        else:
            self.audio_rep_map = {}

        self.avoid_loading_model = avoid_loading_model
        if avoid_loading_model:
            self.model = None
            self.processor = None
        else:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
                # Store the model's dtype for later use to match input types
                self.model_dtype = next(self.model.parameters()).dtype
            else:
                self.model_dtype = torch.float32

    def get_wav2vec_rep_string(self, audio_ids):
        """Get string representation of audio features for display"""
        if self.model is not None:
            reps = self.get_wav2vec_rep_tensor(audio_ids)
            return [np.array2string(r[:10].cpu().numpy().round(3)) for r in reps]
        else:
            raise Exception("The model has not been loaded. Please set avoid_loading_model to False.")
            
    def resample_audio(self, audio_data, orig_sampling_rate, target_sampling_rate=16000):
        """
        Resample audio data to the target sampling rate
        
        Parameters:
        - audio_data: Audio waveform
        - orig_sampling_rate: Original sampling rate
        - target_sampling_rate: Target sampling rate (default: 16000 for wav2vec2)
        
        Returns:
        - Resampled audio data
        """
        if orig_sampling_rate == target_sampling_rate:
            return audio_data
            
        # Use librosa for resampling
        try:
            resampled_audio = librosa.resample(
                y=audio_data.astype(np.float32), 
                orig_sr=orig_sampling_rate,
                target_sr=target_sampling_rate
            )
            return resampled_audio
        except Exception as e:
            print(f"Error during resampling: {e}")
            # Return a short segment of silence as fallback
            return np.zeros(target_sampling_rate // 2, dtype=np.float32)
            
    def extract_features(self, audio_data, sampling_rate):
        """Extract features from raw audio data"""
        try:
            # Ensure audio is in the expected format
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data)
            
            # Check if audio data is empty
            if audio_data.size == 0:
                print(f"Empty audio data detected, using fallback")
                audio_data = np.zeros(16000 // 2, dtype=np.float32)
                sampling_rate = 16000
            
            # Ensure the audio has the expected sampling rate (16kHz for wav2vec2)
            target_sr = 16000  # wav2vec2 expected sampling rate
            if sampling_rate != target_sr:
                print(f"Resampling from {sampling_rate}Hz to {target_sr}Hz")
                audio_data = self.resample_audio(audio_data, sampling_rate, target_sr)
                sampling_rate = target_sr
            
            # Process the audio input
            inputs = self.processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt")
            
            if torch.cuda.is_available():
                # Convert inputs to the same dtype as the model to avoid type mismatch
                inputs = {k: v.to("cuda").to(self.model_dtype) for k, v in inputs.items()}
            
            # Get the model output
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Use the mean of the last hidden state as the representation
            # Shape: [batch_size, sequence_length, hidden_size]
            last_hidden = outputs.last_hidden_state
            
            # Mean pool across the sequence dimension to get a fixed-size representation
            # Shape: [batch_size, hidden_size]
            mean_pooled = torch.mean(last_hidden, dim=1)
            
            return mean_pooled
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            # Create a fallback tensor with the correct shape and dtype
            hidden_size = self.model.config.hidden_size
            fallback = torch.zeros((1, hidden_size), 
                                  device="cuda" if torch.cuda.is_available() else "cpu",
                                  dtype=self.model_dtype if torch.cuda.is_available() else torch.float32)
            return fallback


    def get_wav2vec_rep_tensor(self, audio_data_list, sampling_rate_list=None, audio_ids=None, save_cache=True, skip_return=False):
        """
        Get the wav2vec representation of the given audio data.
        
        Parameters:
        - audio_data_list: List of audio waveforms
        - sampling_rate_list: List of sampling rates for each audio
        - audio_ids: Optional unique identifiers for caching
        - save_cache: Whether to save representations to cache
        - skip_return: If True, just update cache without returning tensors
        
        Returns:
        - Tensor of audio representations
        """
        if self.model is None:
            raise Exception("The model has not been loaded. Please set avoid_loading_model to False.")
        
        # If audio_ids not provided, don't use caching
        if audio_ids is None:
            use_cache = False
        else:
            use_cache = self.use_cache
            if isinstance(audio_ids, str):
                audio_ids = [audio_ids]
        
        # Standardize input format
        if not isinstance(audio_data_list, list):
            audio_data_list = [audio_data_list]
        
        if sampling_rate_list is None:
            # Default to 16kHz if not specified
            sampling_rate_list = [16000] * len(audio_data_list)
        elif not isinstance(sampling_rate_list, list):
            sampling_rate_list = [sampling_rate_list] * len(audio_data_list)
        
        # Check cache for existing representations
        audio_to_process = []
        indices_to_process = []
        
        if use_cache:
            for i, audio_id in enumerate(audio_ids):
                if audio_id not in self.audio_rep_map:
                    audio_to_process.append(audio_data_list[i])
                    indices_to_process.append(i)
        else:
            audio_to_process = audio_data_list
            indices_to_process = list(range(len(audio_data_list)))
        
        # Process audios not in cache
        processed_indices = []
        if audio_to_process:
            for i, idx in enumerate(indices_to_process):
                try:
                    audio = audio_data_list[idx]
                    sample_rate = sampling_rate_list[idx]
                    
                    # Extract features
                    rep = self.extract_features(audio, sample_rate)
                    
                    # Store in cache if we have an ID
                    if use_cache:
                        self.audio_rep_map[audio_ids[idx]] = rep
                        processed_indices.append(idx)
                except Exception as e:
                    print(f"Error processing audio sample {idx}: {e}")
                    # Create a fallback tensor with the correct shape and dtype
                    hidden_size = self.model.config.hidden_size
                    fallback = torch.zeros((1, hidden_size), 
                                         device="cuda" if torch.cuda.is_available() else "cpu",
                                         dtype=self.model_dtype if torch.cuda.is_available() else torch.float32)
                    if use_cache:
                        self.audio_rep_map[audio_ids[idx]] = fallback
                        processed_indices.append(idx)
        
        # Save updated cache if requested
        if use_cache and save_cache and processed_indices:
            try:
                # Use a temporary file to avoid overwriting the original in case of errors
                temp_file_path = self.full_rep_cache_path + ".tmp"
                with open(temp_file_path, "wb") as f:
                    pickle.dump(self.audio_rep_map, f)
                
                # If successful, replace the original file with the temporary one
                os.replace(temp_file_path, self.full_rep_cache_path)
                print("Saved audio_rep_map to cache.")
                
            except (pickle.PicklingError, OSError) as e:
                print(f"Error saving audio_rep_map {temp_file_path}: {e}")
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)  # Clean up the temporary file
        
        # Return the representations if requested
        if skip_return:
            return
        
        # Get the correct hidden size for fallback representations
        hidden_size = self.model.config.hidden_size
        fallback_tensor = torch.zeros((1, hidden_size), 
                                     device="cuda" if torch.cuda.is_available() else "cpu",
                                     dtype=self.model_dtype if torch.cuda.is_available() else torch.float32)
        
        if use_cache:
            # Collect cached representations with fallback for any missing entries
            reps = []
            for audio_id in audio_ids:
                if audio_id in self.audio_rep_map:
                    rep = self.audio_rep_map[audio_id]
                    # Ensure rep is on CPU before adding to list
                    reps.append(rep.cpu() if torch.cuda.is_available() else rep)
                else:
                    print(f"Warning: No representation found for {audio_id}, using fallback")
                    reps.append(fallback_tensor.cpu() if torch.cuda.is_available() else fallback_tensor)
        else:
            # Process all audio data if not using cache
            reps = []
            for i, (audio, sr) in enumerate(zip(audio_data_list, sampling_rate_list)):
                try:
                    rep = self.extract_features(audio, sr)
                    # Ensure rep is on CPU before adding to list
                    reps.append(rep.cpu() if torch.cuda.is_available() else rep)
                except Exception as e:
                    print(f"Error extracting features for sample {i}, using fallback: {e}")
                    reps.append(fallback_tensor.cpu() if torch.cuda.is_available() else fallback_tensor)
        
        # Make sure we have at least one representation
        if len(reps) == 0:
            print("Warning: No valid representations found, using fallback")
            reps = [fallback_tensor.cpu() if torch.cuda.is_available() else fallback_tensor]
        
        # Stack the representations into a single tensor
        try:
            stacked_reps = torch.stack(reps)
            return stacked_reps
        except Exception as e:
            print(f"Error stacking representations: {e}")
            # Return a single fallback tensor with batch dimension
            return fallback_tensor.unsqueeze(0).cpu() if torch.cuda.is_available() else fallback_tensor.unsqueeze(0)
    
def data_split_loading_audio(dataset_name="ESC50", split_type='random', seed=42, split_fracs=None):
    if split_fracs is None:
        split_fracs = [0.7, 0.1, 0.2]
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if dataset_name == "ESC50":
        data = load_and_prepare_esc50()
    elif dataset_name == "VGGSound":
        data = load_and_prepare_vggsound()
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    if split_type == 'random':
        # If we want to override the original splits
        if dataset_name == "ESC50" and split_fracs != [0.7, 0.1, 0.2]:
            # Combine all data
            all_inputs = data["train"]["inputs"] + data["valid"]["inputs"] + data["test"]["inputs"]
            all_srs = data["train"]["sampling_rates"] + data["valid"]["sampling_rates"] + data["test"]["sampling_rates"]
            all_labels = data["train"]["labels"] + data["valid"]["labels"] + data["test"]["labels"]
            
            # Create new splits
            train_size = int(split_fracs[0] * len(all_inputs))
            val_size = int(split_fracs[1] * len(all_inputs))
            
            train_inputs = all_inputs[:train_size]
            train_srs = all_srs[:train_size]
            train_labels = all_labels[:train_size]
            
            valid_inputs = all_inputs[train_size:train_size+val_size]
            valid_srs = all_srs[train_size:train_size+val_size]
            valid_labels = all_labels[train_size:train_size+val_size]
            
            test_inputs = all_inputs[train_size+val_size:]
            test_srs = all_srs[train_size+val_size:]
            test_labels = all_labels[train_size+val_size:]
            
            return {
                "train": {"AUDIO": train_inputs, "SampleRates": train_srs, "Labels": train_labels},
                "valid": {"AUDIO": valid_inputs, "SampleRates": valid_srs, "Labels": valid_labels},
                "test": {"AUDIO": test_inputs, "SampleRates": test_srs, "Labels": test_labels}
            }
    
    # Use the original splits if split_type isn't 'random' or fractions match defaults
    return {
        "train": {"AUDIO": data["train"]["inputs"], "SampleRates": data["train"]["sampling_rates"], "Labels": data["train"]["labels"]},
        "valid": {"AUDIO": data["valid"]["inputs"], "SampleRates": data["valid"]["sampling_rates"], "Labels": data["valid"]["labels"]},
        "test": {"AUDIO": data["test"]["inputs"], "SampleRates": data["test"]["sampling_rates"], "Labels": data["test"]["labels"]}
    }

def preprocess_audio_data(audio_fm, train_inputs, train_y, valid_inputs, valid_y, test_inputs, test_y, 
                         base_data_path=None, skip_check=False):
    """
    Preprocess audio data and handle invalid audio files
    
    Parameters:
    - audio_fm: AudioRepr instance for feature extraction
    - train_inputs, valid_inputs, test_inputs: Audio inputs
    - train_y, valid_y, test_y: Corresponding labels
    - base_data_path: Base path for data storage
    - skip_check: Whether to skip audio validation checks
    
    Returns:
    - Processed inputs and labels for train, validation, and test sets
    """
    if not skip_check:
        # Filter out invalid audio files
        valid_train_indices = []
        for i, audio in enumerate(train_inputs):
            if isinstance(audio, np.ndarray) and audio.size > 0:
                valid_train_indices.append(i)
        
        valid_val_indices = []
        for i, audio in enumerate(valid_inputs):
            if isinstance(audio, np.ndarray) and audio.size > 0:
                valid_val_indices.append(i)
        
        valid_test_indices = []
        for i, audio in enumerate(test_inputs):
            if isinstance(audio, np.ndarray) and audio.size > 0:
                valid_test_indices.append(i)
        
        # Filter the data
        train_inputs = [train_inputs[i] for i in valid_train_indices]
        train_y = train_y[valid_train_indices] if isinstance(train_y, np.ndarray) else [train_y[i] for i in valid_train_indices]
        
        valid_inputs = [valid_inputs[i] for i in valid_val_indices]
        valid_y = valid_y[valid_val_indices] if isinstance(valid_y, np.ndarray) else [valid_y[i] for i in valid_val_indices]
        
        test_inputs = [test_inputs[i] for i in valid_test_indices]
        test_y = test_y[valid_test_indices] if isinstance(test_y, np.ndarray) else [test_y[i] for i in valid_test_indices]
    
    return train_inputs, train_y, valid_inputs, valid_y, test_inputs, test_y
