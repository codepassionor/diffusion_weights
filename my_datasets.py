import os
import torch
from torch.utils.data import Dataset

class ParaDataset(Dataset):
    def __init__(self, para_file):
        """Initialize the dataset.

        Args:
            para_file (str): Path to the .pt file containing the parameter dictionary.
        """
        # Load the parameter dictionary from the .pt file
        self.param_dict = torch.load(para_file)
        self.keys = list(self.param_dict.keys())

    def __len__(self):
        return len(self.param_dict)

    def __getitem__(self, idx):
        key = self.keys[idx]
        param_subdict = self.param_dict[key]
        
        sorted_values = [param_subdict[k] for k in sorted(param_subdict.keys())]
        
        tensor_list = [torch.tensor(value) for value in sorted_values]
        
        return tensor_list
    
class ParaDataset_v1(Dataset):
    def __init__(self, para_file):
        """Initialize the dataset.

        Args:
            para_file (str): Path to the .pt file containing the parameter dictionary.
        """
        # Load the parameter dictionary from the .pt file
        self.param_dict = torch.load(para_file)
        self.keys = list(self.param_dict.keys())

    def __len__(self):
        return len(self.param_dict)-4

    def __getitem__(self, idx):
        key = self.keys[idx]
        key_next = self.keys[idx+1]
        pixel_value = self.param_dict[key]
        
        noise = self.param_dict[key_next] - pixel_value
        
        return idx, pixel_value, noise
