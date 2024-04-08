import os
import pickle

def save_object(obj, filename):
    # Overwrites any existing file.
    with open(filename, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as inp:
        obj = pickle.load(inp)
        return obj
    
    
import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str,
               verbose: bool = True):
    
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)
    
    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt")
    model_save_path = target_dir_path / model_name
    
    # Save the model state_dict()
    if verbose:
        print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)
    

def load_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str,
               verbose: bool = True):
    
    # Create target directory
    target_dir_path = Path(target_dir)
    
    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt")
    model_load_path = target_dir_path / model_name
    
    # Load the model state_dict()
    if verbose:
        print(f"[INFO] Loading model from: {model_load_path}")
    checkpoint = torch.load(model_load_path)
    model.load_state_dict(checkpoint)

    return model