import wandb
import argparse
from transformers import DistilBertTokenizer
import torch
import random
import numpy as np
import os
from data import (
    OrderedNQDataset,
    get_qa_dataset,
    get_data,
    DataInfo
)
from sweep_config import sweep_configuration
from train import train
from params import parse_args

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_with_params(args, use_wandb=False):
    """Run a single training with given parameters."""
    if use_wandb:
        
        args.__dict__.update(wandb.config)
        
    
    set_seed(args.seed)
    
    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Load data
    data = get_data(args, tokenizer)
    if 'train' in data:
        num_train_examples = len(data['train'].dataloader.dataset)
        print(f"Training examples: {num_train_examples}")
        print(f"Number of training batches: {len(data['train'].dataloader)}")
    
    if 'eval' in data:
        num_eval_examples = len(data['eval'].dataloader.dataset)
        print(f"Evaluation examples: {num_eval_examples}")
        print(f"Number of evaluation batches: {len(data['eval'].dataloader)}")
    
    # Train model
    model = train(args, data, tokenizer, use_wandb)
    
    return model

def main():
    
    args = parse_args()
    
    if args.use_wandb:
        # Initialize wandb
        wandb.login()
        
        if args.sweep:
            # Initialize sweep
            sweep_id = wandb.sweep(
                sweep_configuration,
                project="Please pleaaaaaase let's it be the final one"
            )
            
            def run_sweep():
                with wandb.init() as run:
                    
                    wandb.config.update({
                        "torch_version": torch.__version__,
                        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                    })
                    return train_with_params(args, use_wandb=True)
            
            # Start sweep
            wandb.agent(sweep_id, function=run_sweep)
        else:
            # Single run with wandb logging
            with wandb.init(project="nq-qa-bert", config=args.__dict__) as run:
                model = train_with_params(args, use_wandb=True)
    else:
        
        set_seed(args.seed)
        model = train_with_params(args, use_wandb=False)

if __name__ == "__main__":
    main()