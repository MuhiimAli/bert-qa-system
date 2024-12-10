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
from train import train

def parse_args():
    parser = argparse.ArgumentParser(description="Train QA model on Natural Questions dataset")
    
    # Data arguments
    data_args = parser.add_argument_group('Data configuration')
    data_args.add_argument(
        "--train_data_path", 
        type=str, 
        default="/users/mali37/scratch/bert_qa_data/all_train.json",
        help="Path to training data JSON file"
    )
    data_args.add_argument(
        "--eval_data_path", 
        type=str, 
        default="/users/mali37/scratch/bert_qa_data/all_dev.json",
        help="Path to evaluation data JSON file"
    )
    
    # Model arguments
    model_args = parser.add_argument_group('Model configuration')
    model_args.add_argument(
        "--max_seq_length", 
        type=int, 
        default=512,
        help="Maximum sequence length for input text"
    )
    model_args.add_argument(
        "--model_name", 
        type=str, 
        default="distilbert-base-uncased",
        help="Name or path of the pretrained model to use"
    )
    
    # Training arguments
    train_args = parser.add_argument_group('Training configuration')
    train_args.add_argument(
        "--batch_size", 
        type=int, 
        default=2,
        help="Training batch size"
    )
    train_args.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=8,
        help="Evaluation batch size"
    )
    train_args.add_argument(
        "--learning_rate", 
        type=float, 
        default=3e-5,
        help="Learning rate for optimization"
    )
    train_args.add_argument(
        "--num_epochs", 
        type=int, 
        default=3,
        help="Number of training epochs"
    )
    train_args.add_argument(
        "--warmup_steps", 
        type=int, 
        default=0,
        help="Number of warmup steps for learning rate scheduler"
    )
    train_args.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.01,
        help="Weight decay for AdamW optimizer"
    )
    train_args.add_argument(
        "--gradient_accumulation_steps", 
        type=int, 
        default=1,
        help="Number of updates steps to accumulate before backward pass"
    )
    
    # System arguments
    sys_args = parser.add_argument_group('System configuration')
    sys_args.add_argument(
        "--num_workers", 
        type=int, 
        default=4,
        help="Number of data loading workers"
    )
    sys_args.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    sys_args.add_argument(
        "--distributed", 
        action="store_true",
        help="Whether to use distributed training"
    )
    sys_args.add_argument(
        "--local_rank", 
        type=int, 
        default=-1,
        help="Local rank for distributed training"
    )
    
    # Output arguments
    output_args = parser.add_argument_group('Output configuration')
    output_args.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs",
        help="Directory to save model checkpoints and logs"
    )
    output_args.add_argument(
        "--logging_steps", 
        type=int, 
        default=100,
        help="Log training metrics every X updates steps"
    )
    output_args.add_argument(
        "--save_steps", 
        type=int, 
        default=1000,
        help="Save checkpoint every X updates steps"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    return args

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # Parse arguments
    args = parse_args()
    
    
    # Set random seed
    set_seed(args.seed)
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Load data using the imported functions
    print("Loading datasets...")
    data = get_data(args, tokenizer)
    
    if 'train' in data:
        num_train_examples = len(data['train'].dataloader.dataset)
        print(f"Training examples: {num_train_examples}")
        print(f"Number of training batches: {len(data['train'].dataloader)}")
    
    if 'eval' in data:
        num_eval_examples = len(data['eval'].dataloader.dataset)
        print(f"Evaluation examples: {num_eval_examples}")
        print(f"Number of evaluation batches: {len(data['eval'].dataloader)}")
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Train model
    print("Starting training...")
    model, qa_head = train(args, data, tokenizer)


if __name__ == "__main__":
    main()