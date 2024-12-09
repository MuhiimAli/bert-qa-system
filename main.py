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
    parser.add_argument("--train_data_path", type=str, required=True,
                        help="Path to training data JSON file")
    parser.add_argument("--eval_data_path", type=str, required=True,
                        help="Path to evaluation data JSON file")
    # Model arguments
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=8,
                        help="Evaluation batch size")
    
    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Device arguments
    parser.add_argument("--distributed", action="store_true",
                        help="Whether to use distributed training")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save model checkpoints")
    
    return parser.parse_args()

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
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    try:
        model, qa_head = train(args, data, tokenizer)
        
        # Save final model
        final_output_dir = os.path.join(args.output_dir, "final_model")
        os.makedirs(final_output_dir, exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'qa_head_state_dict': qa_head.state_dict(),
            'args': args
        }, os.path.join(final_output_dir, "final_model.pt"))
        
        print(f"Training completed. Final model saved to {final_output_dir}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save interrupted model
        interrupted_model_path = os.path.join(args.output_dir, "interrupted_model.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'qa_head_state_dict': qa_head.state_dict(),
            'args': args
        }, interrupted_model_path)
        print(f"Interrupted model saved to {interrupted_model_path}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()