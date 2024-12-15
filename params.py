import argparse
import os
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
    # WandB settings
    parser.add_argument('--use_wandb', action='store_true', default= False, help='Enable WandB logging')
    parser.add_argument('--sweep', action='store_true', default = False, help='Run hyperparameter sweep')


    # Model arguments
    model_args = parser.add_argument_group('Model configuration')
    model_args.add_argument(
        "--max_seq_length", 
        type=int, 
        default=512,
        help="Maximum sequence length for input text"
    )
    
    # Training arguments
    train_args = parser.add_argument_group('Training configuration')
    train_args.add_argument(
        "--batch_size", 
        type=int, 
        default=8,
        help="Training batch size"
    )
    train_args.add_argument(
        "--learning_rate", 
        type=float, 
        default=4e-5,
        help="Learning rate for optimization"
    )
    train_args.add_argument(
        "--num_epochs", 
        type=int, 
        default=2,
        help="Number of training epochs"
    )
    train_args.add_argument(
        "--weight_decay", 
        type=float, 
        default=0,
        help="Weight decay for AdamW optimizer"
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