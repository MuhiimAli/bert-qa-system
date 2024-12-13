#!/bin/bash

#SBATCH -n 10
#SBATCH --mem=180g
#SBATCH -t 48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -o my-output-%j.out
#SBATCH --job-name=nq_qa_train



# Create output directory for logs
mkdir -p logs

echo "Beginning finetuning"

# Run the training script
python main.py \
    --sweep \
    --use_wandb \
    2>&1 | tee logs/training_log_$(date +%Y%m%d_%H%M%S).log

# Print completion information
echo "Job finished at: $(date)"

# Optional: Cleanup
deactivate