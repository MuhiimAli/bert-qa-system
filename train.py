"""
This class is responsible for training and evaluating the model
"""
import torch
from torch.utils.data import DataLoader
from transformers import (
    DistilBertModel,
    DistilBertTokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW 
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from loss import NQLoss
from collections import Counter
import torch.nn as nn
import wandb

class DistilBertForQA(DistilBertModel):
    def __init__(self, config):
        super().__init__(config)
        
        # Add dropout
        # self.dropout = nn.Dropout(0.1)
        
        #a single linear layer for each output type
        self.qa_start = torch.nn.Linear(config.hidden_size, 1)  # Single output for start position
        self.qa_end = torch.nn.Linear(config.hidden_size, 1)    # Single output for end position
        self.qa_type = torch.nn.Linear(config.hidden_size, 2)   # Binary classification for answer type
        
        # Initialize weights
        self.init_weights()
        
    def forward(self, input_ids, attention_mask=None):
        # Get base model outputs
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        hidden_states = outputs.last_hidden_state
        
        # Apply dropout to hidden states
        # hidden_states = self.dropout(hidden_states)
        
        # Get start and end logits directly from single linear layers
        start_logits = self.qa_start(hidden_states).squeeze(-1)  # [batch_size, seq_len]
        end_logits = self.qa_end(hidden_states).squeeze(-1)     # [batch_size, seq_len]
        
        # Get answer type logits from [CLS] token
        type_logits = self.qa_type(hidden_states[:, 0, :])      # [batch_size, 2]
        
        return start_logits, end_logits, type_logits



def compute_metrics(total_true_pos: int, total_false_pos: int, 
                   total_false_neg: int) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 from token-level metrics.
    
    Args:
        total_true_pos: Number of tokens correctly predicted (in both spans)
        total_false_pos: Number of tokens incorrectly predicted (in pred but not true)
        total_false_neg: Number of tokens missed (in true but not pred)
    
    Returns:
        Tuple of (precision, recall, f1)
    """
    precision = total_true_pos / (total_true_pos + total_false_pos) if (total_true_pos + total_false_pos) > 0 else 0
    recall = total_true_pos / (total_true_pos + total_false_neg) if (total_true_pos + total_false_neg) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def evaluate_model(model: DistilBertModel, dataloader: DataLoader, 
                  criterion: NQLoss, tokenizer, device: str) -> Dict:
    """
    Evaluate the model using token-based metrics, properly handling no-answer cases
    and counting token occurrences correctly.
    """
    model.eval()
    
    total_loss = 0
    total_true_pos = 0
    total_false_pos = 0
    total_false_neg = 0
    num_batches = 0
    exact_matches = 0
    total_questions = 0
    mismatch = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        for batch in progress_bar:
            num_batches += 1
            
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            start_logits, end_logits, type_logits = outputs
            
            # Calculate loss
            loss = criterion(
                start_logits,
                end_logits,
                type_logits,
                batch['start_position'],
                batch['end_position'],
                batch['answer_type']
            )
            
            # Get predictions
            start_preds = torch.argmax(start_logits, dim=1)
            end_preds = torch.argmax(end_logits, dim=1)
            type_preds = torch.argmax(type_logits, dim=1)
            
            # Calculate metrics for each example
            for i in range(len(batch['input_ids'])):
                total_questions += 1
                
                # Get predicted and true spans
                pred_start = start_preds[i].item()
                pred_end = end_preds[i].item()
                # pred_end = max(pred_start, pred_end)
                if pred_end < pred_start:
                    continue
                if batch['answer_type'][i] == 1 and (pred_start == 0 and pred_end == 0):
                    # Model incorrectly predicted no-answer for an answerable question
                    mismatch +=1
                    print(f"True answer: {true_text}")
                    print(f"Type logits: {torch.softmax(type_logits[i], dim=0)}")
                    continue
           

                true_start = batch['start_position'][i].item()
                true_end = batch['end_position'][i].item()
                
       
                
                # Get token IDs for both spans
                input_ids = batch['input_ids'][i]

                # Get token sequences
                pred_tokens = input_ids[pred_start:pred_end].tolist()
                true_tokens = input_ids[true_start:true_end].tolist()
                
                # Create frequency distributions
                pred_counter = Counter(pred_tokens)
                true_counter = Counter(true_tokens)

                #for debugging purposes only
                pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                true_text = tokenizer.decode(true_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                
                
                # Calculate overlap considering token frequencies
                true_pos = sum((pred_counter & true_counter).values())  # Intersection with frequencies
                false_pos = sum((pred_counter - true_counter).values())  # Tokens over-predicted
                false_neg = sum((true_counter - pred_counter).values())  # Tokens under-predicted
                
                # Check for exact match (both token IDs and their frequencies must match)
                if pred_tokens == true_tokens:
                    exact_matches += 1
                
                total_true_pos += true_pos
                total_false_pos += false_pos
                total_false_neg += false_neg
            
            total_loss += loss.item()
            
            # Update progress
            precision, recall, f1 = compute_metrics(
                total_true_pos, total_false_pos, total_false_neg)
            exact_match = exact_matches / total_questions if total_questions > 0 else 0
            
            progress_bar.set_description(
                f"Loss: {total_loss/num_batches:.4f}, F1: {f1:.4f}, EM: {exact_match:.4f}, MM: {mismatch:.4f}"
            )
    
    precision, recall, f1 = compute_metrics(
        total_true_pos, total_false_pos, total_false_neg)
    exact_match = exact_matches / total_questions if total_questions > 0 else 0
    
    return {
        'loss': total_loss / num_batches,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'exact_match': exact_match,
        'true_positives': total_true_pos,
        'false_positives': total_false_pos,
        'false_negatives': total_false_neg,
        'total_questions': total_questions,
        'exact_matches': exact_matches
    }

def train_one_epoch(model: DistilBertModel,dataloader: DataLoader,
                    criterion: NQLoss, optimizer: AdamW, scheduler, device: str) -> float:
    model.train()
    
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        
        start_logits, end_logits, type_logits = outputs
        
        # Convert string answer types to tensor and ensure it's on the right device
        answer_types = batch['answer_type']
        
        # Calculate loss
        loss = criterion(
            start_logits,  # (batch_size, seq_len)
            end_logits,    # (batch_size, seq_len)
            type_logits,   # (batch_size, 2)
            batch['start_position'],  # (batch_size,)
            batch['end_position'],    # (batch_size,)
            answer_types      # (batch_size,)
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update weights
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_description(
            f"Loss: {total_loss/num_batches:.4f}"
        )
    
    return total_loss / num_batches

def train(args, data, tokenizer, use_wandb=False):
    """Main training loop with optional WandB logging."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model components
    model = DistilBertForQA.from_pretrained('distilbert-base-uncased').to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = NQLoss()
    
    # Setup scheduler
    total_steps = len(data['train'].dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_f1 = 0
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # Train
        train_loss = train_one_epoch(
            model=model,
            dataloader=data['train'].dataloader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )
        
        # Evaluate
        eval_metrics = evaluate_model(
            model=model,
            dataloader=data['eval'].dataloader,
            criterion=criterion,
            tokenizer=tokenizer,
            device=device
        )
        
        # Print metrics
        print(f"\nResults:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {eval_metrics['loss']:.4f}")
        print(f"Precision: {eval_metrics['precision']:.4f}")
        print(f"Recall: {eval_metrics['recall']:.4f}")
        print(f"F1: {eval_metrics['f1']:.4f}")
        
        
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': eval_metrics['loss'],
                'val_precision': eval_metrics['precision'],
                'val_recall': eval_metrics['recall'],
                'val_f1': eval_metrics['f1'],
                'val_exact_match': eval_metrics['exact_match']
            })
    
    return model