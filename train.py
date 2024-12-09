import torch
from torch.utils.data import DataLoader
from transformers import (
    DistilBertModel,
    DistilBertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from loss import NQLoss

class QAHead(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.qa_outputs = torch.nn.Linear(hidden_size, 2)  # 2 for start/end
        self.qa_type = torch.nn.Linear(hidden_size, 2)     # 2 for no-answer/short

    def forward(self, hidden_states):
        # Get logits for start/end
        span_logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = span_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        # Get answer type logits from [CLS] token
        type_logits = self.qa_type(hidden_states[:, 0, :])
        
        return start_logits, end_logits, type_logits

def compute_metrics(total_true_pos: int, total_false_pos: int, total_false_neg: int) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1 from totals."""
    precision = total_true_pos / (total_true_pos + total_false_pos) if (total_true_pos + total_false_pos) > 0 else 0
    recall = total_true_pos / (total_true_pos + total_false_neg) if (total_true_pos + total_false_neg) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def evaluate_model(model: DistilBertModel, qa_head: QAHead, dataloader: DataLoader, 
                  criterion: NQLoss, device: str) -> Dict:
    """Evaluate the model on validation data."""
    model.eval()
    qa_head.eval()
    
    total_loss = 0
    total_true_pos = 0
    total_false_pos = 0
    total_false_neg = 0
    num_examples = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            start_logits, end_logits, type_logits = qa_head(outputs.last_hidden_state)
            

            answer_types = torch.tensor(
                [1 if x == 'short' else 0 for x in batch['answer_type']],
                dtype=torch.long,  # Make sure dtype matches loss expectation
                device=device
             )
            max_len = start_logits.size(1)
            start_positions = batch['start_position'].clamp(0, max_len - 1)
            end_positions = batch['end_position'].clamp(0, max_len - 1)
            
            # Calculate loss
            loss = criterion(
                start_logits,
                end_logits,
                type_logits,
                start_positions,
                end_positions,
                answer_types
            )
            
            # Get predictions
            start_preds = torch.argmax(start_logits, dim=1)
            end_preds = torch.argmax(end_logits, dim=1)
            type_preds = torch.argmax(type_logits, dim=1)
            
            # Calculate metrics for each example
            for i in range(len(batch['input_ids'])):
                # Skip if both predicted and true are no-answer
                if type_preds[i] == 0 and answer_types[i] == 0:
                    continue
                
                # Get predicted and true spans
                pred_span = set(range(start_preds[i].item(), end_preds[i].item() + 1))
                true_span = set(range(
                    batch['start_position'][i].item(),
                    batch['end_position'][i].item() + 1
                ))
                
                # Calculate token-level metrics
                true_pos = len(pred_span.intersection(true_span))
                false_pos = len(pred_span - true_span)
                false_neg = len(true_span - pred_span)
                
                total_true_pos += true_pos
                total_false_pos += false_pos
                total_false_neg += false_neg
            
            total_loss += loss.item()
            num_examples += len(batch['input_ids'])
            
            # Update progress bar
            precision, recall, f1 = compute_metrics(total_true_pos, total_false_pos, total_false_neg)
            progress_bar.set_description(
                f"Loss: {total_loss/num_examples:.4f}, F1: {f1:.4f}"
            )
    
    # Calculate final metrics
    precision, recall, f1 = compute_metrics(total_true_pos, total_false_pos, total_false_neg)
    
    return {
        'loss': total_loss / num_examples,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': total_true_pos,
        'false_positives': total_false_pos,
        'false_negatives': total_false_neg
    }

def train_one_epoch(model: DistilBertModel, qa_head: QAHead, dataloader: DataLoader,
                    criterion: NQLoss, optimizer: AdamW, scheduler, device: str) -> float:
    model.train()
    qa_head.train()
    
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
        
        start_logits, end_logits, type_logits = qa_head(outputs.last_hidden_state)
        
        # Convert string answer types to tensor and ensure it's on the right device
        answer_types = torch.tensor(
            [1 if x == 'short' else 0 for x in batch['answer_type']],
            dtype=torch.long,  # Make sure dtype matches loss expectation
            device=device
        )
        # Ensure positions are within valid range
        max_len = start_logits.size(1)
        start_positions = batch['start_position'].clamp(0, max_len - 1)
        end_positions = batch['end_position'].clamp(0, max_len - 1)
        
        # Calculate loss
        loss = criterion(
            start_logits,  # (batch_size, seq_len)
            end_logits,    # (batch_size, seq_len)
            type_logits,   # (batch_size, 2)
            start_positions,  # (batch_size,)
            end_positions,    # (batch_size,)
            answer_types      # (batch_size,)
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(qa_head.parameters(), 1.0)
        
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

def train(args, data, tokenizer):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model components
    model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
    qa_head = QAHead(model.config.hidden_size).to(device)
    criterion = NQLoss()
    
    # Setup optimizer
    optimizer = AdamW(
        list(model.parameters()) + list(qa_head.parameters()),
        lr=args.learning_rate
    )
    
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
            qa_head=qa_head,
            dataloader=data['train'].dataloader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )
        
        # Evaluate
        eval_metrics = evaluate_model(
            model=model,
            qa_head=qa_head,
            dataloader=data['eval'].dataloader,
            criterion=criterion,
            device=device
        )
        
        # Print metrics
        print(f"\nResults:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {eval_metrics['loss']:.4f}")
        print(f"Precision: {eval_metrics['precision']:.4f}")
        print(f"Recall: {eval_metrics['recall']:.4f}")
        print(f"F1: {eval_metrics['f1']:.4f}")
        
        # # Save best model
        # if eval_metrics['f1'] > best_f1:
        #     best_f1 = eval_metrics['f1']
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'qa_head_state_dict': qa_head.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'scheduler_state_dict': scheduler.state_dict(),
        #         'best_f1': best_f1,
        #         'args': args
        #     }, 'best_model.pt')
    
    return model, qa_head