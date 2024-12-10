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

def evaluate_model(model: DistilBertModel, qa_head: QAHead, dataloader: DataLoader, 
                  criterion: NQLoss, tokenizer, device: str) -> Dict:
    """
    Evaluate the model using token-based metrics, properly handling no-answer cases.
    """
    model.eval()
    qa_head.eval()
    
    total_loss = 0
    total_true_pos = 0
    total_false_pos = 0
    total_false_neg = 0
    num_examples = 0
    exact_matches = 0
    total_questions = 0
    
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
            
            # Convert answer types to tensor
            answer_types = torch.tensor(
                [1 if x == 'short' else 0 for x in batch['answer_type']],
                dtype=torch.long,
                device=device
            )
            
            # Calculate loss
            loss = criterion(
                start_logits,
                end_logits,
                type_logits,
                batch['start_position'],
                batch['end_position'],
                answer_types
            )
            
            # Get predictions
            start_preds = torch.argmax(start_logits, dim=1)
            end_preds = torch.argmax(end_logits, dim=1)
            type_preds = torch.argmax(type_logits, dim=1)
            
            # Calculate metrics for each example
            for i in range(len(batch['input_ids'])):
                total_questions += 1
                pred_type = type_preds[i].item()
                true_type = answer_types[i].item()
                
                # Get spans
                pred_start = start_preds[i].item()
                pred_end = end_preds[i].item()
                true_start = batch['start_position'][i].item()
                true_end = batch['end_position'][i].item()
                
                # Fix malformed prediction spans
                if pred_end < pred_start:
                    pred_start, pred_end = pred_end, pred_start
                
                # Handle no-answer case (true_type = 0)
                if true_type == 0:
                    if pred_type == 0:
                        # Correctly predicted no-answer
                        total_true_pos += 1
                        exact_matches += 1
                    else:
                        # Predicted span when should be no-answer
                        pred_length = pred_end - pred_start + 1
                        total_false_pos += pred_length
                        total_false_neg += 1
                    continue
                
                # Handle answerable case (true_type = 1)
                # Get the actual tokens for predicted and true spans
                input_ids = batch['input_ids'][i]
                
                # Get tokens for both spans
                pred_tokens = set(input_ids[pred_start:pred_end + 1].tolist())
                true_tokens = set(input_ids[true_start:true_end + 1].tolist())
                
                # Remove special tokens and padding
                special_tokens = {tokenizer.pad_token_id, tokenizer.cls_token_id, 
                               tokenizer.sep_token_id, tokenizer.unk_token_id}
                pred_tokens = pred_tokens - special_tokens
                true_tokens = true_tokens - special_tokens
                
                # Calculate token-based metrics
                true_pos = len(pred_tokens.intersection(true_tokens))
                false_pos = len(pred_tokens - true_tokens)
                false_neg = len(true_tokens - pred_tokens)
                
                # Check for exact match (same tokens and type)
                if (pred_type == true_type and pred_tokens == true_tokens):
                    exact_matches += 1
                
                total_true_pos += true_pos
                total_false_pos += false_pos
                total_false_neg += false_neg
            
            total_loss += loss.item()
            num_examples += len(batch['input_ids'])
            
            # Update progress bar
            precision, recall, f1 = compute_metrics(
                total_true_pos, total_false_pos, total_false_neg)
            exact_match = exact_matches / total_questions if total_questions > 0 else 0
            
            progress_bar.set_description(
                f"Loss: {total_loss/num_examples:.4f}, F1: {f1:.4f}, EM: {exact_match:.4f}"
            )
    
    # Calculate final metrics
    precision, recall, f1 = compute_metrics(
        total_true_pos, total_false_pos, total_false_neg)
    exact_match = exact_matches / total_questions if total_questions > 0 else 0
    
    return {
        'loss': total_loss / num_examples,
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
        # train_loss = train_one_epoch(
        #     model=model,
        #     qa_head=qa_head,
        #     dataloader=data['train'].dataloader,
        #     criterion=criterion,
        #     optimizer=optimizer,
        #     scheduler=scheduler,
        #     device=device
        # )
        
        # Evaluate
        eval_metrics = evaluate_model(
            model=model,
            qa_head=qa_head,
            dataloader=data['eval'].dataloader,
            criterion=criterion,
            tokenizer=tokenizer,
            device=device
        )
        
        # Print metrics
        print(f"\nResults:")
        # print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {eval_metrics['loss']:.4f}")
        print(f"Precision: {eval_metrics['precision']:.4f}")
        print(f"Recall: {eval_metrics['recall']:.4f}")
        print(f"F1: {eval_metrics['f1']:.4f}")
        
    
    return model, qa_head