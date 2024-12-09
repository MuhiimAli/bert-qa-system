#copy this from the paper
import torch
import torch.nn as nn
import torch.nn.functional as F

class NQLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, start_logits, end_logits, answer_type_logits, 
            start_positions, end_positions, answer_types):
    # Add weights to balance the loss components
        start_loss = F.cross_entropy(start_logits, start_positions, reduction='mean')
        end_loss = F.cross_entropy(end_logits, end_positions, reduction='mean')
        type_loss = F.cross_entropy(answer_type_logits, answer_types, reduction='mean')
        
        # Maybe the components are not balanced
        total_loss = start_loss + end_loss + type_loss
        
        # Add debug prints
        # print(f"Start loss: {start_loss.item():.4f}")
        # print(f"End loss: {end_loss.item():.4f}")
        # print(f"Type loss: {type_loss.item():.4f}")
        
        return total_loss

    def get_answer_scores(self, start_logits, end_logits, input_mask=None):
        """
        Calculate answer span scores using the formula from the paper:
        g(c,s,e) = f_start(s,c,θ) + f_end(e,c,θ) - f_start(s=[CLS],c,θ) - f_end(e=[CLS],c,θ)
        
        Args:
            start_logits: (batch_size, seq_len)
            end_logits: (batch_size, seq_len)
            input_mask: (batch_size, seq_len) optional mask for padding
            
        Returns:
            span_scores: (batch_size, seq_len, seq_len) scores for each possible span
        """
        # Apply softmax to get probabilities
        start_probs = F.softmax(start_logits, dim=-1)  # (batch_size, seq_len)
        end_probs = F.softmax(end_logits, dim=-1)      # (batch_size, seq_len)
        
        # Get [CLS] token probabilities (index 0)
        cls_start_prob = start_probs[:, 0].unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1)
        cls_end_prob = end_probs[:, 0].unsqueeze(1).unsqueeze(2)      # (batch_size, 1, 1)
        
        # Compute outer sum of all possible spans
        # Expand dims for broadcasting
        start_probs = start_probs.unsqueeze(2)  # (batch_size, seq_len, 1)
        end_probs = end_probs.unsqueeze(1)      # (batch_size, 1, seq_len)
        
        # Calculate score for all possible spans
        span_scores = start_probs + end_probs - cls_start_prob - cls_end_prob
        
        # Mask invalid spans if input_mask provided
        if input_mask is not None:
            mask = input_mask.unsqueeze(2) & input_mask.unsqueeze(1)
            span_scores = span_scores.masked_fill(~mask, float('-inf'))
            
        return span_scores