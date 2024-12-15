import torch
import torch.nn as nn
import torch.nn.functional as F

class NQLoss(nn.Module):
    def __init__(self):
        """Initialize the loss function"""
        super(NQLoss, self).__init__()
        
    def forward(self, start_logits, end_logits, answer_type_logits,
                start_positions, end_positions, answer_types):
        """
        Compute negative log probability loss for each component per example
        
        Args:
            start_logits: (batch_size, seq_length)
            end_logits: (batch_size, seq_length) 
            answer_type_logits: (batch_size, num_answer_types)
            start_positions: (batch_size,)
            end_positions: (batch_size,)
            answer_types: (batch_size,)
            
        Returns:
            total_loss: sum of negative log probabilities for start, end, and answer type
        """
        # Calculate probabilities using softmax
        start_probs = F.softmax(start_logits, dim=-1)  # (batch_size, seq_length)
        end_probs = F.softmax(end_logits, dim=-1)      # (batch_size, seq_length)
        type_probs = F.softmax(answer_type_logits, dim=-1)  # (batch_size, num_answer_types)
        
        batch_size = start_logits.size(0)
        
        # Get the probabilities corresponding to the correct positions/types
        # gather: selects values from tensor using indices
        start_selected_probs = start_probs[torch.arange(batch_size), start_positions]  # (batch_size,)
        end_selected_probs = end_probs[torch.arange(batch_size), end_positions]        # (batch_size,)
        type_selected_probs = type_probs[torch.arange(batch_size), answer_types]       # (batch_size,)
        
        # Calculate negative log probabilities (-log p)
        start_loss = -torch.log(start_selected_probs + 1e-10)  # Add small epsilon to prevent log(0)
        end_loss = -torch.log(end_selected_probs + 1e-10)
        type_loss = -torch.log(type_selected_probs + 1e-10)
        
        # Sum losses for each example
        total_loss = start_loss + end_loss + type_loss  # (batch_size,)
        
        # Return mean across batch
        return total_loss.mean()

    