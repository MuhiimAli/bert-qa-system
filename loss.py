#copy this from the paper
import torch
import torch.nn as nn
import torch.nn.functional as F


class NQLoss(nn.Module):  # Inherit from nn.Module
    def __init__(self):
        """Initialize the loss function with CrossEntropyLoss"""
        super(NQLoss, self).__init__()  # Call parent class constructor
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, start_logits, end_logits, answer_type_logits,
                start_positions, end_positions, answer_types):
        """
        Args:
            start_logits: (batch_size, seq_length)
            end_logits: (batch_size, seq_length)
            answer_type_logits: (batch_size, num_answer_types)
            start_positions: (batch_size,)
            end_positions: (batch_size,)
            answer_types: (batch_size,)
        Returns:
            total_loss: sum of start, end, and answer type losses
        """
        # Compute individual losses using CrossEntropyLoss
        start_loss = self.criterion(start_logits, start_positions)
        end_loss = self.criterion(end_logits, end_positions)
        answer_type_loss = self.criterion(answer_type_logits, answer_types)
        
        # Sum the three losses
        total_loss = start_loss + end_loss + answer_type_loss
        
        return total_loss

    