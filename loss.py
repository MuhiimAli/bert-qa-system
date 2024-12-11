#copy this from the paper
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class NQLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def compute_cross_entropy_loss(self, logits, targets, num_classes):
        """
        Generic cross entropy loss computation
        Args:
            logits: (batch_size, num_classes)
            targets: (batch_size,)
            num_classes: int, number of possible classes
        """
        # Convert targets to one-hot vectors
        one_hot = F.one_hot(targets, num_classes=num_classes).float()
        
        # Compute log softmax
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Compute loss: -mean(sum(one_hot * log_probs))
        loss = -torch.mean(torch.sum(one_hot * log_probs, dim=-1))
        return loss
        
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
        """
        seq_length = start_logits.size(1)
        num_answer_types = answer_type_logits.size(1)

    
        start_loss = self.compute_cross_entropy_loss(start_logits, start_positions, seq_length)
        end_loss =   self.compute_cross_entropy_loss(end_logits, end_positions, seq_length)
        answer_type_loss = self.compute_cross_entropy_loss(answer_type_logits, answer_types, num_answer_types)
        
        
        # Average the three losses
        total_loss = (start_loss + end_loss + answer_type_loss) / 3.0
        
        return total_loss

    