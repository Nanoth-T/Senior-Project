import torch
import torch.nn as nn

class CrossRangePenaltyLoss(nn.Module):
    def __init__(self, range_indices, base_loss=None, penalty_factor=2.0):
        """
        Args:
        range_indices (dict): Dictionary that maps categories to their range of indices, e.g.:
                              {'note_on': [0, 10], 'note_off': [11, 20], 'time_shift': [21, 30], 'velocity': [31, 40]}
        base_loss (nn.Module): The base loss function (default: CrossEntropyLoss)
        penalty_factor (float): Multiplier for penalizing cross-range errors.
        """
        super(CrossRangePenaltyLoss, self).__init__()
        self.range_indices = range_indices
        self.base_loss = base_loss if base_loss else nn.CrossEntropyLoss()
        self.penalty_factor = penalty_factor

    def get_range(self, index):
        """
        Determine which range the index belongs to based on the range_indices.
        """
        for category, range_idx in self.range_indices.items():
            if index in range_idx:
                return category
        return None

    def forward(self, predictions, targets):
        """
        Compute the loss, adding a penalty for predictions in the wrong range.
        Args:
        predictions (Tensor): Model predictions (logits) of shape [batch_size, num_classes]
        targets (Tensor): True target labels of shape [batch_size]
        """
        # Base loss (CrossEntropyLoss)
        base_loss = self.base_loss(predictions, targets)
        
        # Get predicted classes
        predicted_classes = torch.argmax(predictions, dim=1)
        
        # Initialize total penalty
        penalty = 0.0
        
        # Calculate penalty for cross-range predictions
        for pred_class, target_class in zip(predicted_classes, targets):
            pred_range = self.get_range(pred_class.item())
            target_range = self.get_range(target_class.item())
            
            if pred_range != target_range:
                # Apply penalty if the prediction is in a different range
                penalty += self.penalty_factor
        
        # Combine base loss with penalty
        total_loss = base_loss + (penalty / len(targets))  # Normalize penalty by batch size
        
        return total_loss
