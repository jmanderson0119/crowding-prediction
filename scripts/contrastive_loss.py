import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float):
        """        
        :param margin: margin for dissimilar pairs
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distances: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """        
        :param distances: euclidean distances between stock embeddings in the batchs
        :param label: similarity labels, where 0 = similar, 1 = dissimilar
        :returns: contrastive loss value
        """
        contrastive_loss = (1 - label) * torch.pow(distances, 2) + label * torch.pow(torch.clamp(self.margin - distances, min=0), 2)
        return contrastive_loss.mean()