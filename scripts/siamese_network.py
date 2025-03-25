import torch
import torch.nn as nn
from stock_encoder import StockEncoder

class SiameseNetwork(nn.Module):
    """
    this network takes pairs of stock return sequences, encodes them into a latent space,
    and computes the Euclidean distance between the embeddings to gauge similarity
    """
    def __init__(self, input_size: int, hidden_size: int, batch_first: bool):
        """        
        :param input_size: number of features in the input sequences
        :param hidden_size: number of features in the hidden state of the encoder
        :param batch_first: whether the input tensors have batch as the first dimension
        """
        super(SiameseNetwork, self).__init__()
        self.encoder = StockEncoder(input_size, hidden_size, batch_first)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        :param x1: first stock return sequence
        :param x2: second stock return sequence
        :returns: euclidean distances between the embeddings of the two stock sequences
        """
        embed1 = self.encoder(x1)
        embed2 = self.encoder(x2)
        
        embed1 = torch.nn.functional.normalize(embed1, p=2, dim=1)
        embed2 = torch.nn.functional.normalize(embed2, p=2, dim=1)
        
        return torch.norm(embed1 - embed2, p=2, dim=1)