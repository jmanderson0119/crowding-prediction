import torch
import torch.nn as nn

class StockEncoder(nn.Module):
    """
    underlying encoding layer for the siamese network architecture.
    This encoder takes time series return data and returns the equivalent embedding
    """
    def __init__(self, input_size: int, hidden_size: int, batch_first: bool):
        """
        :param input_size: number of features in the input sequence
        :param hidden_size: number of features in the hidden state
        :param batch_first: whether the input tensor has batch as the first dimension
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first)
        self.fc = nn.Linear(hidden_size, hidden_size // 2)
        self.bn = nn.BatchNorm1d(hidden_size // 2)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """        
        :param x: input tensor
        :returns: final hidden state of the last lstm layer
        """
        x = x.squeeze(0).unsqueeze(-1)
        _, (h_n, _) = self.lstm(x)
        h_n = self.fc(h_n[-1])
        h_n = self.bn(h_n)
        h_n = self.tanh(h_n)
        return h_n