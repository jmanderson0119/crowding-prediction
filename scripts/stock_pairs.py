import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import json

class StockPairs(Dataset):
    def __init__(self, csv_path: str): self.dataframe = pd.read_csv(csv_path)
        
    def __len__(self) -> int: return len(self.dataframe)

    def __getitem__(self, i: int) -> tuple:
        """        
        :param i: index of the sample
        :returns: labeled pair (stock1, stock2, label)
        """
        row = self.dataframe.iloc[i]
        
        # conv string representations of lists to actual lists
        stock1 = json.loads(row['stock1'].replace("'", "\""))
        stock2 = json.loads(row['stock2'].replace("'", "\""))
        label = row['correlated']
        
        # conv to numpy arrays
        stock1 = np.array(stock1, dtype=np.float32)
        stock2 = np.array(stock2, dtype=np.float32)
        label = np.float32(label)
        
        # conv to pytorch tensors
        stock1_tensor = torch.tensor(stock1)
        stock2_tensor = torch.tensor(stock2)
        label_tensor = torch.tensor(label)
        
        return stock1_tensor, stock2_tensor, label_tensor