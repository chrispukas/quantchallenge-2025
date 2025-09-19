import torch
import random
import numpy as np
import pandas as pd

class Dataset():
    def __init__(self, 
                 dataset_path: str,
                 device: torch.device=torch.device("cpu")) -> None:
        self.device = device
        self.dataset = self.load_dataset(dataset_path)


    def load_dataset(self, dataset_path: str) -> list:
        print("Loading Dataset")
        df = pd.read_csv(dataset_path)
        arr = list(df.itertuples(index=False, name=None))
        return arr
        
        
    def get_batches(self, batch_size: int):
        batches = self.dataset
        random.shuffle(batches)
        batches = np.array_split(batches, len(batches) // batch_size)
        return batches

    def __len__(self):
        return len(self.dataset)
