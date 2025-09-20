import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class DS(Dataset):
    def __init__(self, 
                 dataset_path: str,
                 window_size: int,
                 window_steps: int,
                 eval: bool = False,
                 device: torch.device=torch.device("mps"),
                 dtype: torch.dtype=torch.float16) -> None:
        self.device = device
        self.entries = pd.read_csv(dataset_path)

        if eval:
            print("Evaluation mode activated, created training items")

            self.ids = torch.tensor(self.entries["id"]).to(self.device, dtype=dtype)
            self.train = torch.tensor(self.entries.drop(["id", "time"], axis = 1)\
                                  .values.astype(np.float32)).to(self.device, dtype=dtype)
            

            return
        
        self.train = torch.tensor(self.entries.drop(['Y1', "Y2", "time"], axis = 1)\
                                  .values.astype(np.float32)).to(self.device, dtype=dtype)
        
        self.train_targets = torch.stack((
            torch.tensor(self.entries['Y1']).to(self.device, dtype=dtype),
            torch.tensor(self.entries['Y2']).to(self.device, dtype=dtype),
        )).to(self.device).transpose(1,0)


        n, f = self.train_targets.shape
        ft, nt = self.train.shape

        self.m = int((n-window_size)/window_steps)
        train_stack, target_stack = [], []


        for i in range(self.m): 
            l, r = i*window_steps, (i*window_steps)+window_size
            train_stack.append(self.train[l:r, :])
            target_stack.append(self.train_targets[l:r, :])

        self.window_train = torch.stack(train_stack).to(device=self.device, dtype=dtype)
        self.windows_targets = torch.stack(target_stack).to(device=self.device, dtype=dtype)

        print("Initialized dataset", self.window_train.shape, self.windows_targets.shape)

    def __len__(self):
        return self.window_train.shape[0]
    
    def __getitem__(self, idx):
        return self.window_train[idx], self.windows_targets[idx] # Training tensor for dataloader
    
    
    
