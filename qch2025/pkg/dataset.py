import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class DS(Dataset):
    def __init__(self, 
                 dataset_path: str,
                 dataset_path_2: str,
                 window_size: int,
                 window_steps: int,
                 eval: bool = False,
                 device: torch.device=torch.device("cuda"),
                 dtype: torch.dtype=torch.float16) -> None:
        
        self.device = device
        self.df = pd.read_csv(dataset_path)
        df2 = pd.read_csv(dataset_path_2)
        self.df = pd.concat([self.df.drop(["time"], axis=1), df2.fillna(0)], axis=1)
        

        # Feature generation:

        # Lagged



        self.eval = eval

        cols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', "N", "O", "P"]

        if eval:
            
            self.mean = self.df[cols].mean(skipna=True)
            self.std = self.df[cols].std()

            self.std[self.std==0] = 1 # Clear constant std variables

            self.entries = (self.df - self.mean) / self.std # Normalizes the dataset
            print("Evaluation mode activated, created training items")

            self.ids = torch.tensor(self.entries["id"]).to(self.device, dtype=dtype)
            self.train = torch.tensor(self.entries.drop(["id"], axis = 1)\
                                  .values.astype(np.float32)).to(self.device, dtype=dtype)

            n, f = self.train.shape
            self.m = int((n-window_size)/window_steps)

            train_stack = []

            for l in range(0, n, window_steps): 
                r = l + window_size
                tr_slice = self.train[l:r, :]
                if r > n:
                    print(f"Out of range by: {r-n} items!" )
                    pad_len = r-n
                    pad_tr = tr_slice[-1:].repeat(pad_len, 1)
                    tr_slice = torch.cat([tr_slice, pad_tr], dim=0)
                
                train_stack.append(tr_slice)

            self.window_train = torch.stack(train_stack).to(device=self.device, dtype=dtype)
                

            return

        self.mean = self.df[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', "N", "O", "P", "Y1", "Y2"]].mean(skipna=True)
        self.std = self.df[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', "N", "O", "P", 'Y1', "Y2"]].std()

        self.std[self.std==0] = 1 # Clear constant std variables

        self.entries = (self.df - self.mean) / self.std # Normalizes the dataset
        
        self.entries["Y1"] = self.df["Y1"]
        self.entries["Y2"] = self.df["Y2"]

        train = torch.tensor(self.entries.drop(['Y1', "Y2"], axis = 1)\
                                  .values.astype(np.float32)).to(self.device, dtype=dtype)
        tar = torch.stack((
            torch.tensor(self.entries['Y1']).to(self.device, dtype=dtype),
            torch.tensor(self.entries['Y2']).to(self.device, dtype=dtype),
        )).to(self.device).transpose(1,0)

        print(train.shape) # [points, features]
        print(tar.shape) # [points, features]

        it = int(train.shape[0]*1)

        self.train = train[: ,...]
        self.train_targets = tar[: ,...]

        self.eval_train = train[it: ,...]
        self.eval_targets = tar[it: ,...]

        n, f = self.train_targets.shape
        train_stack, target_stack = [], []

        for l in range(0, n, window_steps): 
            r = l + window_size
            tr_slice = self.train[l:r, :]
            y_slice = self.train_targets[l:r, :]
            
            if r > n:
                print(f"Out of range by: {r-n} items!" )
                pad_len = r-n
                pad_tr = tr_slice[-1:].repeat(pad_len, 1)
                pad_y = y_slice[-1:].repeat(pad_len, 1)

                tr_slice = torch.cat([tr_slice, pad_tr], dim=0)
                y_slice = torch.cat([y_slice, pad_y], dim=0)
            
            train_stack.append(tr_slice)
            target_stack.append(y_slice)

        self.window_train = torch.stack(train_stack).to(device=self.device, dtype=dtype)
        self.windows_targets = torch.stack(target_stack).to(device=self.device, dtype=dtype)

        print("Initialized dataset", self.window_train.shape, self.windows_targets.shape)

    def __len__(self):
        return self.window_train.shape[0]
    
    def __getitem__(self, idx):
        if self.eval:
            return self.window_train[idx]
        else:
            return self.window_train[idx], self.windows_targets[idx] # Training tensor for dataloader
    
    
    
