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
        
        self.eval = eval
        self.device = device

        # Initialize the dataframes, extending the frame to take in all datasets
        self.df = pd.read_csv(dataset_path)
        df2 = pd.read_csv(dataset_path_2)
        self.df = pd.concat([self.df.drop(["time"], axis=1), df2.fillna(0)], axis=1)

        
        # Appl feature engineering to extend the state space
        self.cols_to_normalize = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', "N", "O", "P"]
        self.all_cols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', "N", "O", "P"]

        features_to_extend = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', "N"]
        for feature in features_to_extend:
            params = [f"{feature}_diff", f"{feature}_pos", f"{feature}_neg"]
            self.df[f"{feature}_diff"] = self.df[feature].diff().fillna(0)
            
            self.df[f"{feature}_pos"] = (self.df[feature] > 0).astype(int)
            self.df[f"{feature}_neg"] = (self.df[feature] < 0).astype(int)

            self.all_cols.extend(params)
            self.cols_to_normalize.append(f"{feature}_diff")

        # Feature normailzation
        self.mean = self.df[self.cols_to_normalize].mean(skipna=True)
        self.std = self.df[self.cols_to_normalize].std().replace(0, 1)

        self.entries = self.df.copy(deep=True)
        self.entries[self.cols_to_normalize] = (self.df[self.cols_to_normalize] - self.mean) / self.std # Normalizes the dataset


        # Feature evaluation mode, only copies features, not targets
        if eval:
            self.ids = torch.tensor(self.entries["id"]).to(self.device, dtype=dtype)
            self.features = torch.tensor(self.entries.drop(["id"], axis = 1)\
                                  .values.astype(np.float32)).to(self.device, dtype=dtype)

            feature_stack = []
            n = self.features.shape[0]

            for l in range(0, n, window_steps): 
                r = l + window_size
                tr_slice = self.features[l:r, :]
                if r > n:
                    print(f"Out of range by: {r-n} items!" )
                    pad_len = r-n
                    pad_tr = tr_slice[-1:].repeat(pad_len, 1)
                    tr_slice = torch.cat([tr_slice, pad_tr], dim=0)
                
                feature_stack.append(tr_slice)

            self.window_features = torch.stack(feature_stack).to(device=self.device, dtype=dtype)
            return
        
        self.entries["Y1"] = self.df["Y1"]
        self.entries["Y2"] = self.df["Y2"]

        self.features = torch.tensor(self.entries.drop(['Y1', "Y2"], axis = 1)\
                                  .values.astype(np.float32)).to(self.device, dtype=dtype)
        self.y = torch.stack((
            torch.tensor(self.entries['Y1']).to(self.device, dtype=dtype),
            torch.tensor(self.entries['Y2']).to(self.device, dtype=dtype),
        )).to(self.device).transpose(1,0)

        print(self.features.shape) # [points, features]
        print(self.y.shape) # [points, features]

        train_stack, target_stack = [], []

        for l in range(0, self.features.shape[0], window_steps): 
            r = l + window_size
            tr_slice = self.features[l:r, :]
            y_slice = self.y[l:r, :]
            
            if r > n:
                print(f"Out of range by: {r-n} items!" )
                pad_len = r-n
                pad_tr = tr_slice[-1:].repeat(pad_len, 1)
                pad_y = y_slice[-1:].repeat(pad_len, 1)

                tr_slice = torch.cat([tr_slice, pad_tr], dim=0)
                y_slice = torch.cat([y_slice, pad_y], dim=0)
            
            train_stack.append(tr_slice)
            target_stack.append(y_slice)

        self.window_features = torch.stack(train_stack).to(device=self.device, dtype=dtype)
        self.window_y = torch.stack(target_stack).to(device=self.device, dtype=dtype)

        print("Initialized dataset", self.window_features.shape, self.window_y.shape)

    def __len__(self):
        return self.window_features.shape[0]
    
    def __getitem__(self, idx):
        if self.eval:
            return self.window_features[idx]
        else:
            return self.window_features[idx], self.window_y[idx] # Training tensor for dataloader
    
    
    
