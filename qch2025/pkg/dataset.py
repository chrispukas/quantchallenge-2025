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
                 extension_path: str = None,
                 mean_override=None,
                 std_override=None,
                 device: torch.device=torch.device("cuda"),
                 dtype: torch.dtype=torch.float16) -> None:
        
        self.eval = eval
        self.device = device
        self.extension_size = 0

        # Initialize the dataframes, extending the frame to take in all datasets
        self.df = pd.read_csv(dataset_path, index_col=False)
        self.df = self.df.drop(["time"], axis=1)
        print(self.df.columns)


        if eval and extension_path:
            self.extension_size = window_size*2
            df_extend = pd.read_csv(extension_path, index_col=False).drop(["time", "Y1", "Y2"], axis=1)
            r = len(df_extend["A"])
            l = r-self.extension_size
            extra = df_extend.iloc[l:r]

            print(f"Extended dataframe by: {len(extra["A"])} items.")

            self.df = pd.concat([extra, self.df], ignore_index=True)

        
        # Appl feature engineering to extend the state space
        self.cols_to_normalize = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', "N", "O", "P"]
        self.all_cols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', "N", "O", "P"]

        features_to_extend = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', "N"]
        for feature in features_to_extend:
            params = [f"{feature}_diff", f"{feature}_dir", f"{feature}_sma5", f"{feature}_vol5",
                      f"{feature}_ret", f"{feature}_sign",
                      f"{feature}_sma20", f"{feature}_vol20",
                      f"{feature}_zscore5",
                      f"{feature}_lag1", f"{feature}_lag2", f"{feature}_lag5"]
            self.df[f"{feature}_diff"] = self.df[feature].diff().fillna(0)
            
            self.df[f"{feature}_sma5"] = self.df[feature].rolling(5).mean().fillna(0)
            self.df[f"{feature}_vol5"] = self.df[feature].rolling(5).std().fillna(0)

            self.df[f"{feature}_ret"] = self.df[feature].pct_change().fillna(0)

            self.df[f"{feature}_sign"] = np.where(self.df[feature] > 0, 1, -1)
            self.df[f"{feature}_dir"] = np.where(self.df[f"{feature}_ret"] > 0, 1, -1)

            self.df[f"{feature}_zscore5"] = (
                (self.df[feature] - self.df[f"{feature}_sma5"]) / (self.df[f"{feature}_vol5"].replace(0,1))
            )


            self.df[f"{feature}_sma20"] = self.df[feature].rolling(20).mean().fillna(0)
            self.df[f"{feature}_vol20"] = self.df[feature].rolling(20).std().fillna(0)

            self.df[f"{feature}_lag1"] = self.df[feature].shift(1).fillna(0)
            self.df[f"{feature}_lag2"] = self.df[feature].shift(2).fillna(0)
            self.df[f"{feature}_lag5"] = self.df[feature].shift(5).fillna(0)

            self.all_cols.extend(params)
            self.cols_to_normalize.extend([f"{feature}_diff", f"{feature}_ret", f"{feature}_vol5", f"{feature}_vol20",
                                           f"{feature}_zscore5",])

        # Feature normailzation
        self.mean = self.df[self.cols_to_normalize].mean(skipna=True) if mean_override is None else mean_override
        self.std = self.df[self.cols_to_normalize].std().replace(0, 1) if std_override is None else std_override

        if mean_override is not None:
            print(f"Overrided mean to: {mean_override}, and std to: {std_override}")

        self.entries = self.df.copy(deep=True)
        self.entries[self.cols_to_normalize] = (self.df[self.cols_to_normalize] - self.mean) / self.std # Normalizes the dataset


        # Feature evaluation mode, only copies features, not targets
        if eval:
            cols = self.entries.columns
            if "id" in cols:
                self.ids = torch.tensor(self.entries["id"]).to(self.device, dtype=dtype)
            else:
                self.entries = self.entries.drop(['Y1', 'Y2'], axis=1)

                un = "Unnamed: 0"
                self.entries[un] = self.entries[un] + 1
                self.ids = torch.tensor(self.entries[un]).to(self.device, dtype=dtype)

            self.entries = self.entries.drop(["id", "Unnamed: 0"], axis=1)
            self.features = torch.tensor(self.entries.values.astype(np.float32)).to(self.device, dtype=dtype)
                
            print(list(self.entries.columns))

            self.max_padding = 0
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

                    self.max_padding = pad_len
                
                feature_stack.append(tr_slice)

            self.window_features = torch.stack(feature_stack).to(device=self.device, dtype=dtype)
            return

        self.entries = self.entries.drop(["Unnamed: 0"], axis=1)
        
        self.entries["Y1"] = self.df["Y1"]
        self.entries["Y2"] = self.df["Y2"]

        print(self.entries.columns)

        self.features = torch.tensor(self.entries.drop(['Y1', "Y2"], axis = 1)\
                                  .values.astype(np.float32)).to(self.device, dtype=dtype)
        self.y = torch.stack((
            torch.tensor(self.entries['Y1']).to(self.device, dtype=dtype),
            torch.tensor(self.entries['Y2']).to(self.device, dtype=dtype),
        )).to(self.device).transpose(1,0)

        print(self.features.shape) # [points, features]
        print(self.y.shape) # [points, features]

        self.max_padding = 0

        train_stack, target_stack = [], []

        n = self.features.shape[0]

        for l in range(0, n, window_steps): 
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
                
                self.max_padding = pad_len
            
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
    
    
    
