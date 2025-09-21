import torch
import torch.nn as nn
import torch.optim as optim

class RNN(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int,
                 layers: int,
                 dropout: float,
                 heads: int,
                 output_size: int,
                 device: torch.device = torch.device("mps"),
                 dtype: torch.dtype=torch.float16) -> None:
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size=hidden_size, num_layers=layers, dropout=dropout, bidirectional=True, batch_first=True).to(device=device, dtype=dtype)
        self.norm_1 = nn.LayerNorm(hidden_size*2).to(device=device, dtype=dtype)
        
        self.attn = nn.MultiheadAttention(hidden_size*2, heads, batch_first=True).to(device=device, dtype=dtype)
        self.norm_2 = nn.LayerNorm(hidden_size*2).to(device=device, dtype=dtype)

        self.fc = nn.Linear(in_features=hidden_size*2, out_features=output_size).to(device=device, dtype=dtype)

        self.device = device
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.norm_1(out)

        out, _ = out + self.attn(out, out, out)
        out = self.norm_2(out)
        
        out = self.fc(out)
        return out
