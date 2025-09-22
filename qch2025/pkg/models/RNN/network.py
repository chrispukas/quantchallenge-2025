import torch
import torch.nn as nn
import torch.optim as optim

class RNN(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int,
                 lstm_layers: int,
                 lstm_dropout: float,
                 attn_heads: int,
                 output_size: int,
                 device: torch.device = torch.device("mps"),
                 dtype: torch.dtype=torch.float16) -> None:
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(input_size, 
                            hidden_size=hidden_size, 
                            num_layers=lstm_dropout, 
                            dropout=lstm_dropout, 
                            bidirectional=True, 
                            batch_first=True).to(device=device, dtype=dtype)
        self.norm_1 = nn.LayerNorm(hidden_size*2).to(device=device, dtype=dtype)
        
        self.attn = nn.MultiheadAttention(hidden_size*2, attn_heads, batch_first=True).to(device=device, dtype=dtype)
        self.norm_2 = nn.LayerNorm(hidden_size*2).to(device=device, dtype=dtype)

        self.fc = nn.Linear(in_features=hidden_size*2, out_features=output_size).to(device=device, dtype=dtype)

        self.device = device
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.norm_1(out)

        attn_out, _ = self.attn(out, out, out)
        out = self.norm_2(out + attn_out)
        
        out = self.fc(out)
        return out
