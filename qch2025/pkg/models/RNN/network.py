import torch
import torch.nn as nn
import torch.optim as optim

class RNN(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 output_size: int,
                 device: torch.device = torch.device("cpu")) -> None:
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size=hidden_size)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)

        self.device = device
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out).squeeze(0) # mapping final output, given all previous inputs
        return out.squeeze(0)