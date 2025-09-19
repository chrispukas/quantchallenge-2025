import torch
import torch.nn as nn
import torch.optim as optim

import qch2025.pkg.models.RNN.network as network
from qch2025.pkg.dataset import Dataset


def train(model: nn.RNN,
          dataset: Dataset,
          epochs: int=5,
          batch_size: int=25,
          learning_rate: float=0.001):
    curr_loss, all_loss = 0, []
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    cr = nn.MSELoss()

    pred = []

    for ep in range(1, epochs+1):
        model.zero_grad()
        batches = dataset.get_batches(batch_size=batch_size)
        print(len(batches))
        for batch in batches:
            (t, a, b, c, d, e, f, g, h, i, j, k, l, m, n, y1, y2) = zip(*batch)

            input = (
                torch.tensor(a).to(model.device, dtype=torch.float32),
                torch.tensor(b).to(model.device, dtype=torch.float32),
                torch.tensor(c).to(model.device, dtype=torch.float32),
                torch.tensor(d).to(model.device, dtype=torch.float32),
                torch.tensor(e).to(model.device, dtype=torch.float32),
                torch.tensor(f).to(model.device, dtype=torch.float32),
                torch.tensor(g).to(model.device, dtype=torch.float32),
                torch.tensor(h).to(model.device, dtype=torch.float32),
                torch.tensor(i).to(model.device, dtype=torch.float32),
                torch.tensor(j).to(model.device, dtype=torch.float32),
                torch.tensor(k).to(model.device, dtype=torch.float32),
                torch.tensor(l).to(model.device, dtype=torch.float32),
                torch.tensor(m).to(model.device, dtype=torch.float32),
                torch.tensor(n).to(model.device, dtype=torch.float32),
            )
            input = torch.stack(input).permute(1, 0)

            output = (
                torch.tensor(y1).to(model.device, dtype=torch.float32),
                torch.tensor(y2).to(model.device, dtype=torch.float32),
            )
            output = torch.stack(output).permute(1, 0)


            pred_out = model.forward(input)
            loss = cr(pred_out, output)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        all_loss.append(loss.item()/len(batch))
        print(f"{ep}: avg loss = {all_loss[-1]}")
    return all_loss   


def eval():
    pass
        



    