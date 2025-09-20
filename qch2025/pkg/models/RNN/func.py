import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import datetime as time

import qch2025.pkg.models.RNN.network as network
from qch2025.pkg.dataset import DS


def train(model: nn.RNN,
          dataset: DS,
          epochs: int=5,
          learning_rate: float=0.001):
    rolling_mean: list[int] = []

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    cr = nn.MSELoss()

    for ep in range(1, epochs+1):
        model.zero_grad()
        loader = DataLoader(dataset, batch_size=1, shuffle=True) # Each batch item is its own sliding window
        losses=[]
        t = time.datetime.now()
        for i, (tr, target) in enumerate(loader):
            tr = tr.squeeze(0)
            target = target.squeeze(0)

            pred_out = model.forward(tr)
            loss = cr(pred_out, target)

            losses.append(loss)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()

            if i % 100 == 0:
                s_1, d_1 = sum(losses), len(losses)
                print(f"[{time.datetime.now()}] Current batch item: {i}, took {int((time.datetime.now()-t).total_seconds()*1000)} ms, loss: {loss.item()}, mean loss: {(s_1)/d_1}")

                t = time.datetime.now()
        

        s, d = sum(rolling_mean), len(rolling_mean) + 1
        rolling_mean.append((s+loss.item())/d)

        print(f"{ep}: loss mean = {rolling_mean[-1]}")
    return rolling_mean   


def eval(model: nn.RNN,
         dataset: DS):
    with torch.no_grad():
        all = dataset.train
        ids = dataset.ids
        preds = model.forward(all)
    
    return preds.detach().cpu().numpy(), ids.detach().cpu().numpy()
        



    