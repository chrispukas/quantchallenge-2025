import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import datetime as time
from sklearn.metrics import r2_score

import numpy as np

import qch2025.pkg.models.RNN.network as network
from qch2025.pkg.dataset import DS


def train(model: nn.RNN,
          dataset: DS,
          epochs: int=5,
          batch_size: int=32,
          learning_rate: float=0.001):
    rolling_mean: list[int] = []

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    cr = nn.MSELoss()

    for ep in range(1, epochs+1):
        model.zero_grad()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # Each batch item is its own sliding window
        losses=[]
        t = time.datetime.now()
        for i, (tr, target) in enumerate(loader):
            pred_out = model.forward(tr)
            loss = cr(pred_out, target)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()

            if i % 250 == 0:
                losses.append(loss.item())
                s_1, d_1 = sum(losses), len(losses)
                print(f"[{time.datetime.now()}] Current batch item: {i}, took {int((time.datetime.now()-t).total_seconds()*1000)} ms, loss: {loss.item()}, mean loss: {(s_1)/d_1}")
                t = time.datetime.now()

        rsq = check_training(model, dataset)
        m = max(rolling_mean) if len(rolling_mean) > 0 else 0
        if rsq > m:
            print(f"Saving checkpoint: best rsq: {rsq} vs current: {m}")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, "/home/ubuntu/repos/quantchallenge-2025/weights/weights.pth")
            # Only save weights with best rsq
        rolling_mean.append(rsq)

        print(f"{ep}: loss mean = {rolling_mean[-1]}, RSQ: {rsq}")
    return rolling_mean   




def check_training(model: nn.RNN, dataset: DS):
    with torch.no_grad():
        preds = model.forward(dataset.eval_train).detach().cpu().numpy()
        eval = dataset.eval_targets.detach().cpu().numpy()

        # Denormalize for evaluation
        y1_p = preds[:, 0] * dataset.std["Y1"] + dataset.mean["Y1"]
        y2_p = preds[:, 1] * dataset.std["Y2"] + dataset.mean["Y2"] 

        y1_t = eval[:, 0] * dataset.std["Y1"] + dataset.mean["Y1"]
        y2_t = eval[:, 1] * dataset.std["Y2"] + dataset.mean["Y2"]

        r1 = r2_score(y1_t, y1_p)
        r2 = r2_score(y2_t, y2_p)
    return (r1+r2)/2




def eval(model: nn.RNN,
         dataset: DS,
         batch_size: int=1):
    with torch.no_grad():
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        y1, y2 = {}, {}
        for i, (tr) in enumerate(loader):
            pred = model.forward(tr)
            pred = pred.detach().cpu().numpy()

            for j, elem in enumerate(pred):
                idx = i+j
                if idx in y1:
                    y1[idx].append(elem[0])
                    y2[idx].append(elem[1])
                else:
                    y1[idx] = [elem[0]]
                    y2[idx] = [elem[1]]
        
        y1_f, y2_f, ids = np.zeros(len(y1)), np.zeros(len(y2)), []
        for i, v in enumerate(y1.values()):
            y1_f[i] = np.mean(v)
            ids.append(i)

        for i, v in enumerate(y2.values()):
            y2_f[i] = np.mean(v)
    
    y1_f = y1_f * dataset.std["Y1"] + dataset.mean["Y1"]
    y2_f = y2_f * dataset.std["Y2"] + dataset.mean["Y2"]
    
    return y1_f, y2_f, ids
        



    