import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import datetime as time
#from sklearn.metrics import r2_score

import numpy as np

import qch2025.pkg.models.RNN.network as network
from qch2025.pkg.dataset import DS


def train(model: nn.RNN,
          dataset: DS,
          epochs: int = 5,
          batch_size: int = 32,
          decay: float = 0.1,
          learning_rate: float=0.001):
    rolling_mean: list[int] = []

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    cr = nn.MSELoss()
    mx = 0
    for ep in range(1, epochs+1):
        model.zero_grad()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # Each batch item is its own sliding window
        losses=[]
        t = time.datetime.now()
        for i, (tr, target) in enumerate(loader):
            pred_out = model.forward(tr)
            #loss = combined_loss(pred_out, target, alpha_decay=decay)
            loss = cr(pred_out, target)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())

            if i % 250 == 0:
                print(f"[{time.datetime.now()}] Current batch item: {i}, took {int((time.datetime.now()-t).total_seconds()*1000)} ms, current loss: {losses[-1]}, mean loss: {np.mean(np.array(losses))}")
                t = time.datetime.now()

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, "/home/ubuntu/repos/quantchallenge-2025/weights/weights.pth")

        """
            rsq, acc = check_training(model, dataset)
            if rsq > mx and ep > 3:
                print(f"Saving checkpoint: best rsq: {rsq} vs current: {m}")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, "/home/ubuntu/repos/quantchallenge-2025/weights/weights.pth")
                # Only save weights with best rsq
                mx = max(mx, rsq)
            rolling_mean.append(rsq)
        """

        print(f"{ep}: AVG OVERALL LOSS: {np.mean(np.array(losses))}, FINAL LOSS: {losses[-1]}, SMALLEST LOSS {min(losses)}, LARGEST LOSS: {max(losses)}")
    return rolling_mean   




def check_training(model: nn.RNN, dataset: DS):
    return 0, 0

    with torch.no_grad():
        preds = model.forward(dataset.eval_train).detach().cpu().numpy()
        eval = dataset.eval_targets.detach().cpu().numpy()

        # Denormalize for evaluation
        y1_p = preds[:, 0]
        y2_p = preds[:, 1]

        y1_t = eval[:, 0]
        y2_t = eval[:, 1]

        r1 = r2_score(y1_t, y1_p)
        r2 = r2_score(y2_t, y2_p)

        acc = np.mean(np.absolute(preds - eval) < 0.05)
    return (r1+r2)/2, acc




def eval(model: nn.RNN,
         dataset: DS,
         batch_size: int=1):
    with torch.no_grad():
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        preds_total = []
        for i, (tr) in enumerate(loader):
            pred = model.forward(tr)
            preds_total.append(pred)
        
        final = torch.cat(preds_total).to(device=model.device, dtype=torch.float32).flatten(0, 1)
        split = torch.split(final, 1, dim=1)
    return split[0].squeeze(1).detach().cpu().numpy(), split[1].squeeze(1).detach().cpu().numpy()
        



    


def combined_loss(preds: torch.Tensor, actual: torch.Tensor, alpha_decay: torch.Tensor):
    offset = 0.000001

    mse = torch.mean(torch.mean((preds-actual)**2, dim=-1)) # Combined mean to take in batch structure

    mean_true = torch.mean(actual, dim=(0,1)) # Return tensor of size [features]
    ss_res = torch.sum((preds-actual)**2, dim=(0,1)) # Tensor of size [features]
    ss_tot = torch.sum((actual-mean_true)**2, dim=(0,1))

    r2 = 1 - (ss_res/(ss_tot + offset))
    r2_final = torch.mean(r2)

    comb = mse + (alpha_decay * r2_final)

    return comb
