from qch2025.pkg.models.RNN import network as network
from qch2025.pkg.models.RNN import func as func

from qch2025.pkg.dataset import DS

import qch2025.pkg.plotting as plt

import numpy as np
import pandas as pd
import torch

n_epochs = 20
batch_size = 200
window_steps = 16

learning_rate = 0.001

dtype = torch.float32

# Training
train_dataset = DS(dataset_path="/Users/apple/Documents/github/quantchallenge-2025/qch2025/dataset/train.csv",
                        window_size=batch_size,
                        window_steps=window_steps,
                        dtype=dtype)

rnn = network.RNN(14, 128, 2, device=torch.device("cuda"), dtype=dtype)
losses = func.train(rnn, dataset=train_dataset, epochs=n_epochs, learning_rate=learning_rate)

plt.plot_line(np.arange(1, n_epochs+1), losses)


# Evaluation
eval = DS(dataset_path="/Users/apple/Documents/github/quantchallenge-2025/qch2025/dataset/test.csv",
                    window_size=batch_size,
                    window_steps=1,
                    eval=True,
                    device=torch.device("cuda"),
                    dtype=dtype)
print(eval.ids.shape, eval.train.shape)
preds, ids = func.eval(rnn, eval)

y1, y2 = zip(*preds)

df = pd.DataFrame({"Y1": y1, "Y2": y2})
df.index.name = "id"
df.index = df.index + 1
df.to_csv("/Users/apple/Documents/github/quantchallenge-2025/qch2025/outputs/predicted.csv")
