

import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression

df = pd.read_csv("/Users/apple/Documents/github/quantchallenge-2025/qch2025/pkg/trading/database.csv")

print(df.head())

df_features = df.drop(['home_win_target', 'Unnamed: 0'], axis=1)
df_targets = df['home_win_target']


logreg = LinearRegression()
logreg.fit(df_features, df_targets)

weights = logreg.coef_
intercept = logreg.intercept_

import json

d = {
    "weights": weights.tolist(),
    "intercept": intercept.tolist()
}

print(json.dumps(d))
