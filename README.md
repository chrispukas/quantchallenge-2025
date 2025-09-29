# 1.1 Submission for the QuantChallenge.org 2025 competiton

The competition consisted of 2 components (Research, and Trading) over a 7 day period. This repo contains the submission code for both componenets, and below is an outline of the function of each repo.

This package can be installed by doing the following (MacOS/Linux):
```
git clone https://github.com/chrispukas/quantchallenge-2025.git;
cd ./quantchallenge-2025; bash install.sh;
```

Notebooks can be found in the following repo: ```/Users/apple/Documents/github/quantchallenge-2025/scripts/notebooks```

## 2.1. Research

Implementation of a hybrid attention + LSTM architecture to determine long and short range temporal time-series datapoints, predicting future outcomes. Feature engineering was used to expand the feature-space (momentums, differences, rolling mean, etc). This implementation can be found in the following path:```./qch2025/pkg/research/models/```

## 2.2. Trading

Implementation of a hybrid double-regression regression architecture, with 1 fixed layer, trained on previous game states, and another dynamic layer, with online training to determine unique game dependencies based on a rolling window. ```./qch2025/pkg/trading``` 


![Alt text](/res/img/Screenshot%202025-09-29%20at%2011.30.58.png)