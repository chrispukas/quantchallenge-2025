import json
import os
import pandas as pd
import numpy as np

dicts = []

event_mapping = {
    'JUMP_BALL': 0,
    'SCORE': 1,
    'MISSED': 2,
    'REBOUND': 3,
    'STEAL': 4,
    'BLOCK': 5,
    'TURNOVER': 6,
    'FOUL': 7,
    'TIMEOUT': 8,
    'SUBSTITUTION': 9,
    'START_PERIOD': 10,
    'END_PERIOD': 11,
    'END_GAME': 12,
    'DEADBALL': 13,
    'NOTHING': 14,
    'UNKNOWN': 15,
}


shot_mapping = {
    'THREE_POINT': 0,
    'TWO_POINT': 1,
    'FREE_THROW': 2,
    'DUNK': 3,
    'LAYUP': 4,
    None: 5,
}

home_away_mapping = {
    'home': 0,
    'away': 1,
    'unknown': 2,
}


path_to_games = "/Users/apple/Documents/github/quantchallenge-2025/qch2025/pkg/trading/games"

game_names = [
    "example-game.json",
    'game1.json',
    'game2.json',
    'game3.json',
    'game4.json',
    'game5.json',
]


for g in game_names:
    path_to_events: str = os.path.join(path_to_games, g)
    with open(path_to_events) as json_data:
        data = json.load(json_data)
        json_data.close()
    for i in range(len(data)):
        d = data[i]

        one_shot_event = [0] * 16
        one_shot_event[event_mapping[d.get("event_type")]] = 1

        one_shot_shot = [0] * 6
        one_shot_shot[shot_mapping[d.get("shot_type")]] = 1

        one_shot_shot_home_away = [0] * 3
        one_shot_shot_home_away[home_away_mapping[d.get("home_away")]] = 1

        mp = 150

        dicts.append({
            'home_away': one_shot_shot_home_away,
            'home_score': d.get("home_score") / mp,
            'away_score': d.get("away_score") / mp,
            'score_diff': (d.get("home_score") - d.get("away_score")) / mp,
            'event_type': one_shot_event,
            'shot_type': one_shot_shot,
            'substituted_player': int(d.get('substituted_player_name') is not None and d.get('substituted_player_name') != None),
            'coordinate_x': 0 if d.get('coordinate_x') is None else (d.get('coordinate_x') + 5)/100,
            'coordinate_y': 0 if d.get('coordinate_y') is None else (d.get('coordinate_y') + 5)/100,
            'time_seconds': 2 * (1-d.get("time_seconds")/2880) - 1
        })
df = pd.DataFrame(dicts)
home_away_df = pd.DataFrame(df['home_away'].tolist(), columns=[f'home_away_{i}' for i in range(3)])
event_df = pd.DataFrame(df['event_type'].tolist(), columns=[f'event_{i}' for i in range(16)])
shot_df = pd.DataFrame(df['shot_type'].tolist(), columns=[f'shot_{i}' for i in range(6)])

df = pd.concat([df.drop(['home_away', 'event_type', 'shot_type'], axis=1), home_away_df, event_df, shot_df]).fillna(0)


k_factor = 0.05
df['home_win_target'] = 1 / (1 + np.exp(-k_factor * df["score_diff"]))

df.to_csv("/Users/apple/Documents/github/quantchallenge-2025/qch2025/pkg/trading/database.csv")

print(df.head())
print(len(df['home_win_target']))


