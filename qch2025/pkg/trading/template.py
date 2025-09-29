"""
Quant Challenge 2025

Algorithmic strategy template
"""

from enum import Enum
from typing import Optional

import random
import numpy as np
import json

from sklearn.linear_model import LinearRegression, LogisticRegression
from collections import deque

# ----------------
# ---- CONFIG ----
# ----------------

weights = '{"weights": [0.004166662819330678, -0.004166657496765486, 0.00833332031609705, -1.087690472955819e-10, -3.5944862382913247e-10, 8.057880823442242e-10, -1.7175665473479895e-09, 1.6343559516944177e-09, 1.6343559517956766e-09, 1.6343559516590469e-09, 1.6343559519493939e-09, 1.6343559537922396e-09, 1.634355954285096e-09, 1.6343559482814281e-09, 1.6343559489811514e-09, 1.6343559518442553e-09, 1.6343559512671438e-09, 1.6343559544635965e-09, 1.6343559491626803e-09, 1.6343559504782923e-09, 1.6343559520587112e-09, 1.6343559526569896e-09, 1.6343559519186935e-09, 2.168404344971009e-19, 1.634355950775867e-09, 1.6343559527483379e-09, 1.6343559497087587e-09, 1.6343559540464018e-09, 1.6343559543806878e-09, 1.634355953200088e-09, 1.634355952197975e-09, 1.634355950928503e-09], "intercept": 0.49999999836564407}'
weights = '{"weights": [0.004166652275794218, -0.004166655221812146, 0.008333307497569853, -4.861945307926604e-10, 3.1978285913539027e-10, -3.1978213972840494e-10, 1.3955326418583748e-09, -1.1756656910993373e-09, -1.1756656940226411e-09, -1.1756656904267952e-09, -1.1756657157678883e-09, -1.1756656954917317e-09, -1.1756656872670637e-09, -1.1756657012281147e-09, -1.1756656901481087e-09, -1.1756656487600672e-09, -1.175665711406349e-09, -1.1756656891539086e-09, -1.1756656638159987e-09, -1.1756656932471301e-09, -1.1756657150104462e-09, -1.175665730928725e-09, -1.1756657241273005e-09, -1.1756657230620784e-09, -1.1756656903262967e-09, -1.1756657024279796e-09, -1.1756657152160161e-09, -1.1756656840261738e-09, -1.1756656937449543e-09, -1.1756656736081427e-09, -1.1756656882336412e-09, -1.1756656909964922e-09], "intercept": 0.5000000011756702}'

steps_before_informed = 5
steps_per_retrain = 5
orders = 0

exploration_chance = 1
exploration_decay = 0.9

max_trades_per_min = 25
curr_capital = 100_000
exposure = 0

max_position = 10_000
min_position = 10_000


# -------------------------
# ---- EVENT VARIABLES ----
# -------------------------

market_price = 0
market_bid = deque(maxlen=5)
market_ask = deque(maxlen=5)

train_step = 0
exploration_chance_current = exploration_chance

current_position = 0

curr_prob_raw_buffer = []
curr_prob_informed_buffer = []

# [predicted_probability, market_price, profit/loss, chance_probability]
side_buffer = deque(maxlen=25)

# [predicted_probability, market_price, profit/loss, chance_probability]
profit_buffer = deque(maxlen=25)

orders = deque(maxlen=20)

# ----------------------------------------------------------
# ---- Linear Regression to Encode Events to Win Chance ----
# ----------------------------------------------------------

def predict_home_probability(x):
    dc = json.loads(weights)
    up = np.dot(dc.get("weights"), x) + dc.get("intercept")
    return 1 / (1 + np.exp(-up))

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
def process_event(event: dict):
    one_shot_event = [0] * 16
    one_shot_event[event_mapping[event.get("event_type")]] = 1

    one_shot_shot = [0] * 6
    one_shot_shot[shot_mapping[event.get("shot_type")]] = 1

    one_shot_shot_home_away = [0] * 3
    one_shot_shot_home_away[home_away_mapping[event.get("home_away")]] = 1

    mp = 150

    numeric =  [
        event.get("home_score") / mp,
        event.get("away_score") / mp,
        (event.get("home_score") - event.get("away_score")) / mp,
        int(event.get('substituted_player_name') is not None and event.get('substituted_player_name') != None),
        0 if event.get('coordinate_x') is None else (event.get('coordinate_x') + 5)/100,
        0 if event.get('coordinate_y') is None else (event.get('coordinate_y') + 5)/100,
        2 * (1-event.get("time_seconds")/2880) - 1
    ]

    features = one_shot_shot + one_shot_event + one_shot_shot_home_away + numeric # Append one shot 1
    return np.array(features, dtype=float)



# --------------------------------------------------------------------
# ---- Linear Regression to Encode Win Chance to Potential Income ----
# --------------------------------------------------------------------

model_side: LinearRegression = None # Selects if we should buy up or buy down
model_profit: LinearRegression = None # Determines the coressponding profit for a current trade

def recalculate_weights():
    global model_side, model_profit
    if len(side_buffer) < 5 or len(profit_buffer) < 5:
        return
    prob, mp, prf = zip(*side_buffer)
    x = np.column_stack((prob, mp))
    y = np.array(prf)
    model_side = LogisticRegression(solver='liblinear')
    model_side.fit(x, y)

    prob, mp, prf = zip(*profit_buffer)
    x = np.column_stack((prob, mp))
    y = np.array(prf)
    model_profit = LinearRegression()
    model_profit.fit(x, y)
    








class Side(Enum):
    BUY = 0
    SELL = 1

class Ticker(Enum):
    # TEAM_A (home team)
    TEAM_A = 0

def place_market_order(side: Side, ticker: Ticker, quantity: float) -> None:
    """Place a market order.
    
    Parameters
    ----------
    side
        Side of order to place
    ticker
        Ticker of order to place
    quantity
        Quantity of order to place
    """
    return

def place_limit_order(side: Side, ticker: Ticker, quantity: float, price: float, ioc: bool = False) -> int:
    """Place a limit order.
    
    Parameters
    ----------
    side
        Side of order to place
    ticker
        Ticker of order to place
    quantity
        Quantity of order to place
    price
        Price of order to place
    ioc
        Immediate or cancel flag (FOK)

    Returns
    -------
    order_id
        Order ID of order placed
    """
    return 0

def cancel_order(ticker: Ticker, order_id: int) -> bool:
    """Cancel an order.
    
    Parameters
    ----------
    ticker
        Ticker of order to cancel
    order_id
        Order ID of order to cancel

    Returns
    -------
    success
        True if order was cancelled, False otherwise
    """
    return 0

class Strategy:
    """Template for a strategy."""

    def reset_state(self) -> None:
        """Reset the state of the strategy to the start of game position.
        
        Since the sandbox execution can start mid-game, we recommend creating a
        function which can be called from __init__ and on_game_event_update (END_GAME).

        Note: In production execution, the game will start from the beginning
        and will not be replayed.
        """

        global curr_prob_raw_buffer, curr_prob_informed_buffer, weight_buffer, exploration_chance_current, train_step, curr_capital, current_position, exposure, market_bid, market_ask

        # Reset buffers, and variables from previous game
        curr_prob_raw_buffer = []
        curr_prob_informed_buffer = []

        weight_buffer = []

        exploration_chance_current = exploration_chance
        train_step = 0
        current_position = 0

        curr_capital = 100_000
        exposure = 0

        market_bid = deque(maxlen=5)
        market_ask = deque(maxlen=5)

        pass

    def __init__(self) -> None:
        """Your initialization code goes here."""
        self.reset_state()

    def on_trade_update(
        self, ticker: Ticker, side: Side, quantity: float, price: float
    ) -> None:
        """Called whenever two orders match. Could be one of your orders, or two other people's orders.
        Parameters
        ----------
        ticker
            Ticker of orders that were matched
        side:
            Side of orders that were matched
        quantity
            Volume traded
        price
            Price that trade was executed at
        """
        print(f"Python Trade update: {ticker} {side} {quantity} shares @ {price}")

    def on_orderbook_update(
        self, ticker: Ticker, side: Side, quantity: float, price: float
    ) -> None:
        """Called whenever the orderbook changes. This could be because of a trade, or because of a new order, or both.
        Parameters
        ----------
        ticker
            Ticker that has an orderbook update
        side
            Which orderbook was updated
        price
            Price of orderbook that has an update
        quantity
            Volume placed into orderbook
        """


        global market_price, curr_capital, orders, market_bid, market_ask

        if side == Side.BUY:
            market_bid.append(price)
            
        else:
            market_ask.append(price)
        
        if market_bid and market_ask and len(market_bid) > 0 and len(market_ask) > 0:
            mid = (market_bid[-1] + market_ask[-1])/2
            market_price = (0.7 * market_price + 0.3 * mid) if market_price != 0 else mid
        else:
            market_price = 0

        if len(curr_prob_raw_buffer) == 0:
            return
        
        if orders:
            entry_price = orders[-1]
            if side == Side.BUY:
                trade_profit = quantity * (price-entry_price)
            elif side == Side.SELL:
                trade_profit = quantity * (entry_price - price)
        else:
            trade_profit = 0
        orders.append(price)

        raw_prob_one = curr_prob_raw_buffer[-1]
        side_item = ( 
            raw_prob_one, # Estimated probability
            price,
            1 if ((side==Side.BUY and trade_profit > 0) or (side==Side.SELL and trade_profit < 0)) else 0,
        )
        profit_item = (
            raw_prob_one, # Estimated probability
            price, # Current market price
            trade_profit, # Profit vs Loss
        )
        
        side_buffer.append(side_item)
        profit_buffer.append(profit_item)

        pass

    def on_account_update(
        self,
        ticker: Ticker,
        side: Side,
        price: float,
        quantity: float,
        capital_remaining: float,
    ) -> None:
        """Called whenever one of your orders is filled.
        Parameters
        ----------
        ticker
            Ticker of order that was fulfilled
        side
            Side of order that was fulfilled
        price
            Price that order was fulfilled at
        quantity
            Volume of order that was fulfilled
        capital_remaining
            Amount of capital after fulfilling order
        """

        global exposure, market_price, curr_capital
        exposure += curr_capital-capital_remaining


        change = curr_capital-capital_remaining
        price = market_price

        if len(curr_prob_raw_buffer) < 1:
            return

        momentum_1 = (profit_buffer[-1][-1] - profit_buffer[-2][-1])if len(profit_buffer) > 1 else 0
        momentum_3 = (profit_buffer[-1][-1] - profit_buffer[-4][-1]) if len(profit_buffer) > 3 else 0 
        momentum_5 = (profit_buffer[-1][-1] - profit_buffer[-6][-1]) if len(profit_buffer) > 5 else 0  

        exp_norm = exposure / curr_capital

        raw_prob_one = curr_prob_raw_buffer[-1]

        side_item = ( 
            raw_prob_one, # Estimated probability
            price,
            momentum_1,
            momentum_3,
            momentum_5,
            exp_norm,
            1 if change > 0 else -1,

        )
        profit_item = (
            raw_prob_one, # Estimated probability
            price, # Current market price
            momentum_1,
            momentum_3,
            momentum_5,
            exp_norm,
            change, # Profit vs Loss
        )
        
        side_buffer.append(side_item)
        profit_buffer.append(profit_item)

        curr_capital = capital_remaining

        pass

    def on_game_event_update(self,
                           event_type: str,
                           home_away: str,
                           home_score: int,
                           away_score: int,
                           player_name: Optional[str],
                           substituted_player_name: Optional[str],
                           shot_type: Optional[str],
                           assist_player: Optional[str],
                           rebound_type: Optional[str],
                           coordinate_x: Optional[float],
                           coordinate_y: Optional[float],
                           time_seconds: Optional[float]
        ) -> None:
        """Called whenever a basketball game event occurs.
        Parameters
        ----------
        event_type
            Type of event that occurred
        home_score
            Home team score after event
        away_score
            Away team score after event
        player_name (Optional)
            Player involved in event
        substituted_player_name (Optional)
            Player being substituted out
        shot_type (Optional)
            Type of shot
        assist_player (Optional)
            Player who made the assist
        rebound_type (Optional)
            Type of rebound
        coordinate_x (Optional)
            X coordinate of shot location in feet
        coordinate_y (Optional)
            Y coordinate of shot location in feet
        time_seconds (Optional)
            Game time remaining in seconds
        """

        event = {
            "event_type":event_type,
            'home_away': home_away,
            'home_score': home_score,
            'away_score': away_score,
            'player_name': player_name,
            'substituted_player_name': substituted_player_name,
            'shot_type': shot_type,
            'assist_player': assist_player,
            'rebound_type': rebound_type,
            'coordinate_x': coordinate_x,
            'coordinate_y': coordinate_y,
            'time_seconds': time_seconds,
        }

        score_diff = event.get("home_score") - event.get("away_score")

        global market_price, exploration_chance_current, exploration_decay, curr_prob_informed_buffer, steps_before_informed, steps_per_retrain, max_trades_per_min, exposure, current_position, min_position, max_position

        ev = process_event(event)
        curr_prob_raw = predict_home_probability(ev) # Keeps track of the current probability

        global curr_prob_raw_buffer
        curr_prob_raw_buffer.append(curr_prob_raw)

        print(f"{curr_prob_raw}")

        prob_profit = prob_side = 0

        # Allows for exploration, providing decay with every time step
        rand = random.random()
        if (exploration_chance_current - rand) > 0:
            exploration_chance_current *= exploration_decay
            side_rand = np.round(random.random())

            side = Side.BUY if side_rand > 0.5 else Side.SELL
                
            place_market_order(side=side, ticker=Ticker.TEAM_A, quantity=1)
            print(f"EXPLORATION: {event_type} {home_score} - {away_score}")
        else:
            l = len(profit_buffer)
            if l % steps_per_retrain == 0:
                # Retrain models
                recalculate_weights()

            momentum_1 = (profit_buffer[-1][-1] - profit_buffer[-2][-1]) if len(profit_buffer) > 1 else 0
            momentum_3 = (profit_buffer[-1][-1] - profit_buffer[-4][-1]) if len(profit_buffer) > 3 else 0 
            momentum_5 = (profit_buffer[-1][-1] - profit_buffer[-6][-1]) if len(profit_buffer) > 5 else 0 
                
            exp_norm = exposure / curr_capital

            profit_x = np.array([curr_prob_raw, market_price, momentum_1, momentum_3, momentum_5, exp_norm]) # [predicted_probability, market_price]
            if profit_x.ndim == 1:
                profit_x = profit_x.reshape(1, -1)
            prob_profit = model_profit.predict(profit_x)
            if isinstance(prob_profit, list):
                prob_profit = prob_profit[0]

            side_x = np.array([curr_prob_raw, market_price, momentum_1, momentum_3, momentum_5, exp_norm]) # [predicted_probability, market_price]
            if side_x.ndim == 1:
                side_x = side_x.reshape(1, -1)
            prob_side = model_side.predict(side_x)
            if isinstance(prob_side, list):
                prob_side = prob_side[0]

            side = Side.BUY if curr_prob_raw > 0.5 else Side.SELL
            print(f"profit estimate: {prob_profit}, side estimate: {prob_side}")

            if prob_profit > 2 and curr_prob_raw > 0.25: # Safer

                base_quantity = 100

                confidence_adj = abs(curr_prob_raw - 0.5) * 75
                quantity = base_quantity + base_quantity * confidence_adj


                if score_diff >= 0:
                    quantity *= min(prob_profit / 2, 2) # Take into account certainty of winning
                else:
                    hedge_factor = max(0.1, 1+score_diff/5)
                    quantity *= hedge_factor # Hedge losses


                # Forced exit if exposed too much
                if score_diff < 0 and prob_side < 0 * prob_profit:
                    side = Side.SELL
                    hedge_factor = min(abs((1/max(prob_profit, 0.1)) * 5), 1)
                    quantity *= max(0.1, min(hedge_factor, 0.9))


                max_imbalance = 5000

                quantity = min(quantity, max_imbalance - abs(exposure))
                quantity = max(quantity, 0)
                
                print(f"PROB_PROFIT: {str(prob_profit)}, PROB_SIDE: {str(prob_side)}")
  
                quantity = max(1, min(int(quantity), max_trades_per_min))  # Clamping to filter noise
                quantity = min(quantity, max_position - current_position)
                quantity = max(quantity, min_position - current_position)
                place_market_order(side=side, ticker=Ticker.TEAM_A, quantity=quantity)



            
            print(f"INFORMED: {event_type} {home_score} - {away_score}, {prob_profit}, {prob_side}")

        if event_type == "END_GAME":
            # IMPORTANT: Highly recommended to call reset_state() when the
            # game ends. See reset_state() for more details.

            place_market_order(side=Side.SELL, ticker=Ticker.TEAM_A, quantity=exposure / market_price)
            self.reset_state()
            return