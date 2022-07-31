import gym
import pandas as pd
import numpy as np
from gym import spaces
from sklearn import preprocessing
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, A2C
from stable_baselines3.sac.policies import MlpPolicy
from utils import TradingGraph

import warnings
warnings.filterwarnings('ignore')

MAX_TRADING_SESSION = 4301


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human', 'bot', 'none']}
    scaler = preprocessing.MinMaxScaler()
    viewer = None
    account_history_cols = ['net_worth', 'cash_held', 'asset_held', 'asset_bought', 'cost', 'asset_sold', 'sale']
    trade_cols = ['step', 'amount', 'total', 'type']

    def __init__(self, df, asset, lookback_window_size=50, commission=0.00075, initial_balance=10000, serial=False):
        super(TradingEnv, self).__init__()
        self.df = df.dropna().reset_index()
        self.lookback_window_size = lookback_window_size
        self.initial_balance = initial_balance
        self.commission = commission
        self.serial = serial
        self.asset = asset

        # actions space 0: buy 100%, 1: hold, 2, sell 100%
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, self.lookback_window_size + 1), dtype=np.float16)

    def reset(self):
        self.net_worth = self.initial_balance
        self.cash_held = self.initial_balance
        self.asset_held = 0  
        self.current_step = 0
        self.rewards = [0]
        done = False

        self._reset_session()
        
        net_worth_array = [self.net_worth] * (self.lookback_window_size + 1)
        cash_held_array = [self.cash_held] * (self.lookback_window_size + 1)
        zero_array = [0] * (self.lookback_window_size + 1)

        self.account_history = pd.DataFrame(np.transpose([net_worth_array, cash_held_array, zero_array, zero_array,zero_array, zero_array, zero_array]), columns=self.account_history_cols)

        self.trades = pd.DataFrame(columns= self.trade_cols)
        obs = self._next_observation()

        return obs['net_worth'], obs['High'], obs['Volume']

    def render(self, mode='bot'):
        if mode == 'human':
            if self.viewer is None:
                self.viewer = TradingGraph(self.df)
            self.viewer.render(self.current_step, self.trades, self.account_history)
        elif mode == 'bot':
            print('Current Step: ' + str(self.current_step))
            print('Net Worth: ' + str(self.net_worths[self.current_step]) + ' USD')
            print('Cash Balance: ' + str(self.cash_held) + ' USD')
            print('Asset Balance: ' + str(self.asset_held))

    def step(self, action):
        self.steps_left -= 1
        self.current_step += 1
        prev_net_worth = self.net_worth
        current_price = self._get_current_price() + 0.01
        self._take_action(action, current_price)

        if self.steps_left == 0:
            self.cash_held += self.asset_held * current_price
            self.asset_held = 0
            self._reset_session()
        
        obs = self._next_observation()
        reward = self.net_worth - prev_net_worth
        done = self.net_worth <= 0.1 * self.initial_balance

        return obs['net_worth'], obs['High'], obs['Volume']

    def _reset_session(self):

        if self.serial:
            self.steps_left = len(self.df) - self.lookback_window_size - 1
            self.frame_start = self.lookback_window_size
        else:
            self.steps_left = np.random.randint(1, MAX_TRADING_SESSION)
            self.frame_start = np.random.randint(
                self.lookback_window_size, len(self.df) - self.steps_left)

        self.active_df = self.df[self.frame_start -
                                 self.lookback_window_size:self.frame_start + self.steps_left]

    def _next_observation(self):
        end = self.current_step + self.lookback_window_size + 1

        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        obs = pd.DataFrame(self.active_df[cols].iloc[self.current_step:end], columns= cols)

        scaled_history = self.scaler.fit_transform(self.account_history)

        obs = pd.concat([self.account_history.iloc[self.current_step:end], obs], axis=1)
    
        return obs

    def _take_action(self, action, current_price):
        action_type = action

        asset_bought = 0
        asset_sold = 0
        cost = 0
        sale = 0

        if action < 1: 
            cost = self.cash_held
            asset_bought = (self.cash_held / current_price  * ( 1 - self.commission)) 
            self.asset_held += asset_bought
            self.cash_held = 0
        elif action > 1:
            sale = self.asset_held * current_price * (1 - self.commission)
            asset_sold = self.asset_held
            self.asset_held = 0
            self.cash_held += sale

        if asset_sold > 0 or asset_bought > 0:
            temp = pd.DataFrame([{
                'step': self.current_step + self.lookback_window_size,
                'amount':  asset_sold if asset_sold > 0 else asset_bought,
                'total': sale if asset_sold > 0 else cost,
                'type': "sell" if asset_sold > 0 else "buy",
            }])
            self.trades = self.trades.append(temp, ignore_index=True)
        
        self.net_worth = self.cash_held + self.asset_held * current_price
        temp = pd.DataFrame([{
            'net_worth': self.net_worth,
            'cash_held': self.cash_held,
            'asset_held': self.asset_held, 
            'asset_bought': asset_bought,
            'cost': cost,
            'asset_sold': asset_sold,
            'sale': sale,
        }])
        self.account_history = self.account_history.append(temp, ignore_index=True)

    def _get_current_price(self):
        low = self.df.loc[self.current_step + self.frame_start,'Low']
        high = self.df.loc[self.frame_start + self.current_step, 'High']
        return (low + high) / 2





if __name__ == "__main__":
    filepath = 'data\BTC-USD_hourly2020.csv'
    data = pd.read_csv(filepath)
    env = TradingEnv(data,'BTC', serial=True)
    slice_point = int(len(data) - 800)
    train_df = data[:slice_point]
    test_df = data[slice_point:]
    train_env = DummyVecEnv([lambda: TradingEnv(train_df, asset='BTC', commission=0, serial=True,)])
    test_env = DummyVecEnv([lambda: TradingEnv(test_df, asset='BTC', commission=0, serial=True)])
    model = A2C(MlpPolicy,
             train_env,
             learning_rate=0.01,
             verbose=1, 
             tensorboard_log="./tensorboard/")
    model.learn(total_timesteps=100)