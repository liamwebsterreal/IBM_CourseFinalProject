import pandas as pd
import try1.model as model

df = pd.read_csv('data/pricedata.csv')
df = df.sort_values('Date')

lookback_window_size = 50
train_df = df[:-720-lookback_window_size]
test_df = df[-720-lookback_window_size:] # 30 days

train_env = model.CustomEnv(train_df, lookback_window_size=lookback_window_size)
test_env = model.CustomEnv(test_df, lookback_window_size=lookback_window_size)

model.Random_games(test_env, visualize=True, train_episodes = 1, training_batch_size=300)