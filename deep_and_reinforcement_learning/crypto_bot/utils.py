from cv2 import line
from numpy import block
import pandas as pd
import matplotlib.pyplot as plt



class TradingGraph:

    def __init__(self, df):
        self.df = df
        self.fig, self.ax = fig, ax = plt.subplots()
        

    def render(self, current_step, trades, account_history, window_size=50):
        plt.plot(account_history['net_worth'], 'b')
        plt.ylim(5000,12000)
        plt.xlabel('Time')
        plt.ylabel('Net Worth(USD)')
        plt.title('Net Worth over Time')
        # Necessary to view frames before they are unrendered
        plt.pause(0.1)


    def close(self):
        plt.close()
    
