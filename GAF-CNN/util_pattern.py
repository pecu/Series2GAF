import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import mpl_finance as mpf


def pattern(df, rule, signal):        
    if signal == 'BearishHarami' or signal == 'BullishHarami':
        last1, last2 = 8, 9
    elif signal == 'MorningStar' or signal == 'EveningStar':
        last1, last2 = 7, 8
    target = df         
    fontsize=12
    plt.rcParams['xtick.labelsize'] = fontsize  
    plt.rcParams['ytick.labelsize'] = fontsize 
    plt.rcParams['axes.titlesize'] = fontsize           
    fig = plt.figure(figsize=(24, 8))
    ax = plt.subplot2grid((1, 1), (0, 0))           
    ax.set_xticks(range(10))
    ax.set_xticklabels(target.index)           
    y = target['close'].iloc[0:last1].values.reshape(-1, 1)
    x = np.array(range(1, last2)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)           
    ax.plot(y_pred, label='Trend')           
    arr = np.c_[range(target.shape[0]), target[['open', 'high', 'low', 'close']].values]
    mpf.candlestick_ohlc(ax, arr, width=0.4, alpha=1, colordown='#53c156', colorup='#ff1717')          
    locs, labels = plt.xticks() 
    plt.setp(labels , rotation = 45)
    plt.grid()
    ax.legend(loc = 'best', prop = {'size': fontsize})
    title_name = signal + '_' + rule
    ax.set_title(title_name)
    fig.subplots_adjust(bottom = 0.25)       
    name = signal + '_' + rule
    plt.savefig(name)        
    plt.show()





