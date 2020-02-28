import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import mpl_finance as mpf


class Signal(object):
    def __init__(self, filename, detect_ls, save_plot):
        self.data = pd.read_csv(filename)
        self.detect_ls = detect_ls
        self.time_period = None
        self.save_plot = save_plot

    def trend(self, series):
        y = series.values.reshape(-1,1)
        x = np.array(range(1, series.shape[0] + 1)).reshape(-1,1)
        model = LinearRegression()
        model.fit(x, y)
        slope = model.coef_
        if slope > 0:
            return 1
        elif slope == 0:
            return 0
        else:
            return -1
        
    def dataframe_roll_evening(self, df):       
        def EveningStarSignal(window_series):            
            window_df = df.loc[window_series.index]           
            trend = window_df['trend1'].iloc[-4]
            body1 = window_df['realbody'].iloc[-3]
            body2 = window_df['realbody'].iloc[-2]
            body3 = window_df['realbody'].iloc[-1]
            half1 = window_df['open'].iloc[-3] + (body1 * (1 / 2))
            half2 = window_df['open'].iloc[-3] + (body1 * (4 / 5))
            close3 = window_df['close'].iloc[-1]
            open2 = window_df['open'].iloc[-2]                    
            
            cond1 = (trend == 1) and (body1 > 0) and (body2 > 0) and (body3 < 0)
            cond2 = (open2 > half1)
            cond3 = (close3 < half2)
            
            if cond1 and cond2 and cond3:
                return 1  
            else:
                return 0            
        return EveningStarSignal
    
    def dataframe_roll_morning(self, df):    
        def MorningStarSignal(window_series):           
            window_df = df.loc[window_series.index]           
            trend = window_df['trend1'].iloc[-4]
            body1 = window_df['realbody'].iloc[-3]
            body2 = window_df['realbody'].iloc[-2]
            body3 = window_df['realbody'].iloc[-1]
            half1 = window_df['open'].iloc[-3] + (body1 * (4 / 5))
            half2 = window_df['open'].iloc[-3] + (body1 * (1 / 2))
            close3 = window_df['close'].iloc[-1]
            open2 = window_df['open'].iloc[-2]
            #percentile1 = stats.percentileofscore(abs(window_df['realbody']), abs(window_df['realbody'].iloc[-3]), kind='strict')
            
            cond1 = (trend == -1) and (body1 < 0) and (body2 > 0) and (body3 > 0)
            cond2 = (close3 >= half1)
            cond3 = (open2 <= half2)
            #cond4 = (percentile1 > 60)
                         
            if cond1 and cond2 and cond3:
                return 1 
            else:
                return 0          
        return MorningStarSignal
    
    def dataframe_roll_bear(self, df):   
        def BearishHaramiSignal(window_series):       
            window_df = df.loc[window_series.index]       
            trend = window_df['trend2'].iloc[-3]
            body1 = window_df['realbody'].iloc[-2]
            body2 = window_df['realbody'].iloc[-1]
            open1 = window_df['open'].iloc[-2]
            open2 = window_df['open'].iloc[-1]
            close1 = window_df['close'].iloc[-2]
            close2 = window_df['close'].iloc[-1] 
            
            cond1 = (trend == 1) and (body1 > 0) and (body2 < 0)
            cond2 = (open2 < close1)
            cond3 = (close2 > open1)
            
            if cond1 and cond2 and cond3:
                return 1 
            else:
                return 0        
        return BearishHaramiSignal
    
    def dataframe_roll_bull(self, df):   
        def BullishHaramiSignal(window_series):       
            window_df = df.loc[window_series.index]       
            trend = window_df['trend2'].iloc[-3]
            body1 = window_df['realbody'].iloc[-2]
            body2 = window_df['realbody'].iloc[-1]
            open1 = window_df['open'].iloc[-2]
            open2 = window_df['open'].iloc[-1]
            close1 = window_df['close'].iloc[-2]
            close2 = window_df['close'].iloc[-1] 
            
            cond1 = (trend == -1) and (body1 < 0) and (body2 > 0)
            cond2 = (open2 > close1)
            cond3 = (close2 < open1)
            
            if cond1 and cond2 and cond3:
                return 1 
            else:
                return 0        
        return BullishHaramiSignal
        
    def pattern(self, df, rule, signal): 
        if signal == 'BearishHarami' or signal == 'BullishHarami':
            last1, last2 = 8, 9
        elif signal == 'MorningStar' or signal == 'EveningStar':
            last1, last2 = 7, 8
            
        t_ls = df.loc[df[signal] == 1].index        
        for i, j in zip(t_ls, range(1, len(t_ls) + 1)):
            target = df.loc[df.index <= i].iloc[-10:]              
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
            name = signal + '_' + rule + '_' + str(j)
            plt.savefig(name)        
            plt.close()
        
    def process(self):
        self.data['date'] = pd.to_datetime(self.data['date'], format="%Y-%m-%d %H:%M:%S.%f")
        self.data.set_index('date', inplace=True)
        if (self.data.index[1] - self.data.index[0]).seconds == 60:
            self.time_period = '1m'
        elif (self.data.index[1] - self.data.index[0]).seconds == 1800:
            self.time_period = '30m'
        elif (self.data.index[1] - self.data.index[0]).seconds == 3600:
            self.time_period = '1H'
        elif (self.data.index[1] - self.data.index[0]).days == 1:
            self.time_period = '1D'
        elif (self.data.index[1] - self.data.index[0]).days == 7:
            self.time_period = '1W'
        self.data['trend1'] = self.data['close'].rolling(7).apply(self.trend, raw=False)
        self.data['trend2'] = self.data['close'].rolling(8).apply(self.trend, raw=False)
        
    def detect_all(self):
        for signal in self.detect_ls:
            if signal == 'MorningStar':
                self.data['MorningStar'] = self.data['close'].rolling(4).apply(self.dataframe_roll_morning(self.data), raw=False)
                if self.save_plot == True: 
                    self.pattern(self.data, self.time_period, signal)
            
            elif signal == 'EveningStar':
                self.data['EveningStar'] = self.data['close'].rolling(4).apply(self.dataframe_roll_evening(self.data), raw=False)
                if self.save_plot == True:
                    self.pattern(self.data, self.time_period, signal)
            
            elif signal == 'BearishHarami':
                self.data['BearishHarami'] = self.data['close'].rolling(3).apply(self.dataframe_roll_bear(self.data), raw=False)
                if self.save_plot == True:
                    self.pattern(self.data, self.time_period, signal)
                
            elif signal == 'BullishHarami':
                self.data['BullishHarami'] = self.data['close'].rolling(3).apply(self.dataframe_roll_bull(self.data), raw=False)
                if self.save_plot == True:
                    self.pattern(self.data, self.time_period, signal)
        
        file_name = 'S&P500_' + self.time_period + '_pattern.csv'
        self.data.to_csv(file_name, index=False)
        return file_name
        
    def summary(self):
        period = self.data.index[[1, -1]]
        print('Rule : %s' % (self.time_period))
        print('Period : %s - %s' % (period[0], period[1]), '\n')
        total = self.data.shape[0]
        num = None
        for i in self.detect_ls:
            num = np.sum(self.data[i])
            print('Number of', i, ': %s // %s' % (num, total), '\n')
        
        
if __name__ == "__main__":
    detect_ls = ['MorningStar', 'EveningStar', 'BearishHarami', 'BullishHarami']
    data_1H = 'S&P500_1D_history.csv'
    save_plot = True
    
    S_1H = Signal(data_1H, detect_ls, save_plot)
    S_1H.process()
    S_1H.detect_all()
    S_1H.summary()

    





   