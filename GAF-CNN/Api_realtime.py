import pandas as pd
import numpy as np
import requests
from datetime import datetime
from datetime import timedelta
from pytz import timezone
import time
import pandas_datareader as web


class Api_realtime(object):
    
    def __init__(self, url_his, url_real, his_ls, real_ls):
        self.url_his = url_his
        self.url_real = url_real
        self.his_ls = his_ls
        self.real_ls = real_ls
        self.df_real = None
        self.timezone = timezone('US/Eastern')
        self.data_history = web.data.DataReader('^GSPC', 'yahoo')
        self.data_history.reset_index(inplace=True)
        for i in self.data_history.columns.values:
            self.data_history.rename(columns={i : i.lower()}, inplace=True)
            self.data_history['date'] = self.data_history['date'].astype(str) 
        
        
    def real(self):
        r_real = requests.get(self.url_real)
        packages_json_real = r_real.json()
        df = pd.DataFrame(np.zeros((11, 6)), columns = self.his_ls)
        for i, j in zip(self.real_ls, self.his_ls):
            if i == 'timestamp':
                timestamp = packages_json_real[0][i]
                dt_object = datetime.fromtimestamp(timestamp)
                dt_object2 = dt_object - timedelta(0, 240)
                con1 = (4 <= int(dt_object2.astimezone(self.timezone).strftime('%m')) <= 10)
                con2 = ((int(dt_object2.astimezone(self.timezone).strftime('%m')) == 3) and (int(dt_object.astimezone(self.timezone).strftime('%d')) >= 11))
                con3 = ((int(dt_object2.astimezone(self.timezone).strftime('%m')) == 11) and (int(dt_object.astimezone(self.timezone).strftime('%d')) <= 2))
                if con1 or con2 or con3:
                    t =  dt_object - timedelta(0, 3600)
                else:
                    t = dt_object
                df[j].iloc[-1] = t
            else:
                df[j].iloc[-1] = packages_json_real[0][i]
        r_his = requests.get(self.url_his)
        packages_json_his = r_his.json()
        for i, j in zip(range(0, 10), range(9, -1, -1)):
            for k in self.his_ls:
                if k == 'date':
                    date_str = packages_json_his['historical'][j][k] + ' 16:04:00'
                    dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=self.timezone)
                    timestamp = time.mktime(dt.utctimetuple())
                    utc = datetime.fromtimestamp(timestamp)
                    con1 = (4 <= int(dt.strftime('%m')) <= 10)
                    con2 = ((int(dt.strftime('%m')) == 3) and (int(dt.strftime('%d')) >= 11))
                    con3 = ((int(dt.strftime('%m')) == 11) and (int(dt.strftime('%d')) <= 2))
                    if con1 or con2 or con3:
                        t = utc + timedelta(0, 28800) - timedelta(0, 3600)
                    else:
                        t = utc + timedelta(0, 28800)
                    df[k].iloc[i] = t
                else:
                    df[k].iloc[i] = packages_json_his['historical'][j][k]                  
        self.df_real = df   
    
    def kchart(self, df):
        df['realbody'] = df['close'] - df['open']
    
    
    def process(self, df):
        #df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
        df.set_index('date', inplace=True)
        
        
    def detect_real(self):
        self.real()
        self.process(self.df_real)
        if (timedelta(0, 63000) <= (self.df_real.index[-1] - self.df_real.index[-2]) <= timedelta(0, 86399)):
            self.df_real = self.df_real.iloc[0:10]
            self.kchart(self.df_real)
            period = [self.df_real.index[0], self.df_real.index[9]]
            return (self.df_real, period)
        elif (self.df_real.index[-1] - self.df_real.index[-2]) == timedelta(1, 0):
            self.df_real = self.df_real.iloc[1:11]
            self.kchart(self.df_real)
            period = [self.df_real.index[1], self.df_real.index[10]]
            return (self.df_real, period)
        elif (self.df_real.index[-1] - self.df_real.index[-2]) == timedelta(0, 0):
            self.df_real = self.df_real.iloc[0:10]
            self.kchart(self.df_real)
            period = [self.df_real.index[0], self.df_real.index[9]]
            return (self.df_real, period)   
        else:
            return False      


if __name__ == "__main__":
     # 夏令轉冬令的轉折時間 
     url_his = 'https://financialmodelingprep.com/api/v3/historical-price-full/index/^GSPC'
     url_real = 'https://financialmodelingprep.com/api/v3/quote/^GSPC'
     his_ls = ['date', 'open', 'high', 'low', 'close', 'volume']
     real_ls = ['timestamp', 'open', 'dayHigh', 'dayLow', 'price']
     
     api = Api_realtime(url_his, url_real, his_ls, real_ls)     
     df_real = api.detect_real() 



    
        
        
        
        
