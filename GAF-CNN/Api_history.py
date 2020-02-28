import pandas as pd
import numpy as np
import requests
from datetime import datetime
from datetime import timedelta
from pytz import timezone
import time


class Api_history(object):
    
    def __init__(self, url_his, his_ls, target=None, rule=None):
        self.url_his = url_his
        self.his_ls = his_ls
        self.target = target
        self.rule = rule
        self.df_his = None
        self.timezone = timezone('US/Eastern')
            

    def history(self):
        r = requests.get(self.url_his)
        packages_json = r.json()
        length = len(packages_json['historical'])
        df = pd.DataFrame(np.zeros((length, 6)), columns = self.his_ls)
        for i, j in zip(range(length - 1, -1, -1), range(0, length)):
            for k in self.his_ls:
                if k == 'date':
                    date_str = packages_json['historical'][j][k] + ' 16:04:00'
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
                    df[k].iloc[i] = packages_json['historical'][j][k] 
        self.df_his = df
        
        
    def kchart(self, df):
        df['realbody'] = df['close'] - df['open']
    
    
    def process(self, df):
        #df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
        df.set_index('date', inplace=True)
        
        
    def detect_history(self):
        self.history()
        self.process(self.df_his)
        self.kchart(self.df_his)
        filename = self.target + '_' + self.rule + '_history.csv'
        self.df_his.to_csv(filename)
        return filename
    
    
    def summary_history(self):
        print('target: {}'.format(self.target))
        print('rule: {}'.format(self.rule))
        print('Period: {} - {}'.format(self.df_his.index[0], self.df_his.index[-1]))
        print('shape: {}'.format(self.df_his.shape))
        
        
if __name__ == "__main__":
     # 夏令轉冬令的轉折時間 
     url_his = 'https://financialmodelingprep.com/api/v3/historical-price-full/index/^GSPC'
     his_ls = ['date', 'open', 'high', 'low', 'close', 'volume']
     target = 'S&P500'
     rule = '1D'
     
     api = Api_history(url_his, his_ls, target, rule)
     
     api.detect_history()
     api.summary_history() 
     



    
        
        
        
        
