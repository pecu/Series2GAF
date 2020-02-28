from Api_history import Api_history
from Signal2 import Signal
import util_gasf
from Cnn import CNN
from Api_realtime import Api_realtime
import util_pattern
import numpy as np
import pandas as pd
import pickle
from sklearn.utils import shuffle
from keras.models import load_model


class Main(object):
    
    def __init__(self, target, rule, url_his, url_real, his_ls, real_ls, signal_ls, save_plot):
        self.target = target
        self.rule = rule
        self.url_his = url_his
        self.url_real = url_real
        self.his_ls = his_ls
        self.real_ls = real_ls
        self.signal_ls = signal_ls
        self.save_plot = save_plot
        self.data = None
        self.data_pattern = None
        self.gasf_arr = None
        self.load_data = None
        self.load_model = None
        self.pattern_dict = dict()
        for i, j in zip(signal_ls, range(len(signal_ls))):
            self.pattern_dict[j] = i
        self.pattern_dict[len(signal_ls)] = 'No Pattern'
        self.package_realtime = None
        
    def api_history(self):
        api_his = Api_history(self.url_his, self.his_ls, self.target, self.rule)
        self.data = api_his.detect_history()
        api_his.summary_history()
        
    def rule_based(self):
        Sig = Signal(self.data, self.signal_ls, self.save_plot)
        Sig.process()
        self.data_pattern = Sig.detect_all()
        Sig.summary()
        
    def gasf(self):
        data_1D_pattern = pd.read_csv(self.data_pattern)
        gasf_arr = np.zeros((len(self.signal_ls) + 1, 30, 10, 10, 4))
        for i, j in zip(self.signal_ls, range(len(self.signal_ls))):
            gasf = util_gasf.detect(data_1D_pattern, i)
            gasf_arr[j, :, :, :, :] = gasf[0:30, :, :, :]
        df = data_1D_pattern.copy()
        for i in self.signal_ls:
            df = df.loc[df[i] != 1]
        df = shuffle(df[9::])
        gasf = util_gasf.detect(data_1D_pattern, 'n', df)
        gasf_arr[-1, :, :, :, :] = gasf[0:30, :, :, :]
        self.gasf_arr = 'gasf_arr_' + self.target
        with open(self.gasf_arr, 'wb') as handle:    
            pickle.dump(gasf_arr, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def process_xy(self):
        with open(self.gasf_arr, 'rb') as handle:
            gasf_arr = pickle.load(handle)
        x_arr = np.zeros(((len(self.signal_ls) + 1), 30, 10, 10, 4))
        for i in range(len(self.signal_ls) + 1):
            x_arr[i, :, :, :, :] = gasf_arr[i, 0:30, :, :, :]    
        x_arr = gasf_arr.reshape((len(self.signal_ls) + 1) * 30, 10, 10, 4)
        y_arr = []
        for i in range(len(self.signal_ls) + 1):
            ls = [i] * 30
            y_arr.extend(ls)
        y_arr = np.array(y_arr)
        load_data = {'data': x_arr, 'target': y_arr}
        self.load_data = 'load_data_' + self.target
        with open(self.load_data, 'wb') as handle:
            pickle.dump(load_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def cnn(self):
        with open(self.load_data, 'rb') as handle:
            load_data = pickle.load(handle)
        x_arr = load_data['data']
        y_arr = load_data['target']
        model = CNN(x_arr, y_arr)
        model.process()
        model.build()
        model.train(0.2)
        model.show()
        self.load_model = 'CNN_' + self.target + '.h5'
        model.save(self.load_model)
        
    def api_realtime(self):
        api_real = Api_realtime(self.url_his, self.url_real, self.his_ls, self.real_ls) 
        self.package_realtime = api_real.detect_real() 
        
    def predict_realtime(self):
        model = load_model(self.load_model)
        if self.package_realtime != False:
            df_real, period = self.package_realtime[0], self.package_realtime[1]
            df = shuffle(df_real.iloc[9::])
            gasf = util_gasf.detect(df_real, 'n', df)
            x_realtime_arr = gasf
            y_pred_realtime = model.model.predict_classes(x_realtime_arr)
            pattern = self.pattern_dict[y_pred_realtime[0]]
            print('Target: {}'.format(self.target))
            print('Time Rule: {}'.format(self.rule))
            print('Time Period: {} - {}'.format(period[0], period[1]))
            print('The Pattern of the Realtime Data: {}'.format(pattern))
            util_pattern(df_real, self.rule, pattern)
            return (period, pattern)
        else:
            print('Not in the transaction time')
            return False
    
    
if __name__ == "__main__":
     # 夏令轉冬令的轉折時間 
     target = 'S&P500'
     rule = '1D'
     url_his = 'https://financialmodelingprep.com/api/v3/historical-price-full/index/^GSPC'
     url_real = 'https://financialmodelingprep.com/api/v3/quote/^GSPC'
     his_ls = ['date', 'open', 'high', 'low', 'close', 'volume']
     real_ls = ['timestamp', 'open', 'dayHigh', 'dayLow', 'price']
     signal_ls = ['MorningStar', 'EveningStar', 'BearishHarami', 'BullishHarami']
     save_plot = False
     
     main = Main(target, rule, url_his, url_real, his_ls, real_ls, signal_ls, save_plot)
     main.api_history()
     main.rule_based()
     main.gasf()
     main.process_xy()
     main.cnn()
     main.api_realtime()
     main.predict_realtime()
     
     
    
     
     
     
     
     
     
        
        
        
        
        
        
        
        
        
        
        
        
        







