import pandas as pd
import numpy as np


def ts2gasf(ts, max_v, min_v):
    '''
    Args:
        ts (numpy): (N, )
        max_v (int): max value for normalization
        min_v (int): min value for normalization

    Returns:
        gaf_m (numpy): (N, N)
    '''
    # Normalization : 0 ~ 1
    if max_v == min_v:
        gaf_m = np.zeros((len(ts), len(ts)))
    else:
        ts_nor = np.array((ts-min_v) / (max_v-min_v))
        # Arccos
        ts_nor_arc = np.arccos(ts_nor)
        # GAF
        gaf_m = np.zeros((len(ts_nor), len(ts_nor)))
        for r in range(len(ts_nor)):
            for c in range(len(ts_nor)):
                gaf_m[r, c] = np.cos(ts_nor_arc[r] + ts_nor_arc[c])
    return gaf_m


def get_gasf(arr):
    '''Convert time-series to gasf    
    Args:
        arr (numpy): (N, ts_n, 4)

    Returns:
        gasf (numpy): (N, ts_n, ts_n, 4)

    Todos:
        add normalization together version
    '''
    arr = arr.copy()
    gasf = np.zeros((arr.shape[0], arr.shape[1], arr.shape[1], arr.shape[2]))
    for i in range(arr.shape[0]):
        for c in range(arr.shape[2]):
            each_channel = arr[i, :, c]
            c_max = np.amax(each_channel)
            c_min = np.amin(each_channel)
            each_gasf = ts2gasf(each_channel, max_v=c_max, min_v=c_min)
            gasf[i, :, :, c] = each_gasf
    return gasf


def get_arr(data, signal, d=None):
    if signal != 'n':
        df_es = data.loc[data[signal]==1]
    else:
        df_es = d
    arr = np.zeros((df_es.shape[0], 10, 4))
    for index, N in zip(df_es.index, range(df_es.shape[0])):
        df = data.loc[data.index <= index][-10::]
        arr[N, :, 0] = df['open']
        arr[N, :, 1] = df['high']
        arr[N, :, 2] = df['low']
        arr[N, :, 3] = df['close']
    return arr
    
    
def process(file):
    data = pd.read_csv(file)
    data['date'] = pd.to_datetime(data['date'], format="%Y-%m-%d %H:%M:%S.%f")
    data.set_index('date', inplace=True)
    return data


def detect(data, signal, d=None):
    arr = get_arr(data, signal, d)
    gasf = get_gasf(arr)
    return gasf





