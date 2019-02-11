### About
- **Series2GAF** can be used to transform time series into _**Gramian Angular Field**_.
![GAF DEMO](https://github.com/RainBoltz/Series2GAF/blob/master/gaf_sample.png "Series and its GAF image")
- There are many useful utility features for financial research in **Series2GAF**, and will be mentioned in the following sections.

# Series2GAF
simple time series encoding package, focused on financial tasks. 

this is an simple example:
```python
import numpy as np
from series2gaf import GenerateGAF

# create a random sequence with 200 numbers
# all numbers are in the range of 50.0 to 150.0
random_series = np.random.uniform(low=50.0, high=150.0, size=(200,))

# set parameters
timeSeries = list(random_series)
windowSize = 50
rollingLength = 10
fileName = 'demo_%02d_%02d'%(windowSize, rollingLength)

# generate GAF pickle file (output by Numpy.dump)
GenerateGAF(all_ts = timeSeries,
            window_size = windowSize,
            rolling_length = rollingLength,
            fname = fileName)
```

now we get a file named _**demo_50_10_gaf.pkl**_ in current directory.
inside the pickle file, you got a grammian angular field with shape (15, 50, 50).
- shape\[0\] refers to _data amount_ : floor((len(timeSeries)-(normalize_window_scaling-1)\*windowSize)/windowSize)
- shape\[1\] refers to _image width_ : windowSize
- shape\[2\] refers to _image height_ : windowSize

Thats It!

if we want to preview the image results, just add these to the code:
```python
from series2gaf import PlotHeatmap

gaf = np.load('%s_gaf.pkl'%fileName)
PlotHeatmap(gaf)
```
we can now find GAF heapmap images in a new child directory _**/output_img**_!

## Functions
```python
def GenerateGAF(all_ts, window_size, rolling_length, fname,
                normalize_window_scaling=1.0, method='summation', scale='[0,1]'):
  ...
  
def PlotHeatmap(all_img, save_dir='output_img'):
  ...
```

## Parameters
- _**all_ts: list**_  
&nbsp;&nbsp;&nbsp;&nbsp;the time series we want to transform.

- _**window_size: int**_  
&nbsp;&nbsp;&nbsp;&nbsp;the sliding window size for transforming sequences into GAF images

- _**rolling_length: int**_  
&nbsp;&nbsp;&nbsp;&nbsp;also known as "stride value" for the sliding window 

- _**fname: str**_  
&nbsp;&nbsp;&nbsp;&nbsp;output file name, the output pickle file will be named as "\[fname]\_gaf.pkl"

- _**normalize_window_scaling: float, optional**_  
&nbsp;&nbsp;&nbsp;&nbsp;_default: 1.0_  
&nbsp;&nbsp;&nbsp;&nbsp;normalize the values in the windows, but considering a ratio of previous values

- _**method: str, optional**_  
&nbsp;&nbsp;&nbsp;&nbsp;_default: 'summation'_  
&nbsp;&nbsp;&nbsp;&nbsp;`'summation'` is for GASF ( calculate cos(x1+x2) )  
&nbsp;&nbsp;&nbsp;&nbsp;`'difference'` if for GADF ( calculate sin(x1-x2) )

- _**scale: str, optoinal**_  
&nbsp;&nbsp;&nbsp;&nbsp;_default: '\[0,1]'_  
&nbsp;&nbsp;&nbsp;&nbsp;`'[0,1]'` means normalize the sequence in the range of 0 and 1  
&nbsp;&nbsp;&nbsp;&nbsp;`'[-1,1]'` means normalize the sequence in the range of -1 and 1

- _**all_img: numpy.array**_  
&nbsp;&nbsp;&nbsp;&nbsp;input GAF multi-dimension array

- _**save_dir: str, optional**_  
&nbsp;&nbsp;&nbsp;&nbsp;_default: 'output_img'_  
&nbsp;&nbsp;&nbsp;&nbsp;directory for output images  

------


