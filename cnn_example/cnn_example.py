import numpy as np
from keras.utils import np_utils
from cnn import SimpleCNN
from series2gaf import GenerateGAF

# -------------------------------------------------------------------
# Generate GAF:

# create a random sequence with 200 numbers
# all numbers are in the range of 50.0 to 150.0
random_series = np.random.uniform(low=50.0, high=150.0, size=(200,))

# set parameters
timeSeries = list(random_series)
windowSize = 50
rollingLength = 10
fileName = 'demo_%02d_%02d'%(windowSize, rollingLength)

# generate GAF pickle file (output by function Numpy.dump)
GenerateGAF(all_ts = timeSeries,
            window_size = windowSize,
            rolling_length = rollingLength,
            fname = fileName)

            
# -------------------------------------------------------------------
# CNN Example:

# using the generated GAF in previous step
# data shape: (15, 50, 50)
gaf = np.load('%s_gaf.pkl'%fileName)
gaf = np.reshape(gaf, (gaf.shape[0], gaf.shape[1], gaf.shape[2], 1))

# the label is consisted of numbers 1, 2 and 3
# label shape: (15, )
cut_point = int(gaf.shape[0]/3)
label = np.zeros(gaf.shape[0])
label[cut_point:cut_point*2] = 1
label[cut_point*2:] = 2
label = np_utils.to_categorical(label)

# get cnn model ready
# inputs are single channel data: (15, 15, 1)
# output size is 3 because of {1,2,3}-classes
cnn_model = SimpleCNN(input_shape=(gaf.shape[1], gaf.shape[2], 1),
                        output_size=3)

# train the cnn model
train_history = cnn_model.fit(x=gaf, y=label,
                                epochs=10, batch_size=10,
                                validation_split=0.2, verbose=2)

# save trained model
cnn_model.save_model('model_%s.h5'%datetime.strftime(datetime.today(),'%Y%m%d%H%M'))
