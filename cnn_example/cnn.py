from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
K.clear_session()

# SimpleCNN outputs a CNN model for demo
# (not yet fine-tuned and optimized, used for time series classification)
def SimpleCNN(input_shape, output_size):
    model = Sequential()
    model.add(Conv2D(filters=16,
                     kernel_size=(5, 5),
                     padding='same',
                     input_shape=input_shape,
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=36,
                     kernel_size=(5, 5),
                     padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_size, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model