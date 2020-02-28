import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


class CNN(object):
    def __init__(self, arr_x, arr_y):
        self.arr_x, self.arr_y = arr_x, arr_y
        self.X_train_image, self.y_train_label, self.X_test_image, self.y_test_label = None, None, None, None
        self. y_trainOneHot, self.y_testOneHot = None, None
        self.input_shape = None
        self.label_shape = None
        self.model = None
        self.train_history = None
        
    def process(self):
        self.X_train_image, self.X_test_image, self.y_train_label, self.y_test_label = train_test_split (self.arr_x, self.arr_y, test_size= 0.3, random_state = 42)
        self. y_trainOneHot, self.y_testOneHot = np_utils.to_categorical(self.y_train_label), np_utils.to_categorical(self.y_test_label)
        self.input_shape = self.X_train_image[0].shape
        self.label_shape = self. y_trainOneHot.shape[1]
        
    def plot_image(self, image):
        fig = plt.gcf()
        fig.set_size_inches(2, 2)
        plt.imshow(image, cmap = 'binary')
        plt.show()
      
    def plot_images_labels_prediction(self, images, labels, prediction, idx, num=10):
        fig = plt.gcf()
        fig.set_size_inches(12, 14)
        if num > 25:
            num = 25
        for i in range(0, num):
            ax = plt.subplot(5, 5, 1 + i)
            ax.imshow(images[idx], cmap='binary')
            title = 'label = ' + str(labels[idx])
            if len(prediction) > 0:
                title += ', predict = ' + str(prediction[idx])
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            idx += 1
        plt.show()
    
    def show_train_history(self, train_history, train, validation):
        plt.figure()
        plt.plot(train_history.history[train])
        plt.plot(train_history.history[validation])
        plt.title('Train History')
        plt.ylabel(train)
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
    
    def build(self):
        self.model = Sequential()
        self.model.add(Conv2D(filters=32,
                         kernel_size=(2, 2),
                         padding='same',
                         input_shape = self.input_shape,
                         activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(filters=32,
                         kernel_size=(2, 2),
                         padding='same',
                         activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.label_shape, activation='softmax'))    
        
        print(self.model.summary())
        
    def train(self, split):
        #adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
        self.train_history = self.model.fit(x = self.X_train_image,
                                       y = self.y_trainOneHot,
                                       validation_split = split,
                                       epochs = 100,
                                       batch_size = 300,
                                       verbose = 2)
        
    def show(self):
        self.show_train_history(self.train_history, 'loss', 'val_loss')
        self.show_train_history(self.train_history, 'accuracy', 'val_accuracy')
        score = self.model.evaluate(self.X_test_image, self.y_testOneHot)
        print('Score of the Testing Data: {}'.format(score))
        prediction = self.model.predict_classes(self.X_test_image)
        #self.plot_images_labels_prediction(self.X_test_image, self.y_test_label, prediction, idx=0)
        #print(pd.crosstab(self.y_test_label, prediction, rownames=['label'], colnames=['predict']))
               
    def save(self, filename):
        self.model.save(filename)
        
        
        
if __name__ == "__main__":
    
    with open('x_arr', 'rb') as handle:
        x_arr = pickle.load(handle)     
    with open('y_arr', 'rb') as handle:
        y_arr = pickle.load(handle)     
    
    with open('load_data_sp500', 'rb') as handle:
        load_data = pickle.load(handle)
    x_arr = load_data['data']
    y_arr = load_data['target']
    
    model = CNN(x_arr, y_arr)
    model.process()
    model.build()
    model.train(0.2)
    model.show()
    
    #model.model.predict_classes(model.X_test_image)














