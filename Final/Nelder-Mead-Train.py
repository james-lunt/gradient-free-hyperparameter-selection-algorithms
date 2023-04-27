from NelderMead import nelder_mead

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
import keras
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import json
import numpy as np

plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True

# set display options to print full dataframe
pd.set_option('display.max_rows', None) 
pd.set_option('display.max_columns', None) 
pd.set_option('display.width', None) 

# Model / data parameters
num_classes = 10
input_shape = (32, 32, 3)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
n=5000
x_train = x_train[1:n]; y_train=y_train[1:n]
#x_test=x_test[1:500]; y_test=y_test[1:500]

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
print("orig x_train shape:", x_train.shape)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

accuracies=[]

def run_model(params):

    model = keras.Sequential()
    model.add(Conv2D(16, (3,3), padding='same', input_shape=x_train.shape[1:],activation='relu'))
    model.add(Conv2D(16, (3,3), strides=(2,2), padding='same', activation='relu'))
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3,3), strides=(2,2), padding='same', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax',kernel_regularizer=regularizers.l1(0.0001)))
    optimizer = Adam(learning_rate=params[0], beta_1=params[1], beta_2=params[2])
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    history = model.fit(x_train, y_train, batch_size=params[3].astype(np.int32), epochs=5, validation_split=0.1)
    _,test_accuracy = model.evaluate(x_test,y_test)
    #Takes the training accuracy of the last epoch in the iteration
    train_accuracy = history.history['accuracy'][-1]
    accuracies.append(train_accuracy)
    return train_accuracy

initial_guess = np.array([1e-4,0.8,0.666,128])

print(nelder_mead(run_model,np.array([0.001,0.8,0.666,128]),max_iter=5))
print(accuracies)

