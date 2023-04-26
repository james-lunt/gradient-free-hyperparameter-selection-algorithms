import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.utils import shuffle
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
from keras.callbacks import LambdaCallback
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True

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


param_grid = {
    'batch_size': [64],
    'opt': ["SGD"],
    'lr': [0.001,0.01],
    'beta_1': [0.9],
    'beta_2': [0.9999],
}

def create_model(opt,lr,beta_1,beta_2):
    model = keras.Sequential()
    model.add(Conv2D(16, (3,3), padding='same', input_shape=x_train.shape[1:],activation='relu'))
    model.add(Conv2D(16, (3,3), strides=(2,2), padding='same', activation='relu'))
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3,3), strides=(2,2), padding='same', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax',kernel_regularizer=regularizers.l1(0.0001)))
    if opt == "adam":
        optimizer = Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2)
    else:
        optimizer = SGD(learning_rate=lr)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


model = KerasClassifier(build_fn=create_model)

# Create a GridSearchCV object
grid_search = GridSearchCV(model, param_grid,verbose=2)

# Fit the GridSearchCV object to the training data
grid_search.fit(x_train, y_train,epochs=2)

print("Best hyperparameters: ", grid_search.best_params_)
print("Best validation accuracy: {:.2f}".format(grid_search.best_score_))
print("Test set accuracy: {:.2f}".format(grid_search.score(x_test, y_test)))

# Print the loss history for each iteration
loss_history_list = []
for i in range(len(grid_search.cv_results_['params'])):
    print("Iteration {}: {}".format(i+1, grid_search.cv_results_['params'][i]))
