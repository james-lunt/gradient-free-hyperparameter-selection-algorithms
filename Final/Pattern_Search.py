import itertools
import math
import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import json
import numpy as np

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

def make_grid(batch_size,learning_rates,beta_1,beta_2):
    # Create a grid of permutations
    grid = list(itertools.product(batch_size, learning_rates, beta_1, beta_2))

    # Determine the size of the square grid
    grid_length = len(grid)
    grid_size = math.isqrt(grid_length)
    if grid_size * grid_size < grid_length:
        grid_size += 1

    # Pad the grid with None values to make it a square shape
    padding_length = grid_size * grid_size - grid_length
    grid += [None] * padding_length

    # Reshape the grid into a square 2-dimensional array
    parsed_grid = [grid[i:i+grid_size] for i in range(0, len(grid), grid_size)]

    print(parsed_grid)
    print("Number of rows:", grid_size)
    print("Number of columns:", grid_size)

    return parsed_grid,grid_size

    """# Write the parsed grid to a CSV file
    filename = "parsed_grid.csv"
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(parsed_grid)

    print(f"CSV file '{filename}' has been created.")"""

def cnn_run(parameter_vector,report_metric):
    batch_size,lr,beta_1,beta_2 = parameter_vector
    model = keras.Sequential()
    model.add(Conv2D(16, (3,3), padding='same', input_shape=x_train.shape[1:],activation='relu'))
    model.add(Conv2D(16, (3,3), strides=(2,2), padding='same', activation='relu'))
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3,3), strides=(2,2), padding='same', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax',kernel_regularizer=regularizers.l1(0.0001)))
    adam_optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2)
    model.compile(loss="categorical_crossentropy", optimizer=adam_optimizer, metrics=["accuracy"])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=5, validation_split=0.1)
    loss,test_accuracy = model.evaluate(x_test,y_test)
    #Takes the training accuracy of the last epoch in the iteration
    train_accuracy = history.history['accuracy'][-1]
    if report_metric == 'loss':
        return loss
    if report_metric == 'test_accuracy':
        return test_accuracy
    else:
        return train_accuracy

#Alpha is the step size, gamma is shrink/grow size    
def pattern_search_train(grid,grid_size,starting_point, alpha, gamma, num_iter,report_metric):
    #Initialise
    coord = starting_point
    best_score = 0
    iteration_accuracies = []

    for i in range(num_iter):
        print("New center coord:")
        print(coord) 
        #If best parameters is the center point, Grow
        #If best parameters is a different point, Change center point to different point
        #If no improvement, Shrink
        eval_score_center = cnn_run(grid[coord[0]][coord[1]], report_metric) 
        eval_score_left = cnn_run(grid[coord[0]-alpha][coord[1]], report_metric) 
        eval_score_right = cnn_run(grid[coord[0]+alpha][coord[1]], report_metric) 
        eval_score_down = cnn_run(grid[coord[0]][coord[1]-alpha], report_metric) 
        eval_score_up = cnn_run(grid[coord[0]][coord[1]+alpha], report_metric) 

        scores = [eval_score_center,eval_score_left,eval_score_right,eval_score_down,eval_score_up]
        itr_best_score = max(scores)
        iteration_accuracies.append(itr_best_score)

        #Move
        #Make sure not greater or less than size of grid:
        if (coord[0] < grid_size+alpha) and (coord[1] < grid_size+alpha):  
            if itr_best_score == eval_score_left:
                coord[0] -= alpha
            if itr_best_score == eval_score_right:
                coord[0] += alpha
            if itr_best_score == eval_score_up:
                coord[1] += alpha
            if itr_best_score == eval_score_down:
                coord[1] -= alpha


        #Grow
        if itr_best_score > best_score:
            best_score = itr_best_score
            alpha *= gamma
            print("Expand")
        #Shrink
        else:
            alpha /= gamma
            print("Shrink")

    #print(grid[coord[0]][coord[1]])
    #print(best_score)
    return coord, grid[coord[0]][coord[1]], best_score, iteration_accuracies
    

batch_size = [8, 16, 64, 128]
learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]
beta_1 = [0.3, 0.6, 0.9]
beta_2 = [0.333, 0.666, 0.999]

grid,grid_size = make_grid(batch_size,learning_rates,beta_1,beta_2)
coord, hyperparameters, best_score, iteration_scores = pattern_search_train(
    grid,grid_size,
    [int(grid_size/2),
     int(grid_size/2)],
     1,2,2,
     'test_accuracy')

print(coord)
print(hyperparameters)
print(best_score)
print(iteration_scores)

