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
import time

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
    with tf.device('/GPU:1'):
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
def pattern_search_train(grid,grid_size,starting_point, alpha, gamma, num_iter,report_metric, stop_thr):
    #Initialise
    coord = starting_point
    best_score = 0
    iteration_accuracies = []
    shrink_count = 0

    for i in range(num_iter):
        if shrink_count > stop_thr:
            break

        print("New center coord:")
        print(coord) 
        print("alpha")
        print(alpha)
        #If best parameters is the center point, Grow
        #If best parameters is a different point, Change center point to different point
        #If no improvement, Shrink

        itr_best_score = 0
        eval_score_center = cnn_run(grid[coord[0]][coord[1]], report_metric) 
        if eval_score_center > itr_best_score:
            itr_best_score = eval_score_center
            move = 'stay'
        if(coord[0]-alpha > 0):
            eval_score_left = cnn_run(grid[coord[0]-alpha][coord[1]], report_metric)
            if eval_score_left > itr_best_score:
                itr_best_score = eval_score_left
                move = 'left'
        if (coord[0]+alpha < grid_size):
            eval_score_right = cnn_run(grid[coord[0]+alpha][coord[1]], report_metric)
            if eval_score_right > itr_best_score:
                itr_best_score = eval_score_right
                move = 'right'
        if(coord[1]-alpha > 0): 
            eval_score_down = cnn_run(grid[coord[0]][coord[1]-alpha], report_metric) 
            if eval_score_down > itr_best_score:
                itr_best_score = eval_score_down
                move = 'down'
        if(coord[1]+alpha < grid_size):
            eval_score_up = cnn_run(grid[coord[0]][coord[1]+alpha], report_metric)
            if eval_score_up > itr_best_score:
                itr_best_score = eval_score_up
                move = 'up'

        iteration_accuracies.append(itr_best_score)

        #Move
        #Make sure not greater or less than size of grid:
        if move == 'left':
            coord[0] -= alpha
            print("Move left")
        elif move == 'right':
            coord[0] += alpha
            print("Move right")
        elif move == 'up':
            coord[1] += alpha
            print("Move up")
        elif move == 'down':
            coord[1] -= alpha
            print("Move down")

        else:
        #Grow
            if itr_best_score > best_score:
                best_score = itr_best_score
                alpha += gamma
                print("Expand")
            #Shrink
            else:
                alpha -= gamma
                if alpha == 0:
                    alpha = 1
                    shrink_count+=1
                else:
                    shrink_count=0
                #What if has shrinked to the max?
                print("Shrink")

        if itr_best_score > best_score:
           best_score = itr_best_score

    #print(grid[coord[0]][coord[1]])
    #print(best_score)
    return coord, grid[coord[0]][coord[1]], best_score, iteration_accuracies
    

batch_size = [2,4,6,8,12,16,20,24,28,32]
learning_rates = [0.004,0.0008,0.001,0.0012,0.0016]
beta_1 = [0.3,0.4,0.5,0.6,0.7]
beta_2 = [0.7, 0.742, 0.784, 0.826, 0.867, 0.908, 0.95, 0.999]

i = 1
while i < 3:
    grid,grid_size = make_grid(batch_size,learning_rates,beta_1,beta_2)
    start_time = time.time()
    coord, hyperparameters, best_score, iteration_scores = pattern_search_train(
        grid,grid_size,
        [23,22],
        alpha=12,
        gamma=3,
        num_iter=20,
        report_metric='test_accuracy',
        stop_thr=3
        )
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)


    with open(f'ps_data_{i}.txt', 'w') as file:
        file.write("Coords: " + str(coord)+ '\n')
        file.write("Hyperparameters: " + str(hyperparameters) + '\n')
        file.write("Best Score: " + str(best_score) + '\n')
        file.write("Iteration Scores: " + str(iteration_scores) + '\n')
        file.write(f"Elapsed time: {str(hours)} hr {str(minutes)} min {str(seconds)} sec")

    i+=1



