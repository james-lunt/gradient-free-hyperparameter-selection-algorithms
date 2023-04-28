#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Convolution2D
from keras import regularizers
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator

from ray import air, tune
from ray.tune import Trainable
from ray.tune.schedulers import PopulationBasedTraining

num_classes = 10
NUM_SAMPLES = 5000


class Cifar10Model(Trainable):
    def _read_data(self):
        # The data, split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        n=5000
        x_train = x_train[1:n]; y_train=y_train[1:n]
        #x_test=x_test[1:500]; y_test=y_test[1:500]

        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        print("orig x_train shape:", x_train.shape)

        # convert class vectors to binary class matrices
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)


        return (x_train, y_train), (x_test, y_test)

    def _build_model(self, input_shape):
        x = Input(shape=(32, 32, 3))
        y = x
        y = Convolution2D(
            filters=16,
            kernel_size=(3,3),
            padding="same",
            activation="relu",
        )(y)
        y = Convolution2D(
            filters=16,
            kernel_size=(3,3),
            strides = (2,2),
            padding="same",
            activation="relu",
        )(y)
        y = Convolution2D(
            filters=32,
            kernel_size=(3,3),
            padding="same",
            activation="relu",
        )(y)
        y = Convolution2D(
            filters=32,
            kernel_size=(3,3),
            strides=(2,2),
            padding="same",
            activation="relu",
        )(y)
        y = Dropout(0.5)(y)
        y = Flatten()(y)
        y = Dense(units=num_classes, activation="softmax", kernel_regularizer=regularizers.l1(0.0001))(y)

        model = Model(inputs=x, outputs=y, name="model1")
        return model

    def setup(self, config):
        self.train_data, self.test_data = self._read_data()
        x_train = self.train_data[0]
        model = self._build_model(x_train.shape[1:])
        opt = tf.keras.optimizers.Adam(
            #lr=self.config.get("lr",0.0001), beta_1=self.config.get("b1", 0.9), beta_2=self.config.get("b2",0.999)
            lr=self.config["lr"], beta_1=self.config["b1"], beta_2=self.config["b2"]
        )

        model.compile(
            loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
        )
        self.model = model

    def step(self):
        x_train, y_train = self.train_data
        x_train, y_train = x_train[:NUM_SAMPLES], y_train[:NUM_SAMPLES]
        x_test, y_test = self.test_data
        x_test, y_test = x_test[:NUM_SAMPLES], y_test[:NUM_SAMPLES]

        #batch_size = self.config.get("batch_size", 64)
        batch_size = self.config["batch_size"]
        epochs = self.config.get("epochs", 1)

        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=None)

        # loss, accuracy
        _, accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        return {"mean_accuracy": accuracy}

    def save_checkpoint(self, checkpoint_dir):
        file_path = checkpoint_dir + "/model"
        self.model.save(file_path)
        return file_path

    def load_checkpoint(self, path):
        # See https://stackoverflow.com/a/42763323
        del self.model
        self.model = load_model(path)

    def cleanup(self):
        # If need, save your model when exit.
        # saved_path = self.model.save(self.logdir)
        # print("save model at: ", saved_path)
        pass


if __name__ == "__main__":
    #Best hyperparameters: {"batch_size": 16, "beta_1": 0.727732325422385, "beta_2": 0.6829935115476067, "lr": 0.001, "opt": "adam"}
    space = {
        "epochs": 1,
        "lr": tune.uniform(0.0004,0.0016),
        "b1": tune.uniform(0.3, 0.7),
        "b2": tune.uniform(0.7,0.999),
        "batch_size": tune.randint(2,32)
    }
    perturbation_interval = 4
    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=perturbation_interval,
        hyperparam_mutations={
            "lr": tune.uniform(0.0008,0.0012),
            "b1": tune.uniform(0.1, 0.9),
            "b2": tune.uniform(0.6,0.76),
            "batch_size": tune.randint(1,32)
        },
    )

    tuner = tune.Tuner(
        tune.with_resources(
            Cifar10Model,
            resources={"cpu": 1, "gpu": 1},
        ),
        run_config=air.RunConfig(
            name="pbt_cifar10",
            stop={
                "mean_accuracy": 0.65,
                "training_iteration": 20,
            },
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=perturbation_interval,
                checkpoint_score_attribute="mean_accuracy",
                num_to_keep=2,
            ),
        ),
        tune_config=tune.TuneConfig(
            scheduler=pbt,
            num_samples=10,
            metric="mean_accuracy",
            mode="max",
        ),
        param_space=space,
    )
    import json
    import pandas as pd
    for i in range(1):
        results = tuner.fit()
        best_result = results.get_best_result(metric="mean_accuracy", mode="max")
        df = best_result.metrics_dataframe
        df.to_csv(f'pbt_{i}_df_best.csv')
        df = all_runs = results.metric_dataframe()
        df.to_csv(f'pbt_{i}_df_all.csv')
        df = pd.DataFrame.from_dict(results.get_best_result().config)
        df.to_csv(f'pbt_{i}_best_hyperparamaters')


        ax = None
        for result in results:
            label = f"lr={result.config['lr']:.3f}, momentum={result.config['momentum']}"
            if ax is None:
                ax = result.metrics_dataframe.plot("training_iteration", "mean_accuracy", label=label)
            else:
                result.metrics_dataframe.plot("training_iteration", "mean_accuracy", ax=ax, label=label)
        ax.set_title("Mean Accuracy vs. Training Iteration for All Trials")
        ax.set_ylabel("Mean Test Accuracy")