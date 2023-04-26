import numpy as np
from tensorflow import keras
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Define the search space for hyperparameters
param_dist = {
    'learning_rate': [0.001, 0.01, 0.1],
    'momentum': [0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
    'units': [32, 64, 128, 256],
    'activation': ['relu', 'sigmoid', 'tanh'],
    'optimizer': ['sgd', 'adam', 'nadam']
}

# Define the model architecture
def create_model(units=128, activation='relu', optimizer='adam'):
    model = keras.Sequential([
        keras.layers.Dense(units=units, activation=activation, input_shape=(X_train.shape[1],)),
        keras.layers.Dense(units=1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Wrap the Keras model with a scikit-learn classifier
model = KerasClassifier(build_fn=create_model)

# Perform randomized search using Nesterov accelerated gradient descent
n_iterations = 10
nesterov_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=n_iterations,
    cv=3,
    n_jobs=-1
)

# Fit the search to the data
nesterov_search.fit(X_train, y_train, epochs=100, verbose=0, validation_split=0.2)

# Print the best hyperparameters found
print(nesterov_search.best_params_)