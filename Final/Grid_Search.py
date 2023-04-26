from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

# Load the iris dataset
iris = load_iris()

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42)

# Define the parameter grid to search over
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Create an SVM classifier
svm = SVC()

# Create a GridSearchCV object
grid_search = GridSearchCV(svm, param_grid, cv=5)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and corresponding accuracy score
print("Best hyperparameters: ", grid_search.best_params_)
print("Best validation score: {:.2f}".format(grid_search.best_score_))
print("Test set accuracy: {:.2f}".format(grid_search.score(X_val, y_val)))