import pandas as pd
import matplotlib.pyplot as plt

def plot_mean_train_score(csv_files):
    # Read the CSV file into a pandas DataFrame
    for i, df in enumerate(csv_files):
        data = pd.read_csv(df)

        # Extract the 'mean_train_score' column
        mean_train_score = data['mean_train_score']

        # Create the plot with a legend label
        plt.plot(mean_train_score, label=f'Experiment {i}')

    plt.xlabel('Iteration')
    plt.ylabel('Train Accuracy')
    plt.title('Grid Search: Train Accuracy Plot')

    # Add a legend to the plot
    plt.legend()

    # Display the plot
    plt.show()

def plot_mean_test_score(csv_files):
    # Read the CSV file into a pandas DataFrame
    for i, df in enumerate(csv_files):
        data = pd.read_csv(df)

        # Extract the 'mean_train_score' column
        mean_train_score = data['mean_test_score']

        # Create the plot with a legend label
        plt.plot(mean_train_score, label=f'Experiment {i}')

    plt.xlabel('Iteration')
    plt.ylabel('Test Accuracy')
    plt.title('Grid Search: Test Accuracy Plot')

    # Add a legend to the plot
    plt.legend()

    # Display the plot
    plt.show()

csv_files = ['grid_search_0_df.csv', 'grid_search_1_df.csv',
              'grid_search_2_df.csv',
              'grid_search_3_df.csv', 'grid_search_4_df.csv']
plot_mean_train_score(csv_files)
plot_mean_test_score(csv_files)