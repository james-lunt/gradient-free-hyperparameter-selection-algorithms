import matplotlib.pyplot as plt

def plot_pattern_search_train(file_paths):
    # Read the text file
    for i, file in enumerate(file_paths):
        with open(file, 'r') as file:
            lines = file.readlines()

        # Extract the array values
        line = lines[3]
        values = line.split(':')[1].strip().strip('[]').split(', ')
        values = [float(value) for value in values]

        # Plot the values
        plt.plot(values,  label=f'Experiment {i}')
    plt.xlabel('Iteration')
    plt.ylabel('Training Accuracy')
    plt.title('Pattern Search: Training Accuracy Plot')
    plt.legend()
    plt.show()


def plot_pattern_search_test(file_paths):
    # Read the text file
    for i, file in enumerate(file_paths):
        with open(file, 'r') as file:
            lines = file.readlines()

        # Extract the array values
        line = lines[3]
        values = line.split(':')[1].strip().strip('[]').split(', ')
        values = [float(value) for value in values]

        # Plot the values
        plt.plot(values,  label=f'Experiment {i}')
    plt.xlabel('Iteration')
    plt.ylabel('Test Accuracy')
    plt.title('Pattern Search: Test Accuracy Plot')
    plt.legend()
    plt.show()

#plot_pattern_search_test(['ps_data_0.txt','ps_data_1.txt', 'ps_data_2.txt'])
plot_pattern_search_train(['ps_data_train_1.txt','ps_data_train_2.txt','ps_data_train_3.txt'])