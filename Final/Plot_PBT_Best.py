import pandas as pd
import matplotlib.pyplot as plt

def plot_mean_accuracy(csv_files):
    for i,file in enumerate(csv_files):
        df = pd.read_csv(file)
        plt.plot(df['mean_accuracy'], label=f'Experiment {i}')

    plt.xlabel('Iteration')
    plt.ylabel('Test Accuracy')
    plt.title('Population Based Training: Test Accuracy Plot \n for the best performing instances across experiments')
    plt.legend()
    plt.show()

csv_files = ['pbt_0_df_best.csv','pbt_1_df_best.csv','pbt_2_df_best.csv']
plot_mean_accuracy(csv_files)