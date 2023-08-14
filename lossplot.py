import argparse
import csv
import matplotlib.pyplot as plt

def plot_metrics(csv_file):
    epochs = []
    train_errors = []
    val_errors = []
    learning_rates = []

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_errors.append(float(row['train_error']))
            val_errors.append(float(row['val_error']))
            learning_rates.append(float(row['learning_rate']))

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].plot(epochs, train_errors, label='Train Error')
    axs[0].plot(epochs, val_errors, label='Validation Error')
    axs[0].set_ylabel('Error')
    axs[0].set_xlabel('Epoch')
    axs[0].legend()

    axs[1].plot(epochs, learning_rates, color='orange')
    axs[1].set_ylabel('Learning Rate')
    axs[1].set_xlabel('Epoch')

    axs[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Set X-axis to display integers

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training metrics from a CSV file.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file containing training metrics')
    args = parser.parse_args()

    plot_metrics(args.csv_file)
