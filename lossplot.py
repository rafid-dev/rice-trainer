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

    # Create a subplot for epoch and validation errors
    fig_errors, axs_errors = plt.subplots(1, 2, figsize=(14, 6))

    # Plot Epoch Error in the first subplot
    axs_errors[0].plot(epochs, train_errors, label='Train Error', color='blue')
    axs_errors[0].set_ylabel('Error')
    axs_errors[0].set_xlabel('Epoch')
    axs_errors[0].legend()
    axs_errors[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Set X-axis to display integers

    # Plot Validation Error in the second subplot
    axs_errors[1].plot(epochs, val_errors, label='Validation Error', color='red')
    axs_errors[1].set_ylabel('Error')
    axs_errors[1].set_xlabel('Epoch')
    axs_errors[1].legend()
    axs_errors[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Set X-axis to display integers

    plt.tight_layout()

    # Create a separate figure for the learning rate
    fig_lr, ax_lr = plt.subplots(figsize=(8, 6))
    ax_lr.plot(epochs, learning_rates, label='Learning Rate', color='orange')
    ax_lr.set_ylabel('Learning Rate')
    ax_lr.set_xlabel('Epoch')
    ax_lr.legend()
    ax_lr.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Set X-axis to display integers

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training metrics from a CSV file.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file containing training metrics')
    args = parser.parse_args()

    plot_metrics(args.csv_file)
