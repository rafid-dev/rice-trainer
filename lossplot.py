import argparse
import pandas as pd
import matplotlib.pyplot as plt

# Set up argument parser
parser = argparse.ArgumentParser(description='Plot epoch and error data.')
parser.add_argument('csv_file', type=str, help='Path to CSV file containing epoch and error data')
parser.add_argument('--logx', action='store_true', help='Use logarithmic scale for x-axis')

args = parser.parse_args()

csv_file_path = args.csv_file

try:
    # Read the CSV file
    data = pd.read_csv(csv_file_path)

    # Extract epoch and error columns
    epochs = data['epoch']
    errors = data['avg_epoch_error']

    # Create a line plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, errors, marker='o')
    plt.title('Epoch vs. Average Epoch Error')
    plt.xlabel('Epoch')
    plt.ylabel('Average Epoch Error')

    if args.logx:
        plt.xscale('log')  # Set x-axis to logarithmic scale

    plt.grid(True)
    plt.show()

except FileNotFoundError:
    print("File not found:", csv_file_path)
except Exception as e:
    print("An error occurred:", str(e))
