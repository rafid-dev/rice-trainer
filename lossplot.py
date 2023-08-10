import sys
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print("Usage: python script_name.py <csv_file_path>")
    sys.exit(1)

csv_file_path = sys.argv[1]

try:
    # Read the CSV file
    data = pd.read_csv(csv_file_path)

    # Extract epoch and error columns
    epochs = data['epoch']
    errors = data['avg_epoch_error']

    # Create a line plot with logarithmic x-axis
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, errors, marker='o')
    plt.title('Epoch vs. Average Epoch Error')
    plt.xlabel('Epoch')
    plt.ylabel('Average Epoch Error')
    plt.xscale('log')  # Set x-axis to logarithmic scale
    plt.grid(True)
    plt.show()

except FileNotFoundError:
    print("File not found:", csv_file_path)
except Exception as e:
    print("An error occurred:", str(e))