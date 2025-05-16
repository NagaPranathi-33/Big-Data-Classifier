import numpy as np
import os
import pandas as pd
from csv import reader

# Define the correct path for the Processed folder inside Main
BASE_PATH = "/content/drive/MyDrive/BProject2/146203bk2/Main/Processed"

def load_csv(filename):
    """Reads a CSV file and returns its content as a list."""
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if row:  # Ignore empty rows
                dataset.append(row)
    return dataset

def convert(data):
    """Converts dataset values to float, replacing '0.0' with 0.1."""
    return [[0.1 if val == '0.0' else float(val) for val in row] for row in data]

def find_class(Z, dts):
    """Extracts and converts class labels based on dataset type."""
    clas = Z[:, -1]  # Extract the last column as class labels
    label_map = {
        'Adult': {' <=50K': 0, ' >50K': 1},
        'Credit_Approval': {'-': 0, '+': 1}
    }
    return [label_map[dts].get(c, 0) for c in clas]

def find_unique(column):
    """Finds unique values in a column, excluding '?'."""
    return [val for val in np.unique(column) if val != ' ?']

def str_convert(column, unique_values):
    """Converts string categorical values to integer indices."""
    return ['?' if val == ' ?' else unique_values.index(val) for val in column]

def find_missing(data):
    """Handles missing values in dataset by replacing '?' with column mean."""
    data = np.array([[float(val) if val != '?' else -1000 for val in row] for row in data])
    
    # Compute column-wise mean, ignoring -1000 (missing values)
    valid_data = np.where(data != -1000, data, np.nan)
    column_means = np.nanmean(valid_data, axis=0)

    # Replace -1000 with column mean
    data[data == -1000] = np.take(column_means, np.where(data == -1000)[1])

    return data

def process_dataset(dts):
    """Reads and processes the dataset, handling missing values and categorical data."""
    filename = f"/content/drive/MyDrive/Bha Project/146203bk2/Main/data/{dts}.csv"
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Dataset file not found: {filename}")
    
    data = load_csv(filename)[1:]  # Remove header
    data = np.array(data).T  # Transpose for easier column-wise processing

    # Categorical feature indices for specific datasets
    categorical_indices = {
        'Adult': [1, 3, 5, 6, 7, 8, 9, 13],
        'Credit_Approval': [0, 3, 4, 5, 6, 8, 9, 11, 12]
    }

    # Convert categorical columns
    for i in categorical_indices.get(dts, []):
        unique_vals = find_unique(data[i])
        data[i] = str_convert(data[i], unique_vals)

    data = data.T  # Transpose back
    X, Y = data[:, :-1], find_class(data, dts)

    X = find_missing(X)  # Handle missing values

    # Save processed files in the correct directory
    np.savetxt(os.path.join(BASE_PATH, f"{dts}.csv"), X, delimiter=',', fmt='%s')
    np.savetxt(os.path.join(BASE_PATH, f"{dts}_label.csv"), Y, delimiter=',', fmt='%s')

    return convert(X)

def read_input(dts):
    """Reads processed dataset and labels from files."""
    data_path = os.path.join(BASE_PATH, f"{dts}.csv")
    label_path = os.path.join(BASE_PATH, f"{dts}_label.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found: {data_path}")

    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Processed labels not found: {label_path}")

    data = pd.read_csv(data_path, header=None).values
    label = pd.read_csv(label_path, header=None).values

    return data, label
