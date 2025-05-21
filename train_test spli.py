import numpy as np
import pandas as pd

# Load your dataset using pandas (replace 'path_to_dataset.csv' with the actual path to your dataset)
data = pd.read_csv(r"C:\Users\kesha\Desktop\PROJECT_12_MARCH\GNN_MULTI_NEGATIVE\raw\GNN_multiclass_negative.csv")

# Specify the test size and random seed for reproducibility
test_size = 0.3  # You can change this to the desired test set size
random_seed = 42  # You can change this to any integer for the random seed

# Convert DataFrame to NumPy array
data_array = data.to_numpy()

# Shuffle the data array randomly
np.random.seed(random_seed)  # Set the random seed for reproducibility
np.random.shuffle(data_array)

# Calculate the index to split the array
split_index = int(len(data_array) * (1 - test_size))

# Split the data array into train and test sets
train_data, test_data = data_array[:split_index], data_array[split_index:]

# Convert train and test sets back to DataFrames (assuming the first row contains column names)
train_df = pd.DataFrame(train_data, columns=data.columns)
test_df = pd.DataFrame(test_data, columns=data.columns)

# Save train and test sets as CSV files (replace 'path_to_save_train.csv' and 'path_to_save_test.csv')
train_df.to_csv(r"C:\Users\kesha\Desktop\PROJECT_12_MARCH\GNN_MULTI_NEGATIVE\raw\GNN_train.csv", index=False)
test_df.to_csv(r"C:\Users\kesha\Desktop\PROJECT_12_MARCH\GNN_MULTI_NEGATIVE\raw\GNN_test.csv", index=False)
