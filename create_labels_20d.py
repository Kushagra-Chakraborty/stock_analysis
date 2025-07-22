import pandas as pd
import numpy as np

file_path = 'C:/Users/Kushagra/Desktop/Testmodel-1/REL_final_q.csv'
output_file_path = 'C:/Users/Kushagra/Desktop/Testmodel-1/y_labels_20d.npy'
window_size = 20  # 20 trading days
threshold = 0.01  # 1% price change

try:
    df = pd.read_csv(file_path)

    # Convert 'Date' column to datetime objects and sort
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df = df.sort_values(by='Date')

    # Calculate future price change
    # Shift 'Price' column by 'window_size' to get the price after 20 days
    df['Future_Price'] = df['Price'].shift(-window_size)
    df['Price_Change'] = (df['Future_Price'] - df['Price']) / df['Price']

    # Define labels
    # 'Long' if price increases by more than threshold
    # 'Short' if price decreases by more than threshold
    # 'Hold' otherwise
    def get_label(change):
        if change > threshold:
            return 'Long'
        elif change < -threshold:
            return 'Short'
        else:
            return 'Hold'

    df['Label'] = df['Price_Change'].apply(get_label)

    # Convert labels to numerical format
    label_mapping = {'Long': 0, 'Short': 1, 'Hold': 2}
    df['Label_Encoded'] = df['Label'].map(label_mapping)

    # Define the window size used for features (from create_2d_data.py)
    feature_window_size = 30

    # Calculate the number of samples that create_2d_data.py would produce
    # This is based on the original length of the DataFrame before dropping NaNs
    # and the feature_window_size
    num_samples_X_2d = len(df) - feature_window_size + 1

    # Extract the labels corresponding to the end of each feature window
    # The label for the first window (days 0-29) is based on day 29.
    # So, we start slicing from index (feature_window_size - 1).
    # We need 'num_samples_X_2d' labels.
    y_labels = df['Label_Encoded'].iloc[feature_window_size - 1 : feature_window_size - 1 + num_samples_X_2d].values

    # Check for NaNs within the selected y_labels. If there are any, it indicates a problem.
    # This can happen if look_ahead_days causes NaNs within the required slice.
    if np.isnan(y_labels).any():
        print("Error: NaN values found in the selected labels. Data inconsistency.")
        # Drop NaNs from the end, which result from the shift
        y_labels = y_labels[~np.isnan(y_labels)]
        print(f"NaNs removed. New shape: {y_labels.shape}")


    print(f"Shape of the labels: {y_labels.shape}")
    print(f"Number of 'Long' labels: {np.sum(y_labels == 0)}")
    print(f"Number of 'Short' labels: {np.sum(y_labels == 1)}")
    print(f"Number of 'Hold' labels: {np.sum(y_labels == 2)}")

    # Save the labels to a .npy file
    np.save(output_file_path, y_labels)
    print(f"Labels saved to {output_file_path}")

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
