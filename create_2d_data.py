import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

file_path = './Testmodel-1/REL_final_q.csv'
output_file_path = './Testmodel-1/X_2d_data.npy'
window_size = 30

try:
    df = pd.read_csv(file_path)

    # Convert 'Date' column to datetime objects
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df = df.sort_values(by='Date')

    # Define features for the 2D input
    features = [
        'Price', 'Open', 'High', 'Low', 'Vol.', 'Change %', 'ma_50', 'ma_200',
        'positive_score', 'negative_score', 'neutral_score', 'sentiment_available', 'days_since_last_news'
    ]

    # Ensure all features exist in the DataFrame
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"Error: Missing features in CSV: {missing_features}")
        exit()

    # Select only the feature columns
    df_features = df[features]

    # Handle non-numeric values in 'Vol.' and 'Change %' by converting to numeric
    # This step is crucial as pandas might infer object type if mixed types are present
    df_features['Vol.'] = pd.to_numeric(df_features['Vol.'].astype(str).str.replace(',', ''), errors='coerce')
    df_features['Change %'] = pd.to_numeric(df_features['Change %'].astype(str).str.replace('%', ''), errors='coerce')

    # Impute missing values: forward-fill then backward-fill
    df_features = df_features.ffill().bfill()

    # Check if any NaNs remain after imputation (should not if ffill/bfill are effective)
    if df_features.isnull().sum().sum() > 0:
        print("Warning: NaN values still present after ffill and bfill. Inspect data.")
        print(df_features.isnull().sum())

    # Scale the features
    scaler = StandardScaler()
    df_features_scaled = pd.DataFrame(scaler.fit_transform(df_features), columns=features)
    print("Features scaled using StandardScaler.")

    # Create 2D windows
    X_2d = []
    for i in range(len(df_features_scaled) - window_size + 1):
        window = df_features_scaled.iloc[i:i+window_size].values
        X_2d.append(window)

    X_2d = np.array(X_2d)

    print(f"Shape of the 2D data (samples, window_size, num_features): {X_2d.shape}")
    print(f"Number of samples: {X_2d.shape[0]}")
    print(f"Window size (time steps): {X_2d.shape[1]}")
    print(f"Number of features per time step: {X_2d.shape[2]}")

    # Save the 2D data to a .npy file
    np.save(output_file_path, X_2d)
    print(f"2D data saved to {output_file_path}")

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")