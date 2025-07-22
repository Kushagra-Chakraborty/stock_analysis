import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Assuming SwinFusionTransformer2D is in swin_fusion_transformer_2d.py
from swin_fusion_transformer_2d import SwinFusionTransformer2D

# File paths
x_data_path = './Testmodel-1/X_2d_reduced_data.npy'
fundamental_data_path = './Testmodel-1/fundamental_data.npy'
y_labels_path = './Testmodel-1/y_labels_20d.npy'

# Hyperparameters
import torch.optim.lr_scheduler

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0005 # Slightly increased learning rate
NUM_EPOCHS = 200 # Increased number of epochs


# Model parameters
# IMG_SIZE: (seq_len, num_features) of the input to PatchEmbedding
# Our X_2d_reduced_data.npy is (samples, 30, 7)
IMG_SIZE = (30, 7)  
PATCH_SIZE = (1, 7) # Each patch covers 1 time step and all 7 features
IN_CHANS = 1  # Not directly used for 2D time-series, but kept for Swin analogy
NUM_CLASSES = 3  # Long, Short, Hold
EMBED_DIM = 128 # Increased embedding dimension
DEPTHS = [3, 3] # Increased depths for Swin Transformer layers
NUM_HEADS = [4, 8] # Increased number of heads for attention
FUNDAMENTAL_FEATURES_DIM = 10 # Number of features in fundamental_data.npy

try:
    # Load data
    X_2d = np.load(x_data_path)
    fundamental_data = np.load(fundamental_data_path)
    y_labels = np.load(y_labels_path)

    # Ensure data types are correct for PyTorch
    X_2d_tensor = torch.tensor(X_2d, dtype=torch.float32)
    fundamental_data_tensor = torch.tensor(fundamental_data, dtype=torch.float32)
    y_labels_tensor = torch.tensor(y_labels, dtype=torch.long) # Labels should be long for CrossEntropyLoss

    # Split data into training and validation sets
    X_train, X_val, fund_train, fund_val, y_train, y_val = train_test_split(
        X_2d_tensor, fundamental_data_tensor, y_labels_tensor, test_size=0.2, random_state=42
    )

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(X_train, fund_train, y_train)
    val_dataset = TensorDataset(X_val, fund_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Fundamental training data shape: {fund_train.shape}")
    print(f"Fundamental validation data shape: {fund_val.shape}")

    # Initialize model
    model = SwinFusionTransformer2D(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=IN_CHANS,
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
        depths=DEPTHS,
        num_heads=NUM_HEADS,
        fundamental_features_dim=FUNDAMENTAL_FEATURES_DIM,
        window_size=(5,1) # Adjusted window size for Swin attention
    )

    # Calculate class weights for imbalanced dataset
    class_counts = torch.bincount(y_train)
    total_samples = len(y_train)
    class_weights = total_samples / (len(class_counts) * class_counts.float())
    # Normalize weights to sum to 1, or use as is (CrossEntropyLoss expects unnormalized weights)
    # class_weights = class_weights / class_weights.sum()

    # Loss function and optimizer (adam instead of adamw)(adam gave better accuracy)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train() # Set model to training mode
        running_loss = 0.0
        for batch_idx, (data, fund_data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(data, fund_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation loop
        model.eval() # Set model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, fund_data, labels in val_loader:
                outputs = model(data, fund_data)
                loss = criterion(outputs, labels) # Calculate validation loss
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {accuracy:.2f}%")

        # Step the scheduler based on validation loss
        scheduler.step(val_loss/len(val_loader))

        # Log metrics to a file
        with open("training_log.csv", "a") as f:
            if epoch == 0:
                f.write("epoch,train_loss,val_loss,val_accuracy\n")
            f.write(f"{epoch+1},{running_loss/len(train_loader):.4f},{val_loss/len(val_loader):.4f},{accuracy:.2f}\n")

    print("Training complete.")

    # Final evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, fund_data, labels in val_loader:
            outputs = model(data, fund_data)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n--- Final Validation Metrics ---")
    print(classification_report(all_labels, all_preds, target_names=['Long', 'Short', 'Hold']))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")