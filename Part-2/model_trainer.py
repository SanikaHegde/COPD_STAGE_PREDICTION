import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- 0. Setup ---
# Create a directory to save model artifacts if it doesn't exist
if not os.path.exists('model_artifacts'):
    os.makedirs('model_artifacts')

# --- 1. Define the Neural Network Architecture ---
# This is an improved version of the COPDNet from your project report
class COPDNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(COPDNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256), # Increased layer size for more capacity
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4), # Slightly increased dropout for regularization
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# --- 2. Data Loading and Preprocessing ---
print("Loading and preprocessing data...")
try:
    df = pd.read_csv('NHANES_with_COPD_Stage.csv')
except FileNotFoundError:
    print("Error: 'NHANES_with_COPD_Stage.csv' not found. Please make sure the file is in the correct directory.")
    exit()


# Remove 'Race' from features per request
features = ['Age', 'Sex', 'BMI', 'Baseline_FEV1_L', 'Baseline_FVC_L', 'Baseline_FEV1_FVC_Ratio']
target = 'COPD_Stage'

X = df[features]
y = df[target]

# Encode the categorical target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
np.save('model_artifacts/classes.npy', label_encoder.classes_)


# One-Hot Encode categorical features: only 'Sex' (no Race)
X = pd.get_dummies(X, columns=['Sex'], drop_first=True)
# save model input column names for the interface
joblib.dump(list(X.columns), 'model_artifacts/model_columns.pkl')


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Convert data to PyTorch Tensors for training
X_train_tensor = torch.FloatTensor(X_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)


# --- 3. Model Training (with Class Weights and AdamW Optimizer) ---
input_size = X_train_tensor.shape[1]
num_classes = len(label_encoder.classes_)
model = COPDNet(input_size=input_size, num_classes=num_classes)

# Calculate class weights to handle imbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

# Use the weights in the loss function
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# AdamW is often a better choice for optimizer
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

print("Starting Deep Learning model training...")
epochs = 150 # Increased epochs for better convergence
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

print("Training finished.")

# --- 4. Model Evaluation ---
print("\nEvaluating model on the test set...")
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    _, y_pred = torch.max(y_pred_tensor, 1)

accuracy = accuracy_score(y_test, y_pred.numpy())
print(f"\nModel Accuracy: {accuracy:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred.numpy(), target_names=label_encoder.classes_, zero_division=0))


# --- 5. Generate and Save Confusion Matrix ---
print("Generating confusion matrix...")
cm = confusion_matrix(y_test, y_pred.numpy())
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix for COPD Stage Prediction (Deep Learning)')
plt.ylabel('Actual Stage')
plt.xlabel('Predicted Stage')
plt.savefig('confusion_matrix_deep_learning.png')
print("Confusion matrix saved as 'confusion_matrix_deep_learning.png'")


# --- 6. Save the Model and Supporting Artifacts ---
torch.save(model.state_dict(), 'model_artifacts/copd_model.pth')
joblib.dump(scaler, 'model_artifacts/scaler.pkl')

print("\nDeep Learning Model, scaler, and other artifacts have been saved successfully!")
print("Use the provided app.py to run inference (it expects inputs without 'Race').")
