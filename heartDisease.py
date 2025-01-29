# Heart Disease Prediction Project
# Author: [Your Name]
# Description: A machine learning project to predict the presence of heart disease.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Step 1: Load Data
file_path = "Dataset Heart Disease.csv"  # Replace with the actual file path
data = pd.read_csv(file_path)

# Step 2: Data Cleaning
data = data.drop(columns=['Unnamed: 0'])  # Remove unnecessary columns

# Step 3: Exploratory Data Analysis (EDA)
def eda(data):
    print("Dataset Info:")
    print(data.info())
    print("\nMissing Values:")
    print(data.isnull().sum())
    print("\nTarget Distribution:")
    sns.countplot(x='target', data=data, palette='viridis')
    plt.title("Distribution of Target Variable")
    plt.show()

    print("\nCorrelation Matrix:")
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()

# Uncomment to run EDA
# eda(data)

# Step 4: Data Preprocessing
X = data.drop(columns=['target'])
y = data['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Model Building and Evaluation
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Step 6: Save the Model
joblib.dump(model, "heart_disease_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved!")
