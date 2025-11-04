# importing required libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

def train_model():
    # Load dataset
    heart_df = pd.read_csv("heart_cleveland_upload.csv")

    # Rename columns for clarity
    heart_df = heart_df.rename(columns={'condition': 'target'})
    print("Dataset Preview:\n", heart_df.head(), "\n")

    # Separate features and target
    X = heart_df.drop(columns='target')
    y = heart_df['target']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # FIX: no fit_transform here

    # Model training (Random Forest)
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model and scaler
    with open('heart-disease-prediction-model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('heart-disease-prediction-scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("\n✅ Model and Scaler saved successfully!")

if __name__ == "__main__":
    train_model()
