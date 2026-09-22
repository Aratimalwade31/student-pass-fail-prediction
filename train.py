import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from database.fetch_data import load_data_from_mongo

def train_and_evaluate():
    # 1. Load Data from MongoDB
    df = load_data_from_mongo()
    
    # 2. Preprocessing
    df = df.dropna().drop_duplicates()
    
    # Features and Target
    X = df[['study_hours', 'attendance_percentage', 'internal_marks', 
            'assignment_marks', 'previous_exam_marks', 'practice_test_scores', 'number_of_backlogs']]
    y = df['pass_fail']  # Target variable (1 for Pass, 0 for Fail)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Model Building
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # Model Evaluation
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    print("--- Model Performance ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # 4. Save Model & Scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/logistic_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("Model and Scaler successfully saved in 'models/' folder.")

if __name__ == "__main__":
    train_and_evaluate()