import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from src.data_preprocessing import handle_missing_values, treat_outliers, scale_features, handle_class_imbalance
from src.feature_engineering import create_features
from src.model_training import train_logistic_regression, train_random_forest, train_decision_tree, train_xgboost
from src.evaluation import evaluate_model

def main():
    print("Loading test data...")
    if not os.path.exists("data/credit_data.csv"):
        print("Data not found. Please run generate_data.py first.")
        sys.exit(1)
        
    df = pd.read_csv("data/credit_data.csv")
    
    print("Feature Engineering...")
    df = create_features(df)
    
    target_col = 'SeriousDlqin2yrs'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print("Preprocessing (Imputing & Scaling)...")
    X_train, X_test = handle_missing_values(X_train, X_test)
    X_train = treat_outliers(X_train, X_train.columns)
    X_train, X_test = scale_features(X_train, X_test)
    
    print("Applying SMOTE for class imbalance...")
    X_train, y_train = handle_class_imbalance(X_train, y_train)
    
    print("Training Models (This might take a minute)...")
    
    print("- Logistic Regression")
    train_logistic_regression(X_train, y_train)
    
    print("- Decision Tree")
    train_decision_tree(X_train, y_train)
    
    print("- Random Forest")
    train_random_forest(X_train, y_train)
    
    print("- XGBoost")
    train_xgboost(X_train, y_train)
    
    print("✅ All models generated and stored successfully in the models/ directory!")

if __name__ == "__main__":
    main()
