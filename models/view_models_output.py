import joblib
import pandas as pd
import json

def view_model_metadata():
    models = {
        'Logistic Regression': 'models/logistic_regression.pkl',
        'Decision Tree': 'models/decision_tree.pkl'
    }
    
    print("\n" + "="*50)
    print("                MODEL METADATA VIEWER                ")
    print("="*50 + "\n")
    
    for name, path in models.items():
        try:
            model = joblib.load(path)
            print(f"[{name}] Successfully loaded.")
            print(f"  Model Type: {type(model).__name__}")
            
            if hasattr(model, 'classes_'):
                print(f"  Target Classes: {model.classes_}")
            
            if hasattr(model, 'n_features_in_'):
                print(f"  Expected input features count: {model.n_features_in_}")
                
            if hasattr(model, 'feature_names_in_'):
                features_sample = list(model.feature_names_in_[:5])
                print(f"  Sample Features: {features_sample}...")
                
            if name == 'Decision Tree' and hasattr(model, 'max_depth'):
                print(f"  Max Depth: {model.max_depth}")
                print(f"  Actual Depth: {model.get_depth() if hasattr(model, 'get_depth') else 'Unknown'}")
                print(f"  Number of Leaves: {model.get_n_leaves() if hasattr(model, 'get_n_leaves') else 'Unknown'}")
                
            if name == 'Logistic Regression' and hasattr(model, 'best_params_'):
                print(f"  Best GridSearch Params: {model.best_params_}")
            elif name == 'Logistic Regression' and hasattr(model, 'coef_'):
                print(f"  Coefficients Sample: {model.coef_[0][:5]}...")
                print(f"  Intercept: {model.intercept_[0]}")
                
        except Exception as e:
            print(f"[{name}] Failed to load: {path}")
            print(f"  Error: {e}")
            
        print("-" * 50)

if __name__ == '__main__':
    view_model_metadata()
