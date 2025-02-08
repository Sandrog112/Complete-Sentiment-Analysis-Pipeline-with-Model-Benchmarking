import sys
import os
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data_preprocessor import execute_pipeline


def load_model(model_filename='outputs/models/logistic_regression_model.pkl'):
   
    try:
        model = joblib.load(model_filename)
        print(f"Model loaded from {model_filename}")
        return model
    except FileNotFoundError:
        print(f"Model file not found: {model_filename}")
        return None

def make_predictions(model, X_test_tfidf):
   
    predictions = model.predict(X_test_tfidf)
    return predictions

def evaluate_model(y_test, predictions):
    
    cm = confusion_matrix(y_test, predictions)
    print("\nConfusion Matrix:")
    print(cm)

    report = classification_report(y_test, predictions)
    print("\nClassification Report:")
    print(report)

def main():
    X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf_vectorizer = execute_pipeline()

    model = load_model()

    if model:
        
        predictions = make_predictions(model, X_test_tfidf)

        print("Predictions on test data: ", predictions)

        accuracy = (predictions == y_test).mean()
        print(f"\nModel accuracy on test data: {accuracy:.4f}")

        evaluate_model(y_test, predictions)

if __name__ == "__main__":
    main()
