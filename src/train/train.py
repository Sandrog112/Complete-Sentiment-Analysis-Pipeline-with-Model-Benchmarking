import sys
import os
import joblib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data_preprocessor import execute_pipeline
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Execute the preprocessing pipeline to generate processed datasets
X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf_vectorizer = execute_pipeline()

def train_and_save_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)

    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['liblinear', 'saga']
    }

    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    output_dir = 'outputs/models'
    os.makedirs(output_dir, exist_ok=True)

    # Save the best model to a file
    model_filename = os.path.join(output_dir, 'logistic_regression_model.pkl')
    joblib.dump(best_model, model_filename)
    print(f"Model saved to {model_filename}")

if __name__ == "__main__":
    print("\nTraining Logistic Regression with Stemming + TF-IDF and Hyperparameter Tuning")
    train_and_save_logistic_regression(X_train_tfidf, y_train)