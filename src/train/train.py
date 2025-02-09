import sys
import os
import joblib
import numpy as np
from sklearn.model_selection import cross_validate
import time
import logging
from sklearn.metrics import accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data_preprocessor import execute_pipeline
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Execute the preprocessing pipeline to generate processed datasets
X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf_vectorizer = execute_pipeline()

def train_and_save_logistic_regression(X_train, y_train):
    logger.info(f"Training dataset size: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    
    model = LogisticRegression(max_iter=1000)

    param_grid = {
        'C': [2],
        'penalty': ['l2'],
        'solver': ['saga']
    }

    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)

    start_time = time.time()  
    grid_search.fit(X_train, y_train)
    end_time = time.time()  

    best_model = grid_search.best_estimator_

    # Log the time spent on training
    training_time = end_time - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")

    # Cross-validation evaluation on the entire training data
    logger.info("Evaluating model using cross-validation")
    cv_results = cross_validate(best_model, X_train, y_train, cv=3, scoring='accuracy', return_train_score=True)

    # Log cross-validation results
    logger.info(f"Cross-validation mean accuracy: {np.mean(cv_results['test_score']):.4f}")
    logger.info(f"Cross-validation standard deviation: {np.std(cv_results['test_score']):.4f}")

    output_dir = 'outputs/models'
    os.makedirs(output_dir, exist_ok=True)

    # Save the best model to a file
    model_filename = os.path.join(output_dir, 'logistic_regression_model.pkl')
    joblib.dump(best_model, model_filename)
    logger.info(f"Model saved to {model_filename}")

if __name__ == "__main__":
    logger.info("\nTraining Logistic Regression with Stemming + TF-IDF and Hyperparameter Tuning")
    train_and_save_logistic_regression(X_train_tfidf, y_train)
