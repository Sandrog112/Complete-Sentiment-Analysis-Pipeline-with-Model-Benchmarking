import sys
import os
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data_preprocessor import execute_pipeline


# Loading the model from models
def load_model(model_filename='outputs/models/logistic_regression_model.pkl'):
   
    try:
        model = joblib.load(model_filename)
        print(f"Model loaded from {model_filename}")
        return model
    except FileNotFoundError:
        print(f"Model file not found: {model_filename}")
        return None
    
# Making predictions and doing the evaluation of the model
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

    return cm, report

# Saving those predictions
def save_predictions(y_test, predictions):
   
    predictions_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': predictions
    })
    output_dir = 'outputs/predictions'
    os.makedirs(output_dir, exist_ok=True)
    predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    print("Predictions saved to outputs/predictions/predictions.csv")

# Saving model metrics
def save_metrics(y_test, predictions):
   
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)

    report = classification_report(y_test, predictions)

    metrics_text = (
        f"Accuracy: {accuracy}\n"
        f"Precision: {precision}\n"
        f"Recall: {recall}\n"
        f"F1-Score: {f1}\n\n"
        f"Classification Report:\n{report}"
    )

    output_dir = 'outputs/predictions'
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write(metrics_text)
    
    print("Detailed metrics and classification report saved to outputs/predictions/metrics.txt")

def save_confusion_matrix(cm):
   
    output_dir = 'outputs/figures'
    os.makedirs(output_dir, exist_ok=True)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    print("Confusion matrix plot saved to outputs/figures/confusion_matrix.png")

# Executing all the processes
def main():
    X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf_vectorizer = execute_pipeline()

    model = load_model()

    if model:
        predictions = make_predictions(model, X_test_tfidf)

        accuracy = (predictions == y_test).mean()
        print(f"\nModel accuracy on test data: {accuracy:.4f}")

        cm, report = evaluate_model(y_test, predictions)

        save_predictions(y_test, predictions)
        save_metrics(y_test, predictions)
        save_confusion_matrix(cm)

if __name__ == "__main__":
    main()
