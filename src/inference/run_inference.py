import sys
import os
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import time
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data_preprocessor import execute_pipeline

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Load model from models directory
def load_model(model_filename='outputs/models/logistic_regression_model.pkl'):
    if os.path.exists(model_filename):
        model = joblib.load(model_filename)
        logger.info(f"Model loaded from {model_filename}")
        return model
    else:
        logger.error(f"Model file not found: {model_filename}")
        return None

# Load processed test data for inference
def load_inference_data():
    test_data = pd.read_csv('data/processed/test_processed.csv')
    return test_data['cleaned_review'], test_data['sentiment']

# Make predictions
def make_predictions(model, X_test_tfidf):
    return model.predict(X_test_tfidf)

# Evaluate model performance
def evaluate_model(y_test, predictions):
    cm = confusion_matrix(y_test, predictions)
    logger.info("\nConfusion Matrix:\n%s", cm)

    report = classification_report(y_test, predictions)
    logger.info("\nClassification Report:\n%s", report)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)

    return cm, report, accuracy, precision, recall, f1

# Save predictions
def save_predictions(y_test, predictions):
    output_dir = 'outputs/predictions'
    os.makedirs(output_dir, exist_ok=True)

    predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
    predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    
    logger.info("Predictions saved to outputs/predictions/predictions.csv")

# Save metrics
def save_metrics(y_test, predictions):
    cm, report, accuracy, precision, recall, f1 = evaluate_model(y_test, predictions)

    metrics_text = (
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1-Score: {f1:.4f}\n\n"
        f"Classification Report:\n{report}"
    )

    output_dir = 'outputs/predictions'
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write(metrics_text)
    
    logger.info("Metrics saved to outputs/predictions/metrics.txt")

# Save ROC curve
def save_roc_curve(y_test, model, X_test_tfidf):
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_tfidf)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')

        output_dir = 'outputs/figures'
        os.makedirs(output_dir, exist_ok=True)

        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        plt.close()
        logger.info("ROC curve saved to outputs/figures/roc_curve.png")
    else:
        logger.warning("Model does not support probability predictions. Skipping ROC curve.")

# Save confusion matrix plot
def save_confusion_matrix(cm):
    output_dir = 'outputs/figures'
    os.makedirs(output_dir, exist_ok=True)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)

    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    logger.info("Confusion matrix saved to outputs/figures/confusion_matrix.png")

# Main execution function
def main():
    X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf_vectorizer = execute_pipeline(processed=True)  # Use processed data for inference

    logger.info(f"Test dataset size: {X_test_tfidf.shape[0]} samples, {X_test_tfidf.shape[1]} features")

    model = load_model()

    if model:
        start_time = time.time()
        predictions = make_predictions(model, X_test_tfidf)
        end_time = time.time()

        logger.info(f"Inference completed in {end_time - start_time:.2f} seconds")

        cm, report, accuracy, precision, recall, f1 = evaluate_model(y_test, predictions)

        save_predictions(y_test, predictions)
        save_metrics(y_test, predictions)
        save_confusion_matrix(cm)
        save_roc_curve(y_test, model, X_test_tfidf)
    else:
        logger.error("Model loading failed. Exiting.")

if __name__ == "__main__":
    main()
