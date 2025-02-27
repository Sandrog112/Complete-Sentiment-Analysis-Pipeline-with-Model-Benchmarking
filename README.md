# Sentiment Classification for Movie Reviews
Welcome to the Sentiment Classification for Movie Reviews project! This project demonstrates how to classify movie reviews into binary sentiment categories. The project involves preprocessing the review text data and applying three different classification models to predict sentiment. It is structured to handle everything from data loading and preprocessing to model training and inference, ensuring a comprehensive pipeline for sentiment analysis.

## DS Part: Notebook Overview

### Conclusions from EDA (Exploratory Data Analysis)

In this project, I began by inspecting the train and test datasets, which consist of 40,000 instances in the training set and 10,000 reviews in the test set. Both datasets contain only two columns: review and sentiment (binary: positive or negative).

During the initial analysis, I identified some duplicate movie reviews in the training data, which were removed to ensure dataset quality.

Further exploration focused on the distributions of the target values and the lengths of the reviews:

- **Sentiment distribution**: The training dataset has a near-perfect balance between positive and negative reviews, with approximately 20,000 instances for each sentiment class.

- **Review lengths**: The review length distribution is positively skewed, with an average length of around 1,300 characters. Although the majority of reviews fall within this range, there are some outliers with lengths reaching up to 14,000 characters. While this is significantly longer than average, it is still manageable and can be handled during preprocessing.

### Overview of Data Preprocessing

The preprocessing pipeline for this project involves several key steps to clean and transform the raw movie reviews into a format suitable for machine learning models.

1. **Basic Cleaning**: The text data is first converted to lowercase, and punctuation, numbers, emojis, emoticons, URLs, and HTML tags are removed to eliminate noise and ensure consistency.

2. **Tokenization**: The cleaned text is then tokenized into individual words using the `word_tokenize` function, which splits the text into meaningful units.

3. **Stopword Removal**: Common English stopwords (e.g., "and", "the", "in") are removed to reduce dimensionality and focus on more important words.

4. **Stemming and Lemmatization**: To further process the text, both stemming and lemmatization are applied to the tokens. Stemming reduces words to their root form (e.g., "running" becomes "run"), while lemmatization converts words to their base form (e.g., "better" becomes "good"). The model experiments with both techniques, with the option to choose between stemming and lemmatization based on performance.

5. **Vectorization**: The cleaned and processed text is then transformed into numerical representations using two different vectorization techniques:
   - **TF-IDF Vectorization**: This method assigns weights to words based on their frequency in a document relative to their frequency across all documents.
   - **Count Vectorization**: This method counts the occurrences of each word in the document.

6. **Experimentation**: Throughout the preprocessing phase, various combinations of **stemming/lemmatization** and **TF-IDF/count vectorization** are tested to identify the best performing pair. Each combination is evaluated afterwards using a baseline model to determine which configuration yields the best results.

This iterative approach ensures that the most effective preprocessing steps are used to improve model performance. 


### Model Selection and Evaluation

After preprocessing the data, the next step was to experiment with different combinations of word reduction techniques (stemming vs lemmatization) and vectorizers (Count Vectorizer vs TF-IDF) to find the best pair for feature extraction. Naive Bayes was chosen as the baseline model due to its simplicity and effectiveness for text classification tasks.

After experimenting with these combinations, we found that **TF-IDF** outperforms Count Vectorizer. Moreover, **stemming** and **lemmatization** showed very similar results, so stemming was chosen randomly for the final preprocessing pipeline.

Next, the best-performing pair, **Stemming + TF-IDF**, was used with **Logistic Regression**, which was chosen for its efficiency and effectiveness in binary classification tasks. Hyperparameter tuning was applied to find the optimal configuration for the model.

As seen from the results, the Logistic Regression model achieved an accuracy of **89%**, which is a significant improvement compared to the baseline Naive Bayes model.

Finally, to explore diversity, a tree-based model, **XGBoost**, was tested.

The XGBoost model performed well, but it didn't surpass the Logistic Regression model in terms of accuracy.

The final model chosen for inference was **Logistic Regression with Stemming + TF-IDF**. The performance metrics for the final model are as follows:

## Final Model Metrics
- **Accuracy**: 0.8907
- **Precision**: 0.8908
- **Recall**: 0.8907
- **F1-Score**: 0.8907

### Classification Report:
```
              precision    recall  f1-score   support

           0       0.90      0.88      0.89      5000
           1       0.88      0.90      0.89      5000

    accuracy                           0.89     10000
   macro avg       0.89      0.89      0.89     10000
weighted avg       0.89      0.89      0.89     10000
```


## Potential Business Applications and Value for Business

The sentiment analysis model built in this project has several potential applications for businesses, particularly those in industries where customer feedback plays a crucial role in decision-making and strategy development. Below are some key business applications:

1. **Customer Feedback Analysis**: The model can be used to automatically analyze customer reviews, surveys, or social media posts, categorizing them into positive or negative sentiments. This helps businesses gain insights into customer satisfaction, identify areas of improvement, and prioritize issues that require immediate attention.

2. **Brand Monitoring**: By processing online mentions, businesses can track public sentiment about their brand in real-time. This enables businesses to respond to both positive and negative feedback proactively, improving their public relations efforts and brand reputation.

3. **Product Development**: The sentiment analysis model can be applied to user reviews of products or services, helping businesses understand what features are liked or disliked. This feedback loop can guide product improvements or the development of new features that better meet customer expectations.

4. **Market Research**: Sentiment analysis can assist businesses in understanding broader market trends, enabling them to gauge public opinion on various topics, from entertainment to politics. This can be useful for market research firms or companies seeking to predict trends based on public sentiment.

5. **Customer Support**: Automatically analyzing customer support tickets or interactions can help businesses detect customer issues more efficiently, categorize the urgency of requests, and even predict whether a customer might churn based on their sentiment.

6. **Social Media Campaigns**: Companies running advertising campaigns or engaging with their audience on social media can leverage sentiment analysis to measure the effectiveness of their campaigns in real time. This feedback can help them adjust their marketing strategies, content, or messaging to improve engagement and customer loyalty.

In summary, sentiment analysis of movie reviews (or any textual feedback) provides valuable insights that can be translated into actionable strategies for improving customer experience, brand image, and overall business performance. The ability to automatically categorize large volumes of text data saves time and resources while empowering businesses with real-time feedback to make data-driven decisions.




## ML Part: Guide & Quickstart Instructions

### Prerequisites

Before diving into the detailed steps of setting up and using this project, there are few important prerequisites or requirements that need to be addressed. These prerequisites ensure that your local development environment is ready and capable of efficiently running and supporting the project. 

### Forking and Cloning from GitHub

Create a copy of this repository by forking it on GitHub.

Clone the forked repository to your local machine:

```bash
git clone https://github.com/Sandrog112/EPAM-Final-Project.git
cd EPAM_Final_Project
```

### Setting Up Development Environment
Ensure you have Python 3.8+ installed on your machine. 

Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

Also make sure to have Docker Desktop installed

### Project structure:

This project has a modular structure, where each folder has a specific duty.

```bash
Sentiment-Classification/
├── outputs/                       
│   ├──predictions/            # Model predictions/results.
│   ├──figures/                # figures/plots.
│   ├──models/                 # Trained models and artifacts.
├── src/                       # Source code.
│   ├── __init__.py            # Init file for `src` module.
│   ├── inference/             # Inference code.
│   │   ├── Dockerfile         # Docker setup for inference.
│   │   ├── __init__.py        # Init file for `inference` module.
│   │   └── run_inference.py   # Inference script.
│   ├── train/                 # Training code.
│   │   ├── Dockerfile         # Docker setup for training.
│   │   ├── __init__.py        # Init file for `train` module.
│   │   └── train.py           # Training script.
│   ├── data_loader.py         # Dataset loading.
│   └── data_preprocessor.py   # Data preprocessing.
├── .gitignore                 # Git ignore file.
├── README.md                  # Project overview and setup.
├── notebook.ipynb             # Jupyter Notebook.
├── requirements.txt           # Python dependencies.

```

## Data Overview

This project uses a movie reviews dataset consisting of two parts: a training set and a test set, both in CSV format. The training set contains 40,000 reviews, while the test set contains 10,000 reviews. Each review is paired with a sentiment label, which is binary: "negative" or "positive".

## Data Processing

The data processing is handled by two scripts: `data_loader.py` and `data_preprocessor.py`.

- **`data_loader.py`**: This script loads the movie reviews dataset from a public Google Drive link, fetching both the training and test CSV files.

- **`data_preprocessor.py`**: This script preprocesses the raw reviews data by performing basic text cleaning, tokenization, stopword removal, and vectorization (using TF-IDF or Count Vectorizer). It prepares the data for training and evaluation, ensuring it is in the correct format for machine learning models.
- 
To load data run the following commands:
```bash
python src/data_loader.py
```

## Model Training
The training of the Logistic Regression is handled by the `train.py` script located in the `train` folder. The model is saved in the `models` folder after training.

To train the model on preprocessed data, simply run the following command:

```bash
python src/train/train.py
```

### Running Training with Docker
Build the Docker image for training:

```bash
docker build -t sentiment_train -f src/train/Dockerfile .
```

Run the Docker container to train the model:

```bash
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/outputs:/app/outputs" sentiment_train
```

## Inference 
Once the model is trained, it can be used for inference. The inference pipeline is implemented in the `run_inference.py` script located in the `inference` folder.

To run the inference locally:

```bash
python src/inference/run_inference.py
```

### Running Inference with Docker
Build the Docker image for inference:

```bash
docker build -t sentiment_inference -f src/inference/Dockerfile . 
```

Run the inference Docker container:

```bash
docker run --rm -v "$(pwd)/outputs:/app/outputs" -v "$(pwd)/data/processed:/app/data/processed" sentiment_inference    
```

## Wrap Up

This project provides a complete pipeline for sentiment analysis of movie reviews, from data loading and preprocessing to model training and inference. It utilizes Docker to simplify running the inference in any environment. The code is designed to be modular and easy to extend, allowing for further experimentation with different models and preprocessing techniques.
