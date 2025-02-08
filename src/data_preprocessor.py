import pandas as pd
import string
import re
import emoji
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import os

label_mapping = {'negative': 0, 'positive': 1}

def convert_labels(y):
    return list(map(lambda x: label_mapping[x], y))

def basic_cleaning(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = emoji.replace_emoji(text, replace='')  # Remove emojis
    text = re.sub(r'[:;=8]?-?[)D]', '', text)  # Remove emoticons
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    return text

# Tokenize text
def tokenize_text(text):
    return word_tokenize(text)

# Remove stopwords
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

# Apply stemming
def stemming(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]

# Clean the text with stemming
def clean_text(text, apply_stemming=False):
    text = basic_cleaning(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = stemming(tokens)

    return ' '.join(tokens)

# Vectorize with TF-IDF
def vectorize_tfidf(train, test):
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train = tfidf_vectorizer.fit_transform(train['cleaned_review'])
    X_test = tfidf_vectorizer.transform(test['cleaned_review'])
    return X_train, X_test, tfidf_vectorizer

# Preprocessing pipeline that handles both train and test data
def preprocess_pipeline(train, test, apply_stemming=False):
    train['cleaned_review'] = train['review'].apply(lambda x: clean_text(x, apply_stemming))
    test['cleaned_review'] = test['review'].apply(lambda x: clean_text(x, apply_stemming))

    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = vectorize_tfidf(train, test)

    y_train = convert_labels(train['sentiment'])
    y_test = convert_labels(test['sentiment'])

    return X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf_vectorizer

# Loading the data from data/raw
def load_data():
    train_data = pd.read_csv('data/raw/train.csv')
    test_data = pd.read_csv('data/raw/test.csv')

    train_data = train_data.drop_duplicates(subset='review', keep='first')

    return train_data, test_data

# Saving our preprocessed data to data/processed
def save_processed_data(train, test):
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')

    train.to_csv('data/processed/train_processed.csv', index=False)
    test.to_csv('data/processed/test_processed.csv', index=False)

# Final execution
def execute_pipeline(apply_stemming=False):
    train, test = load_data()

    X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf_vectorizer = preprocess_pipeline(train, test, apply_stemming)

    save_processed_data(train, test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf_vectorizer
