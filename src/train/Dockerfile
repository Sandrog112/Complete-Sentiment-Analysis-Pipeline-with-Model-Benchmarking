# Use a base Python image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy data folder to Docker (adjust the path if needed)
COPY data/raw/ /app/data/raw

# Copy the training code
COPY src/train/ /app/src/train

# Copy the data processing scripts
COPY src/data_loader.py /app/src/data_loader.py
COPY src/data_preprocessor.py /app/src/data_preprocessor.py

# Install any necessary packages listed in requirements.txt
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install NLTK resources 
RUN python -m nltk.downloader punkt_tab
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader wordnet

# Set environment variable for the model directory
ENV MODEL_DIR=/app/outputs/models

# Provision a volume for output files to persist after container execution
VOLUME ["/app/outputs"]

# Set the entrypoint for training or inference
CMD ["python3", "src/train/train.py"]