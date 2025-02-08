import os
import pandas as pd
import gdown

os.makedirs('data/raw', exist_ok=True)

train_file_id = '1HPx9ZT4Z-deX4xBFui2RC04WZirE_qbe'
test_file_id = '1P6frG_5pWPCuiajrcyoUA7oJvCdnx4zc'

train_url = f'https://drive.google.com/uc?id={train_file_id}'
test_url = f'https://drive.google.com/uc?id={test_file_id}'

train_csv_path = 'data/raw/train.csv'
test_csv_path = 'data/raw/test.csv'

gdown.download(train_url, train_csv_path, quiet=False)
gdown.download(test_url, test_csv_path, quiet=False)

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

print("Train Data:")
print(train_df.head())

print("\nTest Data:")
print(test_df.head())