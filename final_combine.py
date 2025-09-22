import pymongo
import pandas as pd
import re
from urllib.parse import urlparse
import os

# --- Feature Extraction Function ---
def extract_features(url):
    features = {}
    parsed_url = urlparse(url)
    features['url_length'] = len(url)
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['num_underscores'] = url.count('_')
    features['num_slashes'] = url.count('/')
    features['num_at_symbols'] = url.count('@')
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['num_params'] = len(parsed_url.query.split('&')) if parsed_url.query else 0
    features['domain_length'] = len(parsed_url.netloc)
    features['is_phish'] = 1 if 'phish' in url.lower() else 0
    features['is_secure'] = 1 if parsed_url.scheme == 'https' else 0
    return features

# --- Main Script ---
if __name__ == "__main__":
    print("Starting data combination and feature extraction...")

    # --- Step 1: Get Scraped Data from MongoDB ---
    try:
        # **CORRECTED HOSTNAME**
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client["phishing_db"]
        collection = db["websites"]
        cursor = collection.find({})
        df_scraped = pd.DataFrame(list(cursor))
        if '_id' in df_scraped.columns:
            df_scraped = df_scraped.drop(columns=['_id'])
        
        # Add a label for scraped phishing data
        df_scraped['label'] = 1
        print(f"Loaded {len(df_scraped)} scraped URLs from MongoDB.")
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        df_scraped = pd.DataFrame(columns=['url', 'label'])

    # --- Step 2: Load Phishing URLs from CSV ---
    try:
        df_phishing_csv = pd.read_csv('phishing_urls.csv', header=None, names=['id', 'url'])
        df_phishing_csv['label'] = 1
        print(f"Loaded {len(df_phishing_csv)} phishing URLs from CSV.")
    except FileNotFoundError:
        print("Error: 'phishing_urls.csv' not found.")
        df_phishing_csv = pd.DataFrame(columns=['url', 'label'])
    
    # --- Step 3: Load Legitimate URLs from CSV ---
    try:
        df_legitimate = pd.read_csv('top-1m.csv', header=None, names=['rank', 'url'])
        df_legitimate['label'] = 0
        print(f"Loaded {len(df_legitimate)} legitimate URLs from CSV.")
    except FileNotFoundError:
        print("Error: 'top-1m.csv' not found.")
        df_legitimate = pd.DataFrame(columns=['url', 'label'])

    # --- Step 4: Combine all URLs into a single DataFrame ---
    combined_urls = pd.concat([df_scraped[['url', 'label']], df_phishing_csv[['url', 'label']], df_legitimate[['url', 'label']]], ignore_index=True)
    combined_urls.drop_duplicates(subset=['url'], inplace=True)
    
    # --- Step 5: Extract features from the combined URLs ---
    features_df = combined_urls['url'].apply(extract_features).apply(pd.Series)
    
    # --- Step 6: Finalize the dataset ---
    final_dataset = pd.concat([combined_urls, features_df], axis=1)
    
    # Save the final dataset to a new CSV file
    final_dataset.to_csv('final_dataset_with_features.csv', index=False)
    
    print("\n--- Final Dataset Summary ---")
    print(f"Total rows in final dataset: {len(final_dataset)}")
    print("Distribution of labels:")
    print(final_dataset['label'].value_counts())
    print("\nFinal dataset saved to 'final_dataset_with_features.csv'.")