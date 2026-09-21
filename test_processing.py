import re
import pandas as pd

print("--- Activity 1: Text Dataset Inspection ---")

df = pd.read_csv('text preprocessing ds.csv') 

print("\nFirst 5 records:")
print(df.head())

print("\nLast 5 records:")
print(df.tail())

print("\nBasic Information:")
df.info()

print(f"\nTotal rows and columns: {df.shape}")

print("\nMissing values per column:")
print(df.isnull().sum())

print("\n--- Activity 2: Handling Missing and Duplicate Text Data ---")

missing_comments = df[df['Comment'].isnull()]
print(f"Records containing missing comments:\n{missing_comments}")

df = df.dropna(subset=['Comment'])

duplicate_count = df['Comment'].duplicated().sum()
print(f"\nNumber of duplicate comments: {duplicate_count}")

df = df.drop_duplicates(subset=['Comment'])

print(f"Number of records remaining: {len(df)}")

print("\n--- Activity 3: Text Cleaning and Preprocessing ---")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    text = re.sub(r'@\w+|#\w+', '', text)
    
    text = re.sub(r'\d+', '', text)
    
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

df['Cleaned_Comment'] = df['Comment'].apply(clean_text)

print("\nSample of Cleaned Comments:")
print(df[['Comment', 'Cleaned_Comment']].head())
