import pandas as pd
from sklearn.utils import resample

def prepare_data():
    print("Stage 1: Data Cleaning & Preparation")
    
    # 1. Load the three CSV files
    # 2. Explore structure (already done in previous step, but loading now)
    print("Loading datasets...")
    try:
        ceas = pd.read_csv('CEAS_08.csv', encoding='utf-8', on_bad_lines='skip')
        nazario = pd.read_csv('Nazario.csv', encoding='utf-8', on_bad_lines='skip')
        spam_assassin = pd.read_csv('SpamAssasin.csv', encoding='utf-8', on_bad_lines='skip')
    except UnicodeDecodeError:
        # Fallback if utf-8 fails
        ceas = pd.read_csv('CEAS_08.csv', encoding='latin1', on_bad_lines='skip')
        nazario = pd.read_csv('Nazario.csv', encoding='latin1', on_bad_lines='skip')
        spam_assassin = pd.read_csv('SpamAssasin.csv', encoding='latin1', on_bad_lines='skip')

    # 3. Identify the column that contains the email text (it's 'body')
    # 4. Identify or create a label column (already exists as 'label')
    # 5. Rename columns to email_text and label
    datasets = [ceas, nazario, spam_assassin]
    clean_datasets = []
    
    for i, df in enumerate(['CEAS_08', 'Nazario', 'SpamAssassin']):
        curr_df = datasets[i]
        print(f"Processing {df}...")
        
        # Select and rename
        if 'body' in curr_df.columns and 'label' in curr_df.columns:
            curr_df = curr_df[['body', 'label']].rename(columns={'body': 'email_text'})
            clean_datasets.append(curr_df)
        else:
            print(f"Warning: {df} missing expected columns. Columns found: {curr_df.columns.tolist()}")

    # 6. Merge the three datasets
    df_merged = pd.concat(clean_datasets, ignore_index=True)
    print(f"Merged shape: {df_merged.shape}")

    # Dropping NaNs if any
    df_merged.dropna(subset=['email_text', 'label'], inplace=True)

    # 7. Remove duplicate emails based on email_text
    initial_len = len(df_merged)
    df_merged.drop_duplicates(subset=['email_text'], inplace=True)
    print(f"Removed {initial_len - len(df_merged)} duplicates.")

    # 8. Fix encoding issues (Force UTF-8 conversion)
    df_merged['email_text'] = df_merged['email_text'].apply(lambda x: str(x).encode('utf-8', 'ignore').decode('utf-8'))

    # 9. Remove empty or extremely short emails (e.g., < 10 characters)
    # Re-checking length after string conversion
    df_merged = df_merged[df_merged['email_text'].str.strip().str.len() > 10]
    print(f"Shape after removing short emails: {df_merged.shape}")

    # 10. Check class distribution
    print("\nClass distribution before balancing:")
    print(df_merged['label'].value_counts())

    # 11. Balance classes if imbalanced
    major_class = df_merged['label'].value_counts().idxmax()
    minor_class = df_merged['label'].value_counts().idxmin()
    
    df_major = df_merged[df_merged.label == major_class]
    df_minor = df_merged[df_merged.label == minor_class]
    
    # We'll downsample the majority class to match the minority class
    df_major_downsampled = resample(df_major, 
                                   replace=False,    # sample without replacement
                                   n_samples=len(df_minor), 
                                   random_state=42)
    
    df_balanced = pd.concat([df_major_downsampled, df_minor])
    
    # 12. Show final class distribution
    print("\nFinal class distribution:")
    print(df_balanced['label'].value_counts())

    # 13. Export the final cleaned dataframe
    df_balanced.to_csv('emails.csv', index=False, encoding='utf-8')
    print("\nExported to emails.csv successfully.")

if __name__ == "__main__":
    prepare_data()
