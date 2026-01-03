# src/data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df, target_column):
    # Drop rows with missing target
    df = df.dropna(subset=[target_column])
    
    # Fill missing values (simple example)
    df = df.fillna({
        'reviews_per_month': 0
    })
    
    # Encode categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Split features/target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test
