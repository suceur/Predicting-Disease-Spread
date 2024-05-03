from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
import pandas as pd


def split_data(data: pd.DataFrame, test_size: float, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the data into training and validation sets."""
    features = data.drop(columns=['total_cases'])
    targets = data['total_cases']
    return train_test_split(features, targets, test_size=test_size, random_state=random_state)

def remove_outliers(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, threshold: int) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Remove outliers from the training and validation sets based on a threshold."""
    z_scores_train = np.abs(stats.zscore(y_train))
    z_scores_val = np.abs(stats.zscore(y_val))
    X_train = X_train[z_scores_train < threshold]
    y_train = y_train[z_scores_train < threshold]
    X_val = X_val[z_scores_val < threshold]
    y_val = y_val[z_scores_val < threshold]
    return X_train, y_train, X_val, y_val

def scale_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale the features using StandardScaler."""
    print("X_train columns:", X_train.columns)
    print("X_test columns:", X_test.columns)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled