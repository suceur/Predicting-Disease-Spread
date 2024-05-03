import pandas as pd
from typing import Tuple


def merge_data(train_features: pd.DataFrame, train_labels: pd.DataFrame) -> pd.DataFrame:
    """Merge the training features and labels based on city, year, and weekofyear."""
    return pd.merge(train_features, train_labels, on=['city', 'year', 'weekofyear'])


def fill_missing_values(data: pd.DataFrame,test_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fill missing values in the data with the median value of each column."""
    for column in data.columns:
        if data[column].isnull().any():
            median_value = data[column].median()
            data[column].fillna(median_value, inplace=True)
            test_features[column].fillna(median_value, inplace=True)
    return data, test_features


def encode_city(train_data: pd.DataFrame, test_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Encode the 'city' column using one-hot encoding."""
    train_data = pd.get_dummies(train_data, columns=['city'], drop_first=True)
    test_features = pd.get_dummies(test_features, columns=['city'], drop_first=True)
    return train_data, test_features

   
def extract_month(train_data: pd.DataFrame, test_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract the month from the 'week_start_date' column and drop the original column."""
    train_data['month'] = pd.to_datetime(train_data['week_start_date']).dt.month
    train_data.drop(columns=['week_start_date'], inplace=True)
    test_features['month'] = pd.to_datetime(test_features['week_start_date']).dt.month
    test_features.drop(columns=['week_start_date'], inplace=True)
    return train_data, test_features   