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

def remove_outliers(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,X_test: pd.DataFrame, threshold: int) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Cap outliers in the training, validation, and test sets based on a threshold."""
    def cap_values_dataframe(df, z_scores, threshold):
        outliers = pd.DataFrame(np.abs(z_scores) > threshold, columns=df.columns, index=df.index)
        for column in df.columns:
            upper_bound = df[column].quantile(0.75) + (threshold * df[column].std())
            lower_bound = df[column].quantile(0.25) - (threshold * df[column].std())
            df.loc[outliers[column] & (df[column] > upper_bound), column] = upper_bound
            df.loc[outliers[column] & (df[column] < lower_bound), column] = lower_bound
        return df

    def cap_values_series(series, z_scores, threshold):
        outliers = np.abs(z_scores) > threshold
        upper_bound = series.quantile(0.75) + (threshold * series.std())
        lower_bound = series.quantile(0.25) - (threshold * series.std())
        series.loc[outliers & (series > upper_bound)] = upper_bound
        series.loc[outliers & (series < lower_bound)] = lower_bound
        return series

    z_scores_Xtrain = stats.zscore(X_train)
    z_scores_Xval = stats.zscore(X_val)
    z_scores_ytrain = stats.zscore(y_train)
    z_scores_yval = stats.zscore(y_val)
    z_scores_Xtest = stats.zscore(X_test)

    X_train = cap_values_dataframe(X_train, z_scores_Xtrain, threshold)
    y_train = cap_values_series(y_train, z_scores_ytrain, threshold)
    X_val = cap_values_dataframe(X_val, z_scores_Xval, threshold)
    y_val = cap_values_series(y_val, z_scores_yval, threshold)
    X_test = cap_values_dataframe(X_test, z_scores_Xtest, threshold)
    """
    X_train = X_train[z_scores_Xtrain < threshold]
    y_train = y_train[z_scores_ytrain < threshold]
    X_val = X_val[z_scores_Xval < threshold]
    y_val = y_val[z_scores_yval < threshold]
    X_test = X_test[z_scores_Xtest < threshold]
    """
    return X_train, y_train, X_val, y_val, X_test
    

def scale_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale the features using StandardScaler."""
    print("X_train columns:", X_train.columns)
    print("X_test columns:", X_test.columns)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled