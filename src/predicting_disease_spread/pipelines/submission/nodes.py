import pandas as pd
import numpy as np

def prepare_submission(test_features: pd.DataFrame, predicted_cases: np.ndarray) -> pd.DataFrame:
    """Prepare the submission DataFrame."""
    submission = test_features[['city', 'year', 'weekofyear']]
    #submission.rename(columns={'city_sj': 'city'}, inplace=True)
    submission['total_cases'] = predicted_cases
    
    def change_value(x: int) -> str:
        """Change the value of 'city' column based on the given mapping."""
        if x == 1:
            return 'sj'
        elif x == 0:
            return 'iq'
        else:
            return str(x)
    
    submission['city'] = submission['city'].apply(change_value)
    submission = submission[['city', 'year', 'weekofyear', 'total_cases']]
    return submission

def save_submission(submission: pd.DataFrame, filepath: str) -> None:
    """Save the submission DataFrame to a CSV file."""
    submission.to_csv(filepath, index=False)