from typing import Tuple, Dict, Union
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np
import pandas as pd

def train_model(X_train: np.ndarray, y_train: pd.Series, param_grid: Dict[str, list]) -> Tuple[Dict[str, Union[int, float]], float]:
    """Train the Gradient Boosting Regressor model with hyperparameter tuning."""
    grid_search = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42),
                               param_grid=param_grid,
                               scoring='neg_mean_absolute_error',
                               cv=5,
                               verbose=1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return grid_search.best_params_, -grid_search.best_score_, best_model

def evaluate_model(model: GradientBoostingRegressor, X_train: np.ndarray, y_train: pd.Series, X_val: np.ndarray, y_val: pd.Series) -> Tuple[ np.ndarray, float]:
    """Evaluate the model using cross-validation."""
    model.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    cv_scores = cross_val_score(model, np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)),
                                cv=5, scoring='neg_mean_absolute_error')
    return -cv_scores, -cv_scores.mean()

def predict(model: GradientBoostingRegressor, X_test: np.ndarray) -> np.ndarray:
    """Make predictions using the trained model."""
    return model.predict(X_test).round().astype(int)