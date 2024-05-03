from typing import List
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
def create_correlation_matrix_plot(train_data: pd.DataFrame) -> go.Figure:
    weather_features = [
        'total_cases',
        'station_max_temp_c', 'station_min_temp_c', 'station_avg_temp_c',
        'precipitation_amt_mm', 'reanalysis_air_temp_k', 'reanalysis_precip_amt_kg_per_m2',
        'reanalysis_relative_humidity_percent', 'reanalysis_specific_humidity_g_per_kg'
    ]

    correlation_matrix = train_data[weather_features].corr()

    fig = px.imshow(correlation_matrix, x=correlation_matrix.columns, y=correlation_matrix.index,
                    color_continuous_scale='RdBu', title='Correlation Matrix of Dengue Cases and Weather Features')
    return fig

def create_feature_importance_plot(model: GradientBoostingRegressor, feature_names: List[str]) -> go.Figure:
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig = px.bar(x=[feature_names[i] for i in indices], y=importances[indices], labels={'x': 'Features', 'y': 'Importance'},
                 title='Feature Importance')
    return fig

def save_plots(correlation_matrix_plot: go.Figure, feature_importance_plot: go.Figure)-> None:
    correlation_matrix_plot.write_html("data/08_reporting/correlation_matrix_plot.html")
    feature_importance_plot.write_html("data/08_reporting/feature_importance_plot.html")
    
    