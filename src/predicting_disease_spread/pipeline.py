from kedro.pipeline import Pipeline, node
from .pipelines.data_processing.nodes import merge_data, fill_missing_values, encode_city, extract_month
from .pipelines.feature_engineering.nodes import split_data, remove_outliers, scale_features
from .pipelines.modeling.nodes import train_model, evaluate_model, predict
from .pipelines.submission.nodes import prepare_submission, save_submission
from .pipelines.visualization.visualization import create_correlation_matrix_plot, create_feature_importance_plot, save_plots

def create_pipeline(**kwargs):
    """Create the Kedro pipeline."""
    data_processing_pipeline = Pipeline(
        [
            node(merge_data, inputs=["dengue_features_train", "dengue_labels_train"], outputs="merged_data"),
            node(fill_missing_values, inputs=["merged_data", "dengue_features_test"], outputs=["filled_data", "filled_test_data"]),
            node(encode_city, inputs=["filled_data", "filled_test_data"], outputs=["encoded_train_data", "encoded_test_features"]),
            node(extract_month, inputs=["encoded_train_data", "encoded_test_features"], outputs=["processed_train_data", "processed_test_features"]),
        ]
    )

    feature_engineering_pipeline = Pipeline(
        [
            node(split_data, inputs=["processed_train_data", "params:test_size", "params:random_state"], outputs=["X_train", "X_val", "y_train", "y_val"]),
            node(remove_outliers, inputs=["X_train", "y_train", "X_val", "y_val", "params:outlier_threshold"], outputs=["X_train_clean", "y_train_clean", "X_val_clean", "y_val_clean"]),
            node(scale_features, inputs=["X_train_clean", "X_val_clean", "processed_test_features"], outputs=["X_train_scaled", "X_val_scaled", "X_test_scaled"]),
        ]
    )

    modeling_pipeline = Pipeline(
        [
            node(train_model, inputs=["X_train_scaled", "y_train_clean", "params:param_grid"], outputs=["best_params", "best_score", "trained_model"]),
            node(evaluate_model, inputs=["trained_model", "X_train_scaled", "y_train_clean", "X_val_scaled", "y_val_clean"], outputs=["cv_scores", "mean_cv_score"]),
            node(predict, inputs=["trained_model", "X_test_scaled"], outputs="predicted_cases"),
        ]
    )

    submission_pipeline = Pipeline(
        [
            node(prepare_submission, inputs=["dengue_features_test", "predicted_cases"], outputs="submission"),
            node(save_submission, inputs=["submission", "params:submission_filepath"], outputs=None),
        ]
        
        
    
    )
    visualization_pipeline = Pipeline(
        [
            node(create_correlation_matrix_plot, inputs=["processed_train_data"], outputs="correlation_matrix_plot"),
            node(create_feature_importance_plot, inputs=["trained_model", "params:feature_names"], outputs="feature_importance_plot"),
            node(save_plots, inputs=["correlation_matrix_plot", "feature_importance_plot"], outputs=None),
        ]
    )

    return data_processing_pipeline + feature_engineering_pipeline + modeling_pipeline + submission_pipeline + visualization_pipeline
   


