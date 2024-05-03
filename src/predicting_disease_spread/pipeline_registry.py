"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from predicting_disease_spread.pipelines import data_processing as dp
from predicting_disease_spread.pipelines import feature_engineering as fe
from predicting_disease_spread.pipelines import modeling as mod
from predicting_disease_spread.pipelines import submission as sub


from predicting_disease_spread.pipeline import create_pipeline
""""""
def register_pipelines() -> Dict[str, Pipeline]:
    
    """Register the project's pipelines.
    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    pipeline = create_pipeline()

    return {
        "__default__": pipeline,
    }

    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    """
    data_processing_pipeline = dp.create_pipeline()
    feature_engineering_pipeline = fe.create_pipeline()
    modeling_pipeline = mod.create_pipeline()
    submission_pipeline = sub.create_pipeline()

    return {
        "dp": data_processing_pipeline,
        "fe": feature_engineering_pipeline,
        "mod": modeling_pipeline,
        "sub": submission_pipeline,
        "__default__": data_processing_pipeline + feature_engineering_pipeline + modeling_pipeline + submission_pipeline,
    }
    
    """
    