from kedro.pipeline import Node, Pipeline

from .nodes import create_lag_features, evaluate_model, split_by_time, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=create_lag_features,
                inputs=["unemployment", "params:lags"],
                outputs="model_input",
                name="create_lag_features",
            ),
            Node(
                func=split_by_time,
                inputs=["model_input", "params:test_quarters"],
                outputs=["train", "test"],
                name="split_by_time",
            ),
            Node(
                func=train_model,
                inputs=["train", "params:lags"],
                outputs="model",
                name="train_model",
            ),
            Node(
                func=evaluate_model,
                inputs=["model", "train", "test", "params:lags"],
                outputs=["predictions", "metrics"],
                name="evaluate_model",
            ),
        ]
    )
