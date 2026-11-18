from kedro.pipeline import Node, Pipeline

from .nodes import clean_unemployment


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=clean_unemployment,
                inputs="sidra_raw",
                outputs="unemployment",
                name="clean_unemployment",
            ),
        ]
    )
