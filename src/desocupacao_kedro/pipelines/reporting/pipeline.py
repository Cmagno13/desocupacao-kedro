from kedro.pipeline import Node, Pipeline

from .nodes import plot_actual_vs_predicted


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=plot_actual_vs_predicted,
                inputs=["unemployment", "predictions", "params:report_ufs"],
                outputs="actual_vs_predicted_plot",
                name="plot_actual_vs_predicted",
            ),
        ]
    )
