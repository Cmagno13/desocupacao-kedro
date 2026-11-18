import pandas as pd
import pytest

from desocupacao_kedro.pipelines.reporting.nodes import plot_actual_vs_predicted


@pytest.fixture
def frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    periods = pd.date_range("2024-01-01", periods=4, freq="QS")
    unemployment = pd.concat(
        [
            pd.DataFrame({"uf_name": name, "period": periods, "rate": [5.0, 6.0, 7.0, 8.0]})
            for name in ("Bahia", "São Paulo")
        ]
    )
    predictions = unemployment[unemployment["period"] >= "2024-07-01"].assign(predicted=6.5)
    return unemployment, predictions


def test_plot_has_one_panel_per_uf(frames):
    figure = plot_actual_vs_predicted(*frames, uf_names=["Bahia", "São Paulo"])

    assert [ax.get_title(loc="left") for ax in figure.axes] == ["Bahia", "São Paulo"]
    observed, predicted = figure.axes[0].lines[:2]
    assert list(observed.get_ydata()) == [5.0, 6.0, 7.0, 8.0]
    assert list(predicted.get_ydata()) == [6.5, 6.5]


def test_plot_rejects_unknown_uf(frames):
    with pytest.raises(ValueError, match="Atlantida"):
        plot_actual_vs_predicted(*frames, uf_names=["Atlantida"])
