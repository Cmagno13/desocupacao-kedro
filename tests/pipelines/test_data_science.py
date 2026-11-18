import pandas as pd
import pytest

from desocupacao_kedro.pipelines.data_science.nodes import (
    create_lag_features,
    evaluate_model,
    split_by_time,
    train_model,
)


def _series(uf_code: int, uf_name: str, rates: list[float], start: str = "2020-01-01"):
    periods = pd.date_range(start, periods=len(rates), freq="QS")
    return pd.DataFrame({"uf_code": uf_code, "uf_name": uf_name, "period": periods, "rate": rates})


@pytest.fixture
def unemployment() -> pd.DataFrame:
    return pd.concat(
        [
            _series(11, "Rondônia", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            _series(35, "São Paulo", [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
        ],
        ignore_index=True,
    )


def test_lags_use_previous_quarters_of_the_same_uf(unemployment):
    features = create_lag_features(unemployment, lags=[1, 2, 4])

    # so restam as linhas com as 3 defasagens: 5o e 6o trimestres de cada UF
    assert len(features) == 4
    row = features[(features["uf_code"] == 35) & (features["period"] == "2021-01-01")].iloc[0]
    assert (row["rate"], row["rate_lag_1"], row["rate_lag_2"], row["rate_lag_4"]) == (
        50.0,
        40.0,
        30.0,
        10.0,
    )
    rondonia = features[features["uf_code"] == 11]
    assert rondonia["rate_lag_1"].max() < 10  # nada vazou de Sao Paulo


def test_lags_respect_calendar_when_a_quarter_is_missing(unemployment):
    with_gap = unemployment[
        ~((unemployment["uf_code"] == 11) & (unemployment["period"] == "2020-10-01"))
    ]

    features = create_lag_features(with_gap, lags=[1])

    rondonia = features[features["uf_code"] == 11].set_index("period")
    # 2021T1 nao tem trimestre anterior observado: sai, em vez de usar 2020T3 como "lag 1"
    assert pd.Timestamp("2021-01-01") not in rondonia.index
    assert rondonia.loc[pd.Timestamp("2021-04-01"), "rate_lag_1"] == 5.0


def test_split_is_chronological(unemployment):
    features = create_lag_features(unemployment, lags=[1])

    train, test = split_by_time(features, test_quarters=2)

    assert train["period"].max() < test["period"].min()
    assert test["period"].nunique() == 2
    assert len(train) + len(test) == len(features)
    assert set(test["uf_code"]) == {11, 35}


def test_split_needs_enough_quarters(unemployment):
    features = create_lag_features(unemployment, lags=[1])

    with pytest.raises(ValueError, match="suficientes"):
        split_by_time(features, test_quarters=5)


def test_model_recovers_a_linear_relation():
    # rate(t) = 2 * rate(t-1), exatamente
    data = create_lag_features(_series(11, "Rondônia", [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]), lags=[1])
    train, test = split_by_time(data, test_quarters=1)

    model = train_model(train, lags=[1])
    predictions, metrics = evaluate_model(model, train, test, lags=[1])

    assert predictions["predicted"].iloc[0] == pytest.approx(32.0)
    assert metrics["mae"]["model"] == pytest.approx(0.0, abs=1e-6)
    # referencia ingenua: repete 16.0 quando o observado e 32.0
    assert metrics["mae"]["naive_last_quarter"] == pytest.approx(16.0)
    assert metrics["model_beats_naive_last_quarter"] is True
    assert "naive_same_quarter_last_year" not in metrics["mae"]


def test_evaluation_reports_both_baselines_on_the_same_rows(unemployment):
    data = create_lag_features(unemployment, lags=[1, 4])
    train, test = split_by_time(data, test_quarters=1)

    predictions, metrics = evaluate_model(train_model(train, [1, 4]), train, test, lags=[1, 4])

    assert len(predictions) == len(test) == metrics["test"]["rows"]
    # 2021T2: observado 6 e 60; trimestre anterior 5 e 50; ano anterior 2 e 20
    assert metrics["mae"]["naive_last_quarter"] == pytest.approx((1 + 10) / 2)
    assert metrics["mae"]["naive_same_quarter_last_year"] == pytest.approx((4 + 40) / 2)
    assert metrics["test"]["first_period"] == metrics["test"]["last_period"] == "2021T2"
