"""Nos de modelagem da taxa de desocupacao um trimestre a frente."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def _lag_columns(lags: list[int]) -> list[str]:
    return [f"rate_lag_{lag}" for lag in lags]


def create_lag_features(unemployment: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """Acrescenta a taxa de trimestres anteriores como atributos de cada UF.

    Cada UF e reindexada na grade completa de trimestres antes do ``shift``, de modo
    que ``rate_lag_4`` seja sempre o mesmo trimestre do ano anterior, mesmo que
    falte algum trimestre na serie. Linhas sem alvo ou sem algum atributo saem.
    """
    periods = pd.date_range(unemployment["period"].min(), unemployment["period"].max(), freq="QS")
    wide = unemployment.pivot(index="period", columns="uf_code", values="rate").reindex(periods)
    wide.index.name = "period"

    features = unemployment[["uf_code", "uf_name", "period", "rate"]]
    for lag in lags:
        lagged = wide.shift(lag).reset_index().melt(id_vars="period", value_name=f"rate_lag_{lag}")
        features = features.merge(lagged, on=["uf_code", "period"], how="left")

    return (
        features.dropna(subset=["rate", *_lag_columns(lags)])
        .sort_values(["period", "uf_code"])
        .reset_index(drop=True)
    )


def split_by_time(
    model_input: pd.DataFrame, test_quarters: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa os ultimos ``test_quarters`` trimestres para teste; o resto e treino.

    A divisao e sempre cronologica: nenhum trimestre de teste e anterior a um de treino.
    """
    periods = sorted(model_input["period"].unique())
    if len(periods) <= test_quarters:
        raise ValueError("Nao ha trimestres suficientes para separar treino e teste.")
    cutoff = periods[-test_quarters]
    train = model_input[model_input["period"] < cutoff].reset_index(drop=True)
    test = model_input[model_input["period"] >= cutoff].reset_index(drop=True)
    return train, test


def train_model(train: pd.DataFrame, lags: list[int]) -> LinearRegression:
    """Ajusta uma regressao linear unica (todas as UFs juntas) sobre as defasagens."""
    model = LinearRegression()
    model.fit(train[_lag_columns(lags)], train["rate"])
    return model


def evaluate_model(
    model: LinearRegression, train: pd.DataFrame, test: pd.DataFrame, lags: list[int]
) -> tuple[pd.DataFrame, dict]:
    """Compara o modelo com as referencias ingenuas no periodo de teste.

    Referencias (calculadas nas mesmas linhas do modelo):
    - ``naive_last_quarter``: proximo trimestre = trimestre atual (``rate_lag_1``);
    - ``naive_same_quarter_last_year``: mesmo trimestre do ano anterior (``rate_lag_4``),
      incluida apenas se a defasagem 4 estiver entre os atributos.
    """
    predictions = test[["uf_code", "uf_name", "period", "rate"]].copy()
    predictions["predicted"] = model.predict(test[_lag_columns(lags)])
    predictions["naive_last_quarter"] = test["rate_lag_1"]

    mae = {
        "model": mean_absolute_error(predictions["rate"], predictions["predicted"]),
        "naive_last_quarter": mean_absolute_error(
            predictions["rate"], predictions["naive_last_quarter"]
        ),
    }
    if 4 in lags:
        predictions["naive_same_quarter_last_year"] = test["rate_lag_4"]
        mae["naive_same_quarter_last_year"] = mean_absolute_error(
            predictions["rate"], predictions["naive_same_quarter_last_year"]
        )

    metrics = {
        "target": "taxa de desocupacao (%), um trimestre a frente",
        "model_type": type(model).__name__,
        "features": _lag_columns(lags),
        "coefficients": dict(zip(_lag_columns(lags), map(float, model.coef_), strict=True)),
        "intercept": float(model.intercept_),
        "train": {
            "rows": len(train),
            "first_period": _quarter_label(train["period"].min()),
            "last_period": _quarter_label(train["period"].max()),
        },
        "test": {
            "rows": len(test),
            "ufs": int(test["uf_code"].nunique()),
            "first_period": _quarter_label(test["period"].min()),
            "last_period": _quarter_label(test["period"].max()),
        },
        "mae": {name: round(float(value), 4) for name, value in mae.items()},
        "model_beats_naive_last_quarter": bool(mae["model"] < mae["naive_last_quarter"]),
    }
    return predictions, metrics


def _quarter_label(timestamp: pd.Timestamp) -> str:
    return f"{timestamp.year}T{timestamp.quarter}"
