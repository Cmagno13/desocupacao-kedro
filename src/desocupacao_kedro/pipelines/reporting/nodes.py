"""No de relatorio: grafico do observado contra o previsto no periodo de teste."""

from __future__ import annotations

import pandas as pd
from matplotlib.figure import Figure


def plot_actual_vs_predicted(
    unemployment: pd.DataFrame, predictions: pd.DataFrame, uf_names: list[str]
) -> Figure:
    """Um painel por UF: serie observada completa e previsoes no periodo de teste."""
    missing = sorted(set(uf_names) - set(unemployment["uf_name"]))
    if missing:
        raise ValueError(f"UFs nao encontradas nos dados: {missing}")

    # Figure direto (sem pyplot): nao depende de backend grafico nem acumula figuras.
    figure = Figure(figsize=(4.2 * len(uf_names), 3.6), layout="constrained")
    axes = figure.subplots(1, len(uf_names), sharey=True, squeeze=False)[0]
    test_start = predictions["period"].min()

    for ax, uf_name in zip(axes, uf_names, strict=True):
        actual = unemployment[unemployment["uf_name"] == uf_name]
        predicted = predictions[predictions["uf_name"] == uf_name]
        ax.plot(actual["period"], actual["rate"], color="#3b4252", linewidth=1.6, label="Observado")
        ax.plot(
            predicted["period"],
            predicted["predicted"],
            color="#d95f02",
            linewidth=1.6,
            marker="o",
            markersize=3.5,
            label="Previsto (teste)",
        )
        ax.axvline(test_start, color="#9aa0a6", linewidth=0.8, linestyle=":")
        ax.set_title(uf_name, fontsize=11, loc="left")
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)

    axes[0].set_ylabel("Taxa de desocupação (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    figure.suptitle(
        "Taxa de desocupação por UF: observado x previsto um trimestre à frente", fontsize=12
    )
    return figure
