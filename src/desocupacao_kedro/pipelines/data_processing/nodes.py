"""Nos de preparacao: do CSV bruto do SIDRA para uma tabela tipada e organizada."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

TIDY_COLUMNS = ["uf_code", "uf_name", "year", "quarter", "period", "rate"]


def clean_unemployment(raw: pd.DataFrame) -> pd.DataFrame:
    """Tipa e organiza a taxa de desocupacao por UF e trimestre.

    Valores ausentes: o SIDRA usa marcadores de texto (``...``, ``-``, ``X``) quando
    nao ha estimativa. Qualquer valor nao numerico vira ``NaN`` e a linha e
    descartada, com aviso no log; nada e imputado.
    """
    rate = pd.to_numeric(raw["V"], errors="coerce")
    quarter_code = raw["D3C"].astype(str)
    year = quarter_code.str[:4].astype(int)
    quarter = quarter_code.str[4:].astype(int)

    tidy = pd.DataFrame(
        {
            "uf_code": raw["D1C"].astype(int),
            "uf_name": raw["D1N"].astype(str),
            "year": year,
            "quarter": quarter,
            # primeiro dia do trimestre
            "period": pd.PeriodIndex.from_fields(year=year, quarter=quarter, freq="Q").start_time,
            "rate": rate,
        }
    )

    missing = int(tidy["rate"].isna().sum())
    if missing:
        logger.warning("%d linhas sem valor numerico de taxa foram descartadas.", missing)

    tidy = tidy.dropna(subset=["rate"])
    if tidy.duplicated(subset=["uf_code", "period"]).any():
        raise ValueError("Ha mais de um valor para a mesma UF e trimestre.")

    return tidy.sort_values(["uf_code", "period"]).reset_index(drop=True)[TIDY_COLUMNS]
