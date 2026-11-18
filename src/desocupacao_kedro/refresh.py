"""Atualiza o snapshot bruto da tabela SIDRA 4099 (taxa de desocupacao por UF).

Uso: ``uv run python -m desocupacao_kedro.refresh``

O pipeline nunca acessa a rede: ele le apenas o CSV versionado em ``data/01_raw``.
Este modulo existe para regravar esse CSV quando o IBGE publica um novo trimestre.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

SIDRA_URL = "https://apisidra.ibge.gov.br/values/t/4099/n3/all/v/4099/p/all"
RAW_COLUMNS = ["D1C", "D1N", "D3C", "D3N", "V"]
SNAPSHOT_PATH = Path(__file__).resolve().parents[2] / "data" / "01_raw" / "sidra_4099_uf.csv"


def sidra_payload_to_frame(payload: list[dict[str, str]]) -> pd.DataFrame:
    """Converte a resposta JSON do SIDRA em tabela, sem interpretar os valores.

    O primeiro elemento da resposta e uma linha de cabecalho (descricao dos campos)
    e e descartado. Os valores ficam como texto, exatamente como vieram da API.
    """
    rows = payload[1:]
    if not rows:
        raise ValueError("Resposta do SIDRA sem linhas de dados.")
    return pd.DataFrame(rows, columns=RAW_COLUMNS).astype(str)


def main() -> None:
    with urllib.request.urlopen(SIDRA_URL, timeout=60) as response:  # noqa: S310
        payload = json.load(response)
    frame = sidra_payload_to_frame(payload)
    frame.to_csv(SNAPSHOT_PATH, index=False)
    print(f"{len(frame)} linhas gravadas em {SNAPSHOT_PATH}")


if __name__ == "__main__":
    main()
