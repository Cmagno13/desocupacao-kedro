import pandas as pd
import pytest

from desocupacao_kedro.pipelines.data_processing.nodes import TIDY_COLUMNS, clean_unemployment
from desocupacao_kedro.refresh import sidra_payload_to_frame


def _raw(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["D1C", "D1N", "D3C", "V"])


def test_clean_types_and_period():
    tidy = clean_unemployment(_raw([("35", "São Paulo", "202403", "6.2")]))

    assert list(tidy.columns) == TIDY_COLUMNS
    row = tidy.iloc[0]
    assert (row["uf_code"], row["uf_name"], row["year"], row["quarter"]) == (
        35,
        "São Paulo",
        2024,
        3,
    )
    assert row["period"] == pd.Timestamp("2024-07-01")
    assert row["rate"] == pytest.approx(6.2)


@pytest.mark.parametrize("marker", ["...", "-", "X", ""])
def test_clean_drops_missing_markers(marker, caplog):
    raw = _raw([("35", "São Paulo", "202401", marker), ("35", "São Paulo", "202402", "7.0")])

    with caplog.at_level("WARNING"):
        tidy = clean_unemployment(raw)

    assert tidy["rate"].tolist() == [7.0]
    assert "1 linhas sem valor" in caplog.text


def test_clean_sorts_by_uf_and_period():
    raw = _raw(
        [
            ("35", "São Paulo", "202402", "7.0"),
            ("11", "Rondônia", "202401", "3.0"),
            ("35", "São Paulo", "202401", "7.5"),
        ]
    )

    tidy = clean_unemployment(raw)

    assert tidy[["uf_code", "quarter"]].to_numpy().tolist() == [[11, 1], [35, 1], [35, 2]]


def test_clean_rejects_duplicate_uf_quarter():
    raw = _raw([("35", "São Paulo", "202401", "7.0"), ("35", "São Paulo", "202401", "7.1")])

    with pytest.raises(ValueError, match="mesma UF e trimestre"):
        clean_unemployment(raw)


def test_sidra_payload_drops_header_row():
    header = {
        "D1C": "UF (Código)",
        "D1N": "UF",
        "D3C": "Trimestre (Código)",
        "D3N": "T",
        "V": "Valor",
    }
    data = {
        "D1C": "35",
        "D1N": "São Paulo",
        "D3C": "202403",
        "D3N": "3º trimestre 2024",
        "V": "6.2",
    }

    frame = sidra_payload_to_frame([header, data])

    assert frame.to_dict("records") == [data]


def test_sidra_payload_without_data_fails():
    with pytest.raises(ValueError, match="sem linhas"):
        sidra_payload_to_frame([{"V": "Valor"}])
