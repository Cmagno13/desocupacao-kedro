"""Executa o pipeline completo sobre o snapshot versionado em data/01_raw."""

import json
import shutil
from pathlib import Path

import pytest
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project_copy(tmp_path: Path) -> Path:
    """Copia minima do projeto, para o teste nao sobrescrever as saidas em data/."""
    for name in ("pyproject.toml", "conf", "data/01_raw"):
        source, target = PROJECT_ROOT / name, tmp_path / name
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy(source, target)
    (tmp_path / "src").mkdir()
    return tmp_path


def test_full_pipeline_on_snapshot(project_copy: Path):
    bootstrap_project(PROJECT_ROOT)
    with KedroSession.create(project_path=project_copy) as session:
        session.run()

    reporting = project_copy / "data" / "08_reporting"
    metrics = json.loads((reporting / "metrics.json").read_text(encoding="utf-8"))

    assert metrics["test"]["ufs"] == 27
    assert metrics["test"]["rows"] == 27 * 8
    assert metrics["train"]["last_period"] < metrics["test"]["first_period"]
    assert set(metrics["mae"]) == {
        "model",
        "naive_last_quarter",
        "naive_same_quarter_last_year",
    }
    assert all(0 < value < 5 for value in metrics["mae"].values())
    assert metrics["model_beats_naive_last_quarter"] == (
        metrics["mae"]["model"] < metrics["mae"]["naive_last_quarter"]
    )
    assert (reporting / "actual_vs_predicted.png").stat().st_size > 10_000
    assert (project_copy / "data" / "06_models" / "model.pkl").exists()
