from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from perceptron.core.paths import ProjectPaths
from perceptron.data.profiling.card import AlertCode, ProfileCard, sig
from perceptron.data.profiling.images import near_duplicate_pairs
from perceptron.data.profiling.profile import profile_dataset
from perceptron.data.versioning.ingest import IngestRequest, ingest
from perceptron.data.view import DatasetView
from perceptron.domain.enums import Modality, TaskType


def _profile(paths: ProjectPaths, src: Path, target: str | None = None) -> ProfileCard:
    v = ingest(paths, IngestRequest(project_id="prj_p", source=src, target=target))
    return profile_dataset(DatasetView(paths.dataset(v.content_hash)), content_hash=v.content_hash)


@pytest.fixture
def paths(workspace_dir: Path) -> ProjectPaths:
    return ProjectPaths(workspace_dir / "projects" / "prj_p").ensure()


def test_sig() -> None:
    assert sig(123456.789) == 123500
    assert sig(0.000123456) == 0.0001235
    assert sig(float("nan")) is None
    assert sig(0) == 0.0


def test_profile_churn(paths: ProjectPaths, fixtures_dir: Path) -> None:
    card = _profile(paths, fixtures_dir / "uc01_churn" / "churn.csv")
    assert card.modality is Modality.TABULAR
    assert card.num_samples == 400
    assert card.profiled_samples == card.split_counts["train"] + card.split_counts["val"]
    assert card.target is not None
    assert card.target.task_hint is TaskType.CLASSIFICATION
    assert {c.value for c in card.target.classes or []} == {"0", "1"}
    assert not card.alerts_by(AlertCode.CLASS_IMBALANCE)  # 20 % de churn: ratio ≈ 0,26 > 0,2
    cols = {c.name: c for c in card.columns}
    assert cols["edad"].null_fraction > 0
    assert cols["edad"].numeric is not None
    assert cols["tickets_90d"].target_association is not None
    assert cols["tickets_90d"].target_association > 0.1  # el fixture la hace predictiva
    assert card.alerts_by(AlertCode.ID_COLUMN)[0].column == "customer_id"
    assert not card.alerts_by(AlertCode.TARGET_LEAKAGE)


def test_leakage_and_constant_detected(paths: ProjectPaths, tmp_path: Path) -> None:
    n = 200
    df = pl.DataFrame(
        {
            "x": [float(i % 17) for i in range(n)],
            "constante": ["a"] * n,
            "fuga": [i % 2 for i in range(n)],  # idéntica al target
            "y": [i % 2 for i in range(n)],
        }
    )
    src = tmp_path / "leak.csv"
    df.write_csv(src)
    card = _profile(paths, src, target="y")
    leaks = card.alerts_by(AlertCode.TARGET_LEAKAGE)
    assert [a.column for a in leaks] == ["fuga"]
    assert [a.column for a in card.alerts_by(AlertCode.CONSTANT_COLUMN)] == ["constante"]


def test_imbalance_and_insufficient(paths: ProjectPaths, tmp_path: Path) -> None:
    df = pl.DataFrame({"x": list(range(60)), "y": ["raro"] * 3 + ["comun"] * 57})
    src = tmp_path / "imb.csv"
    df.write_csv(src)
    card = _profile(paths, src, target="y")
    assert card.alerts_by(AlertCode.CLASS_IMBALANCE)
    assert card.alerts_by(AlertCode.INSUFFICIENT_DATA)


def test_profile_images(paths: ProjectPaths, fixtures_dir: Path) -> None:
    card = _profile(paths, fixtures_dir / "uc04_defects")
    assert card.images is not None
    assert card.images.corrupt == 0
    assert [r.value for r in card.images.resolutions] == ["32x32"]
    assert card.target is not None
    assert {c.value for c in card.target.classes or []} == {"ok", "defect"}


def test_near_duplicate_pairs() -> None:
    a = 0xFFFF_0000_FFFF_0000
    hashes = [a, a ^ 0b111, a ^ 0xFFFF, 0x1234_5678_9ABC_DEF0]
    assert near_duplicate_pairs(hashes) == [(0, 1)]


# ------------------------------------------------------------------ privacidad L1

# Múltiplos de 10 000 coinciden con su propio redondeo a 4 cifras (un agregado, no un valor).
_marker = st.integers(min_value=10_000_000, max_value=99_999_999).filter(lambda m: m % 10_000)


@settings(
    max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(seed=st.integers(0, 10_000), marker=_marker)
def test_card_contains_no_individual_values(
    tmp_path_factory: pytest.TempPathFactory, seed: int, marker: int
) -> None:
    """Ningún valor individual (número exacto de alta precisión o texto único) llega al card."""
    import random

    rng = random.Random(seed)  # noqa: S311 - datos de prueba
    n = 80
    nums = [marker + rng.random() for _ in range(n)]  # p. ej. 12345678.83412
    secrets = [f"secreto-{marker}-{i}" for i in range(n)]
    df = pl.DataFrame(
        {
            "monto": nums,
            "nota": secrets,  # texto/ids únicos
            "email": [f"u{i}.{marker}@empresa.com" for i in range(n)],
            "categoria": [rng.choice(["a", "b", "c"]) for _ in range(n)],
            "y": [i % 2 for i in range(n)],
        }
    )
    root = tmp_path_factory.mktemp("priv")
    src = root / "datos.csv"
    df.write_csv(src)
    paths = ProjectPaths(root / "prj").ensure()
    card_json = _profile(paths, src, target="y").model_dump_json()

    for value in nums:
        assert repr(value) not in card_json
        assert f"{value:.6f}" not in card_json
    for s in secrets:
        assert s not in card_json
    assert "@empresa.com" not in card_json
    assert str(marker) not in card_json
