from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from perceptron.core.errors import ValidationError
from perceptron.core.paths import ProjectPaths
from perceptron.data.schema import SemanticType, infer_schema
from perceptron.data.sources.files import SourceKind, open_source, scan_table
from perceptron.data.splits import SPLIT_COLUMN, SplitRequest, assign_splits, split_counts
from perceptron.data.versioning.ingest import IngestRequest, ingest
from perceptron.data.view import DatasetView, Purpose, SealedTestSetError
from perceptron.domain.enums import Modality, SplitStrategy

# ------------------------------------------------------------------ esquema


def test_infer_schema_churn(fixtures_dir: Path) -> None:
    df = scan_table(fixtures_dir / "uc01_churn" / "churn.csv").collect()
    schema = infer_schema(df)
    sem = {c.name: c.semantic for c in schema.columns}
    assert sem["customer_id"] is SemanticType.ID
    assert sem["edad"] is SemanticType.NUMERIC
    assert schema.column("edad").nullable
    assert sem["region"] is SemanticType.CATEGORICAL
    assert sem["plan"] is SemanticType.CATEGORICAL
    assert sem["cargo_mensual"] is SemanticType.NUMERIC
    assert sem["churn"] is SemanticType.BOOLEAN
    assert schema.target_candidates[0] == "churn"


def test_infer_schema_text_and_dates(fixtures_dir: Path) -> None:
    tickets = scan_table(fixtures_dir / "uc02_tickets_time" / "tickets.csv").collect()
    sem = {c.name: c.semantic for c in infer_schema(tickets).columns}
    assert sem["categoria"] is SemanticType.CATEGORICAL
    assert sem["horas_resolucion"] is SemanticType.NUMERIC

    demand = scan_table(fixtures_dir / "uc07_demand" / "demanda.csv").collect()
    sem = {c.name: c.semantic for c in infer_schema(demand).columns}
    assert sem["semana"] is SemanticType.DATETIME
    assert sem["sku"] is SemanticType.CATEGORICAL


def test_infer_schema_rejects_unknown_target() -> None:
    with pytest.raises(KeyError):
        infer_schema(pl.DataFrame({"a": [1, 2]}), target="b")


# ------------------------------------------------------------------ lectores


def test_readers_all_formats(tmp_path: Path) -> None:
    df = pl.DataFrame({"x": [1, 2, 3], "ciudad": ["Córdoba", "Salta", "Paraná"]})
    df.write_csv(tmp_path / "a.csv")
    df.write_csv(tmp_path / "b.tsv", separator="\t")
    df.write_csv(tmp_path / "c.csv", separator=";")
    df.write_parquet(tmp_path / "d.parquet")
    df.write_ndjson(tmp_path / "e.jsonl")
    df.write_json(tmp_path / "f.json")
    df.write_excel(tmp_path / "g.xlsx")
    (tmp_path / "h.csv").write_bytes("x,ciudad\n1,Córdoba\n".encode("cp1252"))
    for name in ("a.csv", "b.tsv", "c.csv", "d.parquet", "e.jsonl", "f.json", "g.xlsx"):
        out = scan_table(tmp_path / name).collect()
        assert out.columns == ["x", "ciudad"], name
        assert out["ciudad"].to_list() == ["Córdoba", "Salta", "Paraná"], name
    assert scan_table(tmp_path / "h.csv").collect()["ciudad"].to_list() == ["Córdoba"]


def test_unsupported_and_missing(tmp_path: Path) -> None:
    (tmp_path / "x.docx").write_bytes(b"")
    with pytest.raises(ValidationError), open_source(tmp_path / "x.docx"):
        pass
    with pytest.raises(ValidationError), open_source(tmp_path / "nada.csv"):
        pass


def test_zip_with_image_folder(tmp_path: Path, fixtures_dir: Path) -> None:
    z = tmp_path / "piezas.zip"
    src = fixtures_dir / "uc04_defects"
    with zipfile.ZipFile(z, "w") as zf:
        for p in src.rglob("*.png"):
            zf.write(p, Path("uc04") / p.relative_to(src))
    with open_source(z) as detected:
        assert detected.kind is SourceKind.IMAGE_FOLDER


# ------------------------------------------------------------------ splits


def _df(n: int, classes: int = 2) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "y": [i % classes for i in range(n)],
            "g": [i // 5 for i in range(n)],
            "t": list(range(n)),
        }
    )


@settings(max_examples=40, deadline=None)
@given(
    n=st.integers(min_value=30, max_value=400),
    classes=st.integers(min_value=2, max_value=4),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_stratified_split_properties(n: int, classes: int, seed: int) -> None:
    req = SplitRequest(strategy=SplitStrategy.STRATIFIED, seed=seed)
    out = assign_splits(_df(n, classes), req, "y")
    counts = split_counts(out)
    assert sum(counts.values()) == n
    assert abs(counts["test"] - round(n * 0.15)) <= classes
    # Estratificación: cada clase aparece en train
    assert out.filter(pl.col(SPLIT_COLUMN) == "train")["y"].n_unique() == classes
    # Determinismo
    assert assign_splits(_df(n, classes), req, "y")[SPLIT_COLUMN].equals(out[SPLIT_COLUMN])


def test_group_split_keeps_groups_together() -> None:
    out = assign_splits(
        _df(200), SplitRequest(strategy=SplitStrategy.GROUP, group_column="g"), None
    )
    per_group = out.group_by("g").agg(pl.col(SPLIT_COLUMN).n_unique().alias("k"))
    assert per_group["k"].max() == 1


def test_temporal_split_puts_latest_in_test() -> None:
    df = _df(100).sample(fraction=1.0, shuffle=True, seed=1)
    out = assign_splits(df, SplitRequest(strategy=SplitStrategy.TEMPORAL, time_column="t"), None)
    test_min = out.filter(pl.col(SPLIT_COLUMN) == "test")["t"].min()
    val = out.filter(pl.col(SPLIT_COLUMN) == "val")["t"]
    train_max = out.filter(pl.col(SPLIT_COLUMN) == "train")["t"].max()
    assert train_max < val.min() <= val.max() < test_min


def test_kfold_assigns_folds_outside_test() -> None:
    out = assign_splits(_df(100), SplitRequest(strategy=SplitStrategy.KFOLD, folds=5), "y")
    non_test = out.filter(pl.col(SPLIT_COLUMN) != "test")
    assert sorted(non_test["__fold__"].unique().to_list()) == [0, 1, 2, 3, 4]
    assert out.filter(pl.col(SPLIT_COLUMN) == "test")["__fold__"].unique().to_list() == [-1]


def test_split_request_validation() -> None:
    with pytest.raises(ValueError, match="group_column"):
        SplitRequest(strategy=SplitStrategy.GROUP)
    with pytest.raises(ValueError, match="< 1"):
        SplitRequest(val_fraction=0.6, test_fraction=0.5)


# ------------------------------------------------------------------ ingesta


@pytest.fixture
def project_paths(workspace_dir: Path) -> ProjectPaths:
    return ProjectPaths(workspace_dir / "projects" / "prj_test").ensure()


def test_ingest_tabular_is_content_addressed(
    project_paths: ProjectPaths, fixtures_dir: Path
) -> None:
    src = fixtures_dir / "uc01_churn" / "churn.csv"
    v1 = ingest(project_paths, IngestRequest(project_id="prj_test", source=src))
    assert v1.modality is Modality.TABULAR
    assert v1.target == "churn"
    assert v1.num_samples == 400
    assert v1.split is not None
    assert v1.split.train + v1.split.val + v1.split.test == 400

    # Misma fuente → mismo hash y mismo directorio (idempotente)
    v2 = ingest(project_paths, IngestRequest(project_id="prj_test", source=src))
    assert v2.content_hash == v1.content_hash
    assert [p.name for p in project_paths.datasets_dir.iterdir()] == [v1.content_hash]

    # Otro split → otra versión
    v3 = ingest(
        project_paths,
        IngestRequest(project_id="prj_test", source=src, split=SplitRequest(seed=7)),
    )
    assert v3.content_hash != v1.content_hash

    view = DatasetView(project_paths.dataset(v1.content_hash))
    assert view.target == "churn"
    assert view.read().height == v1.split.train + v1.split.val
    assert view.read("train").height == v1.split.train


def test_ingest_images(project_paths: ProjectPaths, fixtures_dir: Path) -> None:
    v = ingest(
        project_paths, IngestRequest(project_id="prj_test", source=fixtures_dir / "uc04_defects")
    )
    assert v.modality is Modality.IMAGE
    assert v.target == "label"
    assert v.num_samples == 40
    view = DatasetView(project_paths.dataset(v.content_hash))
    index = view.read(purpose=Purpose.FINAL_EVALUATION)
    assert set(index["label"].unique()) == {"ok", "defect"}
    assert index["width"].unique().to_list() == [32]
    assert not index["corrupt"].any()
    first = index["path"][0]
    assert (view.files_dir / first).is_file()


def test_corrupt_image_is_flagged(
    project_paths: ProjectPaths, fixtures_dir: Path, tmp_path: Path
) -> None:
    src = tmp_path / "imgs"
    shutil.copytree(fixtures_dir / "uc04_defects", src, ignore=shutil.ignore_patterns("*.json"))
    (src / "ok" / "rota.png").write_bytes(b"\x89PNG no es una imagen")
    v = ingest(project_paths, IngestRequest(project_id="prj_test", source=src))
    index = DatasetView(project_paths.dataset(v.content_hash)).read(
        purpose=Purpose.FINAL_EVALUATION
    )
    assert index.filter(pl.col("corrupt"))["path"].to_list() == ["ok/rota.png"]


def test_test_set_is_sealed(project_paths: ProjectPaths, fixtures_dir: Path) -> None:
    v = ingest(
        project_paths,
        IngestRequest(project_id="prj_test", source=fixtures_dir / "uc01_churn" / "churn.csv"),
    )
    view = DatasetView(project_paths.dataset(v.content_hash))
    with pytest.raises(SealedTestSetError):
        view.read("test")
    with pytest.raises(SealedTestSetError):
        view.read("test", purpose=Purpose.PROFILING)
    assert v.split is not None
    assert view.read("test", purpose=Purpose.FINAL_EVALUATION).height == v.split.test
    assert "test" not in view.read()[SPLIT_COLUMN].unique().to_list()


def test_failed_ingest_leaves_no_staging(project_paths: ProjectPaths, tmp_path: Path) -> None:
    empty = tmp_path / "vacio.csv"
    empty.write_text("a,b\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        ingest(project_paths, IngestRequest(project_id="prj_test", source=empty))
    assert list(project_paths.datasets_dir.iterdir()) == []
