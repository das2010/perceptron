"""Diff de configuración entre runs (RF-TRK-04)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from perceptron.api.context import EngineContext
from perceptron.domain.models import ArchSpecRecord, Pipeline, Project, Run
from perceptron.services.compare import MISSING, config_diff, flatten


def test_flatten_indexes_lists_of_nodes_by_id() -> None:
    spec = {"nodes": [{"id": "enc", "params": {"hidden": 64}}], "tags": ["a", "b"], "empty": {}}
    assert flatten(spec) == {
        "empty": {},
        "nodes[enc].id": "enc",
        "nodes[enc].params.hidden": 64,
        "tags[0]": "a",
        "tags[1]": "b",
    }


def test_only_differences_and_reordering_is_not_a_change() -> None:
    a = {"nodes": [{"id": "enc", "p": 64}, {"id": "head", "p": 2}], "epochs": 30}
    b = {"nodes": [{"id": "head", "p": 2}, {"id": "enc", "p": 128}], "epochs": 30, "extra": True}
    diff = config_diff(["r1", "r2"], [a, b], [{"steps": []}, {"steps": []}])
    assert [(r.path, r.values) for r in diff.archspec] == [
        ("extra", [MISSING, True]),
        ("nodes[enc].p", [64, 128]),
    ]
    assert not diff.same_archspec and diff.same_pipeline and diff.pipeline == []


def test_compare_configs_endpoint(client: TestClient, ctx: EngineContext) -> None:
    project = ctx.projects.add(Project(name="Comparar"))
    runs = []
    for hidden, impute in ((64, "median"), (128, "mean")):
        arch = ctx.repo(ArchSpecRecord).add(
            ArchSpecRecord(
                project_id=project.id,
                name=f"mlp-{hidden}",
                spec={"nodes": [{"id": "enc", "params": {"hidden": hidden}}]},
                content_hash=f"h{hidden}",
            )
        )
        pipe = ctx.repo(Pipeline).add(
            Pipeline(
                project_id=project.id, name="p", graph={"steps": [{"id": "imp", "op": impute}]}
            )
        )
        runs.append(
            ctx.repo(Run).add(
                Run(
                    project_id=project.id,
                    archspec_id=arch.id,
                    pipeline_id=pipe.id,
                    dataset_version_id="dsv_x",
                )
            )
        )
    r = client.post("/api/v1/runs/compare/config", json={"run_ids": [x.id for x in runs]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["archspec"] == [{"path": "nodes[enc].params.hidden", "values": [64, 128]}]
    assert body["pipeline"] == [{"path": "steps[imp].op", "values": ["median", "mean"]}]
