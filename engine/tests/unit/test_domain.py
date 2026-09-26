from __future__ import annotations

import pydantic
import pytest

from perceptron.core.ids import IdPrefix, is_valid_id
from perceptron.domain.enums import LLMPurpose, PrivacyLevel, ProjectScope, SplitStrategy
from perceptron.domain.models import (
    ALL_ENTITIES,
    ArchSpecRecord,
    DatasetVersion,
    LLMCall,
    Project,
    Split,
)


def test_project_defaults() -> None:
    p = Project(name="Churn")
    assert is_valid_id(p.id, IdPrefix.PROJECT)
    assert p.privacy_level is PrivacyLevel.L1  # default SPEC §7.7.3
    assert p.scope is ProjectScope.LOCAL
    assert p.version == 1


def test_extra_fields_forbidden() -> None:
    with pytest.raises(pydantic.ValidationError):
        Project(name="x", unknown=1)  # type: ignore[call-arg]


def test_validate_assignment() -> None:
    p = Project(name="x")
    with pytest.raises(pydantic.ValidationError):
        p.name = ""


def test_dataset_version_requires_sha256() -> None:
    with pytest.raises(pydantic.ValidationError):
        DatasetVersion(project_id="prj_1", content_hash="abc", num_samples=1)
    dv = DatasetVersion(
        project_id="prj_1",
        content_hash="a" * 64,
        num_samples=10,
        split=Split(strategy=SplitStrategy.TEMPORAL, train=7, val=2, test=1, time_column="fecha"),
    )
    assert dv.split is not None
    assert dv.split.test_sealed


def test_privacy_rank_ordering() -> None:
    assert [lvl.rank for lvl in PrivacyLevel] == [0, 1, 2, 3]


def test_archspec_record_declarative_flag() -> None:
    a = ArchSpecRecord(project_id="prj_1", name="mlp", spec={"nodes": []}, content_hash="h")
    b = ArchSpecRecord(project_id="prj_1", name="code", code_path="code/m.py", content_hash="h")
    assert a.is_declarative
    assert not b.is_declarative


@pytest.mark.parametrize("model", ALL_ENTITIES, ids=lambda m: m.__name__)
def test_every_entity_exports_json_schema(model: type[pydantic.BaseModel]) -> None:
    schema = model.model_json_schema()
    assert schema["type"] == "object"
    assert "id" in schema["properties"]


def test_llm_call_json_roundtrip() -> None:
    call = LLMCall(
        project_id="prj_1",
        purpose=LLMPurpose.ARCHITECT,
        provider="anthropic",
        model="configured-model",
        privacy_level=PrivacyLevel.L1,
        payload={"profile": {"rows": 1000}},
    )
    again = LLMCall.model_validate_json(call.model_dump_json())
    assert again == call
