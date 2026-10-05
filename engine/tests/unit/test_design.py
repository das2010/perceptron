"""Requisitos de diseño por escenario y evaluación de propuestas (ADR-0041)."""

from __future__ import annotations

from typing import Any

from perceptron.catalog.templates import image_template, tabular_template
from perceptron.data.profiling.card import ProfileCard, TargetProfile
from perceptron.data.schema import SemanticType
from perceptron.domain.enums import Modality, TaskType
from perceptron.services.design import (
    DesignRequirements,
    assess,
    design_requirements,
    missing_any_musts,
    recommend_index,
    supplement_spec,
    unmet_musts,
)


def _card(
    modality: Modality, n_train: int, imbalance: float | None = None, task: TaskType | None = None
) -> ProfileCard:
    target = None
    if imbalance is not None or task is not None:
        target = TargetProfile(
            name="y",
            semantic=SemanticType.CATEGORICAL,
            task_hint=task or TaskType.CLASSIFICATION,
            imbalance_ratio=imbalance,
        )
    return ProfileCard(
        modality=modality,
        num_samples=n_train,
        profiled_samples=n_train,
        split_counts={"train": n_train},
        num_features=1,
        target=target,
        columns=[],
    )


def _reqs(
    modality: Modality,
    n_train: int,
    task: TaskType = TaskType.CLASSIFICATION,
    use_case: dict[str, Any] | None = None,
    **kw: Any,
) -> DesignRequirements:
    kw.setdefault("device", "cpu")
    kw.setdefault("allow_pretrained", True)
    return design_requirements(
        _card(modality, n_train, kw.pop("imbalance", None)), task, use_case, **kw
    )


def _codes(reqs: DesignRequirements) -> dict[str, str]:
    return {r.code: r.level for r in reqs.items}


def _img(backbone: str, pretrained: bool = True) -> Any:
    return image_template(
        backbone, task=TaskType.CLASSIFICATION, num_classes=3, image_size=128, pretrained=pretrained
    )


def test_few_images_require_a_pretrained_option_tubos() -> None:
    reqs = _reqs(Modality.IMAGE, 300, image_size=128)
    assert _codes(reqs)["pretrained_backbone"] == "must"
    scratch, mobile = _img("small_cnn", pretrained=False), _img("mobilenetv3_small_100")
    # El arquitecto propuso solo redes desde cero: se le pide una preentrenada.
    [msg] = unmet_musts(reqs, [("CNN compacta", scratch, 50_000.0)])
    assert "pretrained_backbone" in msg
    assert not unmet_musts(reqs, [("CNN", scratch, 5e4), ("MobileNet", mobile, 1.5e6)])
    # La preentrenada queda recomendada aunque el arquitecto la haya puesto segunda.
    scores = [assess(reqs, scratch, 5e4, 10.0, 0.8), assess(reqs, mobile, 1.5e6, 40.0, 0.6)]
    assert recommend_index(scores) == 1
    assert not scores[0].checks[0].met and scores[1].checks[0].met


def test_pretrained_requirement_depends_on_size_connection_and_amount() -> None:
    assert _codes(_reqs(Modality.IMAGE, 3_000, image_size=128))["pretrained_backbone"] == "should"
    assert _codes(_reqs(Modality.IMAGE, 300, image_size=32))["pretrained_backbone"] == "should"
    assert "pretrained_backbone" not in _codes(_reqs(Modality.IMAGE, 50_000))
    offline = _codes(_reqs(Modality.IMAGE, 300, allow_pretrained=False))
    assert "pretrained_backbone" not in offline and offline["no_pretrained_offline"] == "should"
    # Texto: solo si el pipeline tokeniza para un encoder preentrenado.
    assert "pretrained_backbone" not in _codes(_reqs(Modality.TEXT, 500))
    assert _codes(_reqs(Modality.TEXT, 500, pretrained_text=True))["pretrained_backbone"]


def test_rule_or_extrapolation_requires_a_linear_option_tabla3() -> None:
    reqs = _reqs(Modality.TABULAR, 80, TaskType.REGRESSION, {"problem": "rule"})
    assert _codes(reqs)["linear_option"] == "must"
    small = _codes(reqs)["small_model"]
    assert small == "should"
    mlp = tabular_template(
        "mlp", task=TaskType.REGRESSION, num_classes=None, num_numeric=1, cardinalities=[]
    )
    [missing] = missing_any_musts(reqs, [mlp])
    title, linear = supplement_spec(missing, mlp, "cpu") or ("", mlp)
    assert title and [n.block for n in linear.nodes] == ["input.tabular", "head.linear"]
    assert linear.optimizer.weight_decay == 0.0 and linear.input == mlp.input
    assert not missing_any_musts(reqs, [mlp, linear])
    scores = [assess(reqs, mlp, 4_000.0, 1.0, 0.9), assess(reqs, linear, 2.0, 1.0, 0.5)]
    assert recommend_index(scores) == 1
    # Sin ficha que lo pida, no hay requisito lineal; en clasificación tampoco.
    assert "linear_option" not in _codes(_reqs(Modality.TABULAR, 80, TaskType.REGRESSION))
    cls = _reqs(Modality.TABULAR, 80, TaskType.CLASSIFICATION, {"problem": "rule"})
    assert "linear_option" not in _codes(cls)


def test_small_tables_cap_parameters() -> None:
    reqs = _reqs(Modality.TABULAR, 1_000, TaskType.REGRESSION)
    [cap] = [r for r in reqs.items if r.code == "small_model"]
    assert cap.params["max_params"] == 20_000
    spec = tabular_template(
        "ft_transformer",
        task=TaskType.REGRESSION,
        num_classes=None,
        num_numeric=4,
        cardinalities=[],
    )
    big, small = assess(reqs, spec, 250_000.0, 5.0), assess(reqs, spec, 8_000.0, 5.0)
    assert big.score < small.score
    assert "small_model" not in _codes(_reqs(Modality.TABULAR, 50_000, TaskType.REGRESSION))


def test_edge_deployment_caps_every_proposal() -> None:
    reqs = _reqs(Modality.IMAGE, 50_000, use_case={"deployment": "edge"})
    assert _codes(reqs)["edge_size"] == "must"
    big = _img("convnext_tiny")
    errors = unmet_musts(reqs, [("ConvNeXt", big, 28e6), ("MobileNet", _img("small_cnn"), 1e6)])
    assert len(errors) == 1 and "ConvNeXt" in errors[0]


def test_imbalance_and_error_costs_need_compensation() -> None:
    reqs = _reqs(Modality.TABULAR, 50_000, imbalance=0.05)
    assert _codes(reqs)["imbalance_handling"] == "should"
    spec = tabular_template(
        "mlp", task=TaskType.CLASSIFICATION, num_classes=2, num_numeric=3, cardinalities=[]
    )
    assert assess(reqs, spec, 1e4, 1.0).checks[0].met  # las plantillas ya pesan las clases
    spec.loss.class_weights = "none"
    assert not assess(reqs, spec, 1e4, 1.0).checks[0].met
    spec.training.oversample = True
    assert assess(reqs, spec, 1e4, 1.0).checks[0].met
    costs = _reqs(Modality.TABULAR, 50_000, use_case={"error_costs": "false_negative_worse"})
    assert "imbalance_handling" in _codes(costs)


def test_epoch_time_budget_depends_on_device() -> None:
    cpu = _reqs(Modality.TABULAR, 50_000, device="cpu")
    gpu = _reqs(Modality.TABULAR, 50_000, device="cuda")
    [c] = [r for r in cpu.items if r.code == "epoch_time"]
    [g] = [r for r in gpu.items if r.code == "epoch_time"]
    assert c.params["max_s"] > g.params["max_s"]
    spec = _img("small_cnn", pretrained=False)
    assert assess(cpu, spec, 1e5, 900.0).score < assess(cpu, spec, 1e5, 30.0).score
    assert assess(cpu, spec, 1e5, None).checks[-1].met  # sin estimación no se penaliza


def test_requirements_are_deterministic_and_serializable() -> None:
    a = _reqs(Modality.IMAGE, 300, use_case={"deployment": "edge", "explainability": True})
    b = _reqs(Modality.IMAGE, 300, use_case={"deployment": "edge", "explainability": True})
    assert a == b and a.for_llm() == b.for_llm()
    assert all({"code", "level", "scope", "message"} <= set(r) for r in a.for_llm())
    assert a.scenario["n_train"] == 300 and a.scenario["deployment"] == "edge"
    # Una ficha inválida no rompe: se ignora.
    assert _reqs(Modality.TABULAR, 80, use_case={"deployment": "luna"}).items
