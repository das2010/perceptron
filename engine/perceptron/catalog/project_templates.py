"""Plantillas de proyecto por caso de uso (RF-PRJ-02, SPEC §5): UC-01…UC-09.

Preconfiguran modalidad, tarea y métrica objetivo. La métrica es una de las que el
entrenamiento registra en validación (`val_*`), así el HPO puede optimizarla. Los textos
(título, descripción, ejemplo de objetivo) viven en la UI (i18n es/en).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from perceptron.core.errors import ValidationError
from perceptron.domain.enums import Modality, TaskType


class ProjectTemplate(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    use_case: str
    modalities: list[Modality]
    task: TaskType
    target_metric: str


PROJECT_TEMPLATES: tuple[ProjectTemplate, ...] = (
    ProjectTemplate(
        id="churn",
        use_case="UC-01",
        modalities=[Modality.TABULAR],
        task=TaskType.CLASSIFICATION,
        target_metric="val_auroc",
    ),
    ProjectTemplate(
        id="ticket-resolution-time",
        use_case="UC-02",
        modalities=[Modality.TABULAR, Modality.TEXT],
        task=TaskType.REGRESSION,
        target_metric="val_mae",
    ),
    ProjectTemplate(
        id="ticket-category",
        use_case="UC-03",
        modalities=[Modality.TEXT],
        task=TaskType.CLASSIFICATION,
        target_metric="val_f1_macro",
    ),
    ProjectTemplate(
        id="visual-inspection",
        use_case="UC-04",
        modalities=[Modality.IMAGE],
        task=TaskType.CLASSIFICATION,
        target_metric="val_f1_macro",
    ),
    ProjectTemplate(
        id="damage-segmentation",
        use_case="UC-05",
        modalities=[Modality.IMAGE],
        task=TaskType.SEGMENTATION,
        target_metric="val_iou",
    ),
    ProjectTemplate(
        id="document-ocr",
        use_case="UC-06",
        modalities=[Modality.IMAGE],
        task=TaskType.OCR,
        target_metric="val_loss",
    ),
    ProjectTemplate(
        id="demand-forecast",
        use_case="UC-07",
        modalities=[Modality.TIMESERIES],
        task=TaskType.FORECASTING,
        target_metric="val_mae",
    ),
    ProjectTemplate(
        id="sensor-anomalies",
        use_case="UC-08",
        modalities=[Modality.TIMESERIES],
        task=TaskType.ANOMALY_DETECTION,
        target_metric="val_loss",
    ),
    ProjectTemplate(
        id="machine-sound",
        use_case="UC-09",
        modalities=[Modality.AUDIO],
        task=TaskType.CLASSIFICATION,
        target_metric="val_f1_macro",
    ),
)

_BY_ID = {t.id: t for t in PROJECT_TEMPLATES}
_BY_USE_CASE = {t.use_case.lower(): t for t in PROJECT_TEMPLATES}


def get_template(template_id: str) -> ProjectTemplate:
    """Por id (`churn`) o por caso de uso (`UC-01`)."""
    template = _BY_ID.get(template_id) or _BY_USE_CASE.get(template_id.lower())
    if template is None:
        raise ValidationError(
            f"plantilla desconocida: {template_id}", details={"templates": sorted(_BY_ID)}
        )
    return template
