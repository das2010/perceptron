"""Detección (CenterNet), segmentación (U-Net) y OCR (CRNN + CTC) — RF-EVL-01 visión avanzada."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn

from perceptron.domain.enums import TaskType
from perceptron.tasks.base import Predictions, Resolver, StepOutput, TaskAdapter, TaskEvaluation

if TYPE_CHECKING:
    from perceptron.archspec.schema import ArchSpec
    from perceptron.data.pipeline.pipeline import FittedPipeline

SCORE_THRESHOLD = 0.05
MAX_DETECTIONS = 50


# ------------------------------------------------------------------ detección


def centernet_targets(
    targets: list[dict[str, torch.Tensor]], k: int, h: int, w: int, stride: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Heatmaps gaussianos [B,K,h,w], regresión [B,4,h,w] y máscara [B,h,w] de centros."""
    b = len(targets)
    heat = torch.zeros(b, k, h, w)
    reg = torch.zeros(b, 4, h, w)
    mask = torch.zeros(b, h, w)
    ys, xs = torch.meshgrid(
        torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij"
    )
    for i, t in enumerate(targets):
        for box, label in zip(t["boxes"], t["labels"], strict=True):
            x1, y1, x2, y2 = (float(v) / stride for v in box)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            ix, iy = min(int(cx), w - 1), min(int(cy), h - 1)
            bw, bh = max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)
            sigma = max(max(bw, bh) / 6, 0.5)
            g = torch.exp(-((xs - ix) ** 2 + (ys - iy) ** 2) / (2 * sigma**2))
            heat[i, int(label)] = torch.maximum(heat[i, int(label)], g)
            reg[i, :, iy, ix] = torch.tensor([bw, bh, cx - ix, cy - iy])
            mask[i, iy, ix] = 1
    return heat, reg, mask, torch.tensor(stride)


def focal_heatmap_loss(logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Focal loss de CenterNet (Law & Deng, 2018)."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    pos_loss = -torch.log(p) * (1 - p) ** 2 * pos
    neg_loss = -torch.log(1 - p) * p**2 * (1 - gt) ** 4 * (1 - pos)
    return (pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1)


class CenterNetLoss(nn.Module):
    def __init__(self, num_classes: int, image_size: int, size_weight: float = 0.1) -> None:
        super().__init__()
        self.k = num_classes
        self.image_size = image_size
        self.size_weight = size_weight

    def forward(self, out: torch.Tensor, targets: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        _, _, h, w = out.shape
        heat, reg, mask, _ = centernet_targets(targets, self.k, h, w, self.image_size / w)
        heat, reg, mask = heat.to(out.device), reg.to(out.device), mask.to(out.device)
        loss = focal_heatmap_loss(out[:, : self.k], heat)
        m = mask.unsqueeze(1)
        n = mask.sum().clamp(min=1)
        pred_reg = out[:, self.k :]
        size_l1 = (F.l1_loss(pred_reg[:, :2], reg[:, :2], reduction="none") * m).sum() / n
        off_l1 = (F.l1_loss(pred_reg[:, 2:], reg[:, 2:], reduction="none") * m).sum() / n
        return loss + self.size_weight * size_l1 + off_l1


def decode_centernet(out: torch.Tensor, k: int, image_size: int) -> list[dict[str, torch.Tensor]]:
    b, _, h, w = out.shape
    stride = image_size / w
    heat = torch.sigmoid(out[:, :k])
    peaks = heat * (heat == F.max_pool2d(heat, 3, stride=1, padding=1)).float()
    results = []
    for i in range(b):
        scores, idx = peaks[i].flatten().topk(min(MAX_DETECTIONS, peaks[i].numel()))
        keep = scores > SCORE_THRESHOLD
        scores, idx = scores[keep], idx[keep]
        labels = idx // (h * w)
        pix = idx % (h * w)
        ys, xs = (pix // w).float(), (pix % w).float()
        reg = out[i, k:].flatten(1)[:, pix]
        bw, bh, dx, dy = reg[0].clamp(min=0), reg[1].clamp(min=0), reg[2], reg[3]
        cx, cy = (xs + dx) * stride, (ys + dy) * stride
        boxes = torch.stack(
            [
                cx - bw * stride / 2,
                cy - bh * stride / 2,
                cx + bw * stride / 2,
                cy + bh * stride / 2,
            ],
            dim=1,
        )
        results.append({"boxes": boxes.clamp(0, image_size), "scores": scores, "labels": labels})
    return results


def detection_collate(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    images = torch.stack([b[0] for b in batch])
    return images, [b[1] for b in batch]


class DetectionAdapter(TaskAdapter):
    task: ClassVar[TaskType] = TaskType.OBJECT_DETECTION

    def num_outputs(self, spec: ArchSpec) -> int:
        if not spec.task.num_classes:
            raise ValueError("task.num_classes es requerido para detección")
        return spec.task.num_classes

    def check_output(self, kind: str, shape: tuple[int, ...], spec: ArchSpec) -> str | None:
        k = self.num_outputs(spec)
        if kind != "feature_map" or shape[0] != k + 4:
            return f"la salida debe ser un mapa [{k} clases + 4, h, w] y es {kind} {list(shape)}"
        return None

    def build_loss(
        self, spec: ArchSpec, resolve: Resolver, class_weights: torch.Tensor | None
    ) -> nn.Module:
        return CenterNetLoss(self.num_outputs(spec), int((spec.input.shape or [3, 64, 64])[-1]))

    def build_metrics(self, spec: ArchSpec) -> torchmetrics.MetricCollection:
        return torchmetrics.MetricCollection({})

    def step(self, model: nn.Module, loss_fn: nn.Module, batch: Any) -> StepOutput:
        images, targets = batch
        out: torch.Tensor = model(images)
        return StepOutput(loss=loss_fn(out, targets), metric_args=(), batch_size=len(images))

    @torch.no_grad()
    def predict(
        self,
        model: nn.Module,
        loader: torch.utils.data.DataLoader[Any],
        spec: ArchSpec,
        pipeline: FittedPipeline,
    ) -> Predictions:
        k = self.num_outputs(spec)
        size = int((spec.input.shape or [3, 64, 64])[-1])
        preds, gts = [], []
        for images, targets in loader:
            preds += decode_centernet(model(images), k, size)
            gts += targets
        return Predictions(
            y_true=None,
            y_pred=np.array([len(p["boxes"]) for p in preds]),
            extra={"preds": preds, "targets": gts},
        )

    def evaluate(
        self,
        preds: Predictions,
        spec: ArchSpec,
        pipeline: FittedPipeline,
        calibration: dict[str, Any] | None = None,
    ) -> TaskEvaluation:
        from torchmetrics.detection import MeanAveragePrecision

        metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
        metric.update(
            preds.extra["preds"],
            [{"boxes": t["boxes"], "labels": t["labels"]} for t in preds.extra["targets"]],
        )
        res = metric.compute()
        classes = pipeline.classes or []
        per_class = {}
        for cls_id, ap in zip(
            res.get("classes", torch.tensor([])).tolist() if res.get("classes") is not None else [],
            res.get("map_per_class", torch.tensor([])).flatten().tolist(),
            strict=False,
        ):
            name = classes[int(cls_id)] if int(cls_id) < len(classes) else str(cls_id)
            per_class[name] = round(float(ap), 6)
        metrics = {
            k: float(v)
            for k, v in res.items()
            if isinstance(v, torch.Tensor) and v.numel() == 1 and k != "classes"
        }
        return TaskEvaluation(metrics=metrics, detail={"ap_per_class": per_class})


# ------------------------------------------------------------------ segmentación


class DiceCELoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, mask)
        prob = logits.softmax(1)
        onehot = F.one_hot(mask, logits.shape[1]).permute(0, 3, 1, 2).float()
        inter = (prob * onehot).sum((0, 2, 3))
        dice = 1 - ((2 * inter + 1) / (prob.sum((0, 2, 3)) + onehot.sum((0, 2, 3)) + 1)).mean()
        return ce + self.dice_weight * dice


class SegmentationAdapter(TaskAdapter):
    task: ClassVar[TaskType] = TaskType.SEGMENTATION

    def num_outputs(self, spec: ArchSpec) -> int:
        return spec.task.num_classes or 2

    def check_output(self, kind: str, shape: tuple[int, ...], spec: ArchSpec) -> str | None:
        k = self.num_outputs(spec)
        hw = tuple((spec.input.shape or [])[1:])
        if kind != "feature_map" or shape != (k, *hw):
            return f"la salida debe ser [{k}, {hw}] (logits por píxel) y es {kind} {list(shape)}"
        return None

    def build_loss(
        self, spec: ArchSpec, resolve: Resolver, class_weights: torch.Tensor | None
    ) -> nn.Module:
        return DiceCELoss()

    def build_metrics(self, spec: ArchSpec) -> torchmetrics.MetricCollection:
        k = self.num_outputs(spec)
        return torchmetrics.MetricCollection(
            {"iou": torchmetrics.JaccardIndex(task="multiclass", num_classes=k)}
        )

    def step(self, model: nn.Module, loss_fn: nn.Module, batch: Any) -> StepOutput:
        images, masks = batch
        out: torch.Tensor = model(images)
        return StepOutput(
            loss=loss_fn(out, masks),
            metric_args=(out.detach().argmax(1), masks),
            batch_size=len(images),
        )

    @torch.no_grad()
    def predict(
        self,
        model: nn.Module,
        loader: torch.utils.data.DataLoader[Any],
        spec: ArchSpec,
        pipeline: FittedPipeline,
    ) -> Predictions:
        k = self.num_outputs(spec)
        conf = torch.zeros(k, k, dtype=torch.long)
        n = 0
        for images, masks in loader:
            pred = model(images).argmax(1)
            conf += torch.bincount((masks * k + pred).flatten(), minlength=k * k).reshape(k, k)
            n += len(images)
        return Predictions(y_true=None, y_pred=np.zeros(n), extra={"confusion": conf.numpy()})

    def evaluate(
        self,
        preds: Predictions,
        spec: ArchSpec,
        pipeline: FittedPipeline,
        calibration: dict[str, Any] | None = None,
    ) -> TaskEvaluation:
        conf = preds.extra["confusion"].astype(float)
        tp = np.diag(conf)
        iou = tp / np.clip(conf.sum(0) + conf.sum(1) - tp, 1e-12, None)
        dice = 2 * tp / np.clip(conf.sum(0) + conf.sum(1), 1e-12, None)
        classes = pipeline.classes or [str(i) for i in range(len(tp))]
        metrics = {
            "mean_iou": float(iou.mean()),
            "mean_dice": float(dice.mean()),
            "pixel_accuracy": float(tp.sum() / max(conf.sum(), 1)),
            "iou_foreground": float(iou[1:].mean()) if len(iou) > 1 else float(iou[0]),
        }
        detail = {
            "iou_per_class": {c: round(float(v), 6) for c, v in zip(classes, iou, strict=False)},
            "dice_per_class": {c: round(float(v), 6) for c, v in zip(classes, dice, strict=False)},
        }
        return TaskEvaluation(metrics=metrics, detail=detail)


# ------------------------------------------------------------------ OCR


def greedy_ctc(logits: torch.Tensor, alphabet: list[str]) -> list[str]:
    """Decodificación greedy: colapsa repetidos y quita el blank (índice 0)."""
    best = logits.argmax(-1)
    texts = []
    for row in best.tolist():
        out, prev = [], 0
        for i in row:
            if i not in (prev, 0) and i - 1 < len(alphabet):
                out.append(alphabet[i - 1])
            prev = i
        texts.append("".join(out))
    return texts


def edit_distance(a: str | list[str], b: str | list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class OCRAdapter(TaskAdapter):
    """Batch: (imágenes [B,1,32,W], ids [B,Lmax], largos [B]). Vocabulario: blank + alfabeto."""

    task: ClassVar[TaskType] = TaskType.OCR

    def num_outputs(self, spec: ArchSpec) -> int:
        if not spec.task.num_classes:
            raise ValueError("task.num_classes (alfabeto + blank) es requerido para OCR")
        return spec.task.num_classes

    def check_output(self, kind: str, shape: tuple[int, ...], spec: ArchSpec) -> str | None:
        v = self.num_outputs(spec)
        if kind != "sequence" or shape[-1] != v:
            return f"la salida debe ser una secuencia [T, {v}] y es {kind} {list(shape)}"
        return None

    def build_loss(
        self, spec: ArchSpec, resolve: Resolver, class_weights: torch.Tensor | None
    ) -> nn.Module:
        return nn.CTCLoss(blank=0, zero_infinity=True)

    def build_metrics(self, spec: ArchSpec) -> torchmetrics.MetricCollection:
        return torchmetrics.MetricCollection({})

    def step(self, model: nn.Module, loss_fn: nn.Module, batch: Any) -> StepOutput:
        images, ids, lengths = batch
        logits: torch.Tensor = model(images)  # [B, T, V]
        log_probs = logits.log_softmax(-1).permute(1, 0, 2)
        input_lengths = torch.full((len(images),), logits.shape[1], dtype=torch.long)
        loss = loss_fn(log_probs, ids, input_lengths, lengths)
        return StepOutput(loss=loss, metric_args=(), batch_size=len(images))

    @torch.no_grad()
    def predict(
        self,
        model: nn.Module,
        loader: torch.utils.data.DataLoader[Any],
        spec: ArchSpec,
        pipeline: FittedPipeline,
    ) -> Predictions:
        alphabet = pipeline.classes or []
        preds, truth = [], []
        for images, ids, lengths in loader:
            preds += greedy_ctc(model(images), alphabet)
            truth += [
                "".join(alphabet[i - 1] for i in row[:n].tolist())
                for row, n in zip(ids, lengths, strict=True)
            ]
        return Predictions(
            y_true=np.array(truth, dtype=object), y_pred=np.array(preds, dtype=object)
        )

    def evaluate(
        self,
        preds: Predictions,
        spec: ArchSpec,
        pipeline: FittedPipeline,
        calibration: dict[str, Any] | None = None,
    ) -> TaskEvaluation:
        truth = [str(t) for t in (preds.y_true if preds.y_true is not None else [])]
        pred = [str(p) for p in preds.y_pred]
        chars = sum(len(t) for t in truth) or 1
        words = sum(len(t.split()) for t in truth) or 1
        cer = sum(edit_distance(p, t) for p, t in zip(pred, truth, strict=True)) / chars
        wer = (
            sum(edit_distance(p.split(), t.split()) for p, t in zip(pred, truth, strict=True))
            / words
        )
        exact = float(np.mean([p == t for p, t in zip(pred, truth, strict=True)])) if truth else 0.0
        examples = [{"pred": p, "true": t} for p, t in list(zip(pred, truth, strict=True))[:10]]
        return TaskEvaluation(
            metrics={"cer": cer, "wer": wer, "exact_match": exact}, detail={"examples": examples}
        )
