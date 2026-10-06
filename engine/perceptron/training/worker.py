"""Proceso worker de un run: `python -m perceptron.training.worker <run.json>` (ADR-0015).

stdout: solo eventos JSONL. stderr: logs. stdin: órdenes del Engine.
Códigos de salida: 0 ok/cancelado/pausado · 2 error · 3 memoria insuficiente (OOM).
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

EXIT_ERROR = 2
EXIT_OOM = 3


def _environment(device: str, seed: int, deterministic: bool) -> dict[str, Any]:
    import lightning
    import torch

    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "lightning": lightning.__version__,
        "device": device,
        "seed": seed,
        "deterministic": deterministic,
    }
    if torch.cuda.is_available():
        env["cuda"] = torch.version.cuda
        env["gpu"] = torch.cuda.get_device_name(0)
    return env


def _precision(requested: str, device: str) -> str:
    import torch

    if requested != "auto":
        return requested if requested != "32" else "32-true"
    if device in ("cuda", "rocm") and torch.cuda.is_available():
        return "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
    return "32-true"


def _accelerator(device: str) -> str:
    return {"cuda": "gpu", "rocm": "gpu", "xpu": "xpu", "cpu": "cpu"}.get(device, "cpu")


def run(config_path: Path) -> int:
    from perceptron.training.callbacks import EventEmitter
    from perceptron.training.config import RunConfig

    cfg = RunConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    emitter = EventEmitter(cfg.run_id)
    start = time.time()
    try:
        return _train(cfg, emitter, start)
    except Exception as e:  # informar cualquier fallo al supervisor
        import torch

        oom = isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()
        emitter.emit(
            "error",
            data={
                "code": "oom" if oom else "exception",
                "type": type(e).__name__,
                "message": str(e)[:2000],
                "traceback": traceback.format_exc()[-4000:],
                "suggestion": "Reducí el batch size o la resolución de entrada." if oom else None,
            },
        )
        logging.getLogger(__name__).exception("run falló")
        return EXIT_OOM if oom else EXIT_ERROR


def _train(cfg: Any, emitter: Any, start: float) -> int:
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    from perceptron.archspec.schema import ArchSpec, resolve
    from perceptron.data.pipeline.pipeline import FittedPipeline
    from perceptron.data.view import DatasetView
    from perceptron.training.callbacks import (
        ControlCallback,
        FreezeBackboneCallback,
        ProgressCallback,
    )
    from perceptron.training.config import (
        ARCHSPEC_FILE,
        CHECKPOINTS_DIR,
        LAST_CKPT,
        PIPELINE_FILE,
        RESULT_FILE,
        RunResult,
    )
    from perceptron.training.data import (
        auto_batch_size,
        auto_num_workers,
        class_weights,
        make_dataset,
        make_loader,
    )
    from perceptron.training.module import PerceptronModule, monitor_mode
    from perceptron.training.tuning import find_lr, tune_batch_size

    L.seed_everything(cfg.seed, workers=True, verbose=False)
    spec = ArchSpec.model_validate(cfg.archspec)
    fitted = FittedPipeline.model_validate(cfg.pipeline)
    view = DatasetView(cfg.dataset_dir)
    run_dir: Path = cfg.run_dir
    ckpt_dir = run_dir / CHECKPOINTS_DIR
    (run_dir / ARCHSPEC_FILE).write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / PIPELINE_FILE).write_text(fitted.model_dump_json(indent=2), encoding="utf-8")

    train_ds = make_dataset(view, fitted, "train", train=True)
    val_ds = make_dataset(view, fitted, "val", train=False)
    n_train = len(train_ds)  # type: ignore[arg-type]
    if n_train < 2:
        raise ValueError("el split de train tiene menos de 2 ejemplos")

    bs_cfg = cfg.batch_size or resolve(spec.training.batch_size, cfg.overrides)
    batch_size = (
        auto_batch_size(view.modality, n_train) if bs_cfg in (None, "auto") else int(bs_cfg)
    )
    workers = (
        auto_num_workers(view.modality, n_train)
        if cfg.num_workers == "auto"
        else int(cfg.num_workers)
    )
    if cfg.code is not None:
        workers = 0  # el sandbox no admite procesos hijos
    train_dl = make_loader(
        train_ds,
        batch_size,
        shuffle=True,
        num_workers=workers,
        seed=cfg.seed,
        oversample=spec.training.oversample,
    )
    val_dl = make_loader(val_ds, batch_size, shuffle=False, num_workers=workers, seed=cfg.seed)

    if cfg.code is not None:
        _enter_sandbox(cfg, run_dir)

    module = PerceptronModule(
        spec,
        cfg.overrides,
        class_weights=class_weights(fitted, train_ds),
        pretrained_allowed=cfg.pretrained_allowed,
        target_scale=(
            (fitted.target_mean, fitted.target_std)
            if fitted.target_mean is not None and fitted.target_std
            else None
        ),
    )

    epochs = cfg.max_epochs or int(resolve(spec.training.epochs, cfg.overrides))
    min_epochs = spec.training.min_epochs or max(1, epochs // 3)
    es = spec.training.early_stopping
    monitor = es.monitor if es else "val_loss"
    mode = (es.mode if es and es.mode else None) or monitor_mode(monitor)
    progress = ProgressCallback(emitter, cfg.emit_every_n_batches)
    control = ControlCallback()
    best_ckpt = ModelCheckpoint(
        dirpath=ckpt_dir, filename="best", monitor=monitor, mode=mode, save_top_k=1, save_last=True
    )
    best_ckpt.CHECKPOINT_NAME_LAST = "last"
    callbacks: list[L.Callback] = [progress, control, best_ckpt]
    if es:
        # Paciencia proporcional al presupuesto: con muchas épocas, 5 es poco para salir de
        # una meseta; con pocas, no se alarga más de lo que dura el entrenamiento.
        patience = max(es.patience, epochs // 6)
        callbacks.append(EarlyStopping(monitor=monitor, mode=mode, patience=patience))
    freeze = spec.training.freeze_backbone_epochs
    for node in spec.nodes:
        f = str(resolve(node.params.get("freeze", "none"), cfg.overrides))
        if f.startswith("until_epoch:"):
            freeze = max(freeze, int(f.split(":", 1)[1]))
    progressive = spec.training.unfreeze == "progressive"
    if (freeze or progressive) and module.backbones:
        callbacks.append(FreezeBackboneCallback(freeze, progressive=progressive))

    precision = _precision(spec.training.precision, cfg.device.value)
    tuner_kwargs = {"accelerator": _accelerator(cfg.device.value), "precision": precision}

    def tuning_loader(size: int, train: bool) -> Any:
        ds = train_ds if train else val_ds
        return make_loader(ds, size, shuffle=train, num_workers=0, seed=cfg.seed)

    tuned: dict[str, Any] = {}
    ddp = cfg.devices > 1 and cfg.device.value != "cpu"
    if (
        bs_cfg in (None, "auto")
        and cfg.device.value != "cpu"
        and cfg.resume_from is None
        and not ddp
    ):
        # En GPU el mayor batch que entra (búsqueda binaria contra OOM), con techo por datos.
        found = tune_batch_size(
            module,
            tuning_loader,
            start=batch_size,
            n_train=n_train,
            trainer_kwargs=tuner_kwargs,
            root=run_dir / "tuning",
        )
        if found != batch_size:
            batch_size = found
            tuned["batch_size"] = found
            train_dl = make_loader(
                train_ds,
                batch_size,
                shuffle=True,
                num_workers=workers,
                seed=cfg.seed,
                oversample=spec.training.oversample,
            )
            val_dl = make_loader(
                val_ds, batch_size, shuffle=False, num_workers=workers, seed=cfg.seed
            )
    if spec.training.lr_finder and cfg.resume_from is None and not ddp:
        lr = find_lr(
            module,
            tuning_loader,
            batch_size=batch_size,
            trainer_kwargs=tuner_kwargs,
            root=run_dir / "tuning",
        )
        if lr is not None:
            module.lr_override = lr
            tuned["lr"] = lr

    trainer = L.Trainer(
        accelerator=_accelerator(cfg.device.value),
        # Varias GPUs para un mismo run: DDP en un nodo (RF-TRN-08); el batch es por GPU.
        devices=cfg.devices if ddp else 1,
        strategy="ddp" if ddp else "auto",
        max_epochs=epochs,
        min_epochs=min(min_epochs, epochs),
        max_time={"seconds": cfg.max_time_s} if cfg.max_time_s else None,
        precision=precision,  # type: ignore[arg-type]
        gradient_clip_val=spec.training.gradient_clip,
        deterministic="warn" if cfg.deterministic else False,
        callbacks=callbacks,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=run_dir,
        limit_train_batches=cfg.limit_train_batches,
        num_sanity_val_steps=0,
    )
    env = _environment(cfg.device.value, cfg.seed, cfg.deterministic)
    emitter.emit(
        "started",
        data={
            "batch_size": batch_size,
            "num_workers": workers,
            "precision": precision,
            "max_epochs": epochs,
            "n_train": n_train,
            "n_val": len(val_ds),  # type: ignore[arg-type]
            "num_params": sum(p.numel() for p in module.parameters()),
            "environment": env,
            "tuned": tuned,
        },
    )
    trainer.fit(
        module, train_dl, val_dl, ckpt_path=str(cfg.resume_from) if cfg.resume_from else None
    )

    status = "succeeded"
    if control.command == "pause":
        status = "paused"
    elif control.command == "stop":
        status = "cancelled"
    history = progress.history
    best_score = best_ckpt.best_model_score
    best_metrics: dict[str, float] = {}
    if history:
        best_row = (
            min(history, key=lambda r: r.get(monitor, float("inf")))
            if mode == "min"
            else max(history, key=lambda r: r.get(monitor, float("-inf")))
        )
        best_metrics = {k: v for k, v in best_row.items() if k.startswith("val_") or k == "epoch"}
    if best_score is not None:
        best_metrics[monitor] = float(best_score)
    last = ckpt_dir / f"{LAST_CKPT}"
    best = Path(best_ckpt.best_model_path) if best_ckpt.best_model_path else None
    if status == "succeeded" and best is not None and not ddp and cfg.code is None:
        exact = _exact_linear_fit(spec, module, trainer, train_ds, val_dl, best, batch_size, cfg)
        exact_ckpt = best.with_name(EXACT_CKPT)
        if exact is not None and _better(exact.get(monitor), best_metrics.get(monitor), mode):
            exact_ckpt.replace(best)  # el ajuste exacto pasa a ser el mejor checkpoint
            best_metrics.update(exact)
            env["exact_fit"] = "least_squares"
            emitter.emit(
                "log",
                data={
                    "message": "Regresión lineal: pesos finales por mínimos cuadrados (ajuste "
                    "exacto, sin la oscilación del optimizador)."
                },
            )
        exact_ckpt.unlink(missing_ok=True)
    result = RunResult(
        run_id=cfg.run_id,
        status=status,  # type: ignore[arg-type]
        epochs=trainer.current_epoch,
        best_metrics=best_metrics,
        last_metrics=history[-1] if history else {},
        monitor=monitor,
        best_checkpoint=best,
        last_checkpoint=last if last.exists() else None,
        duration_s=round(time.time() - start, 3),
        environment=env,
        history=history,
    )
    if emitter.enabled:  # en DDP, solo el proceso principal deja el resultado
        (run_dir / RESULT_FILE).write_text(result.model_dump_json(indent=2), encoding="utf-8")
    emitter.emit(
        "paused" if status == "paused" else "finished", data=json.loads(result.model_dump_json())
    )
    return 0


EXACT_CKPT = "exact.ckpt"


def _better(new: float | None, old: float | None, mode: str) -> bool:
    if new is None:
        return False
    if old is None:
        return True
    return new <= old if mode == "min" else new >= old


def _exact_linear_fit(
    spec: Any,
    module: Any,
    trainer: Any,
    train_ds: Any,
    val_dl: Any,
    best: Path,
    batch_size: int,
    cfg: Any,
) -> dict[str, float] | None:
    """Regresión lineal pura: parte del mejor checkpoint, calcula la capa lineal exacta por
    mínimos cuadrados y, si mejora la validación, la guarda como el mejor checkpoint.
    Devuelve las métricas de validación del ajuste exacto (o None si no aplica)."""
    import torch

    from perceptron.archspec.defaults import is_linear_regression
    from perceptron.training.data import make_loader
    from perceptron.training.exact import least_squares_head

    if not is_linear_regression(spec):
        return None
    state = torch.load(best, map_location="cpu", weights_only=True)["state_dict"]
    module.load_state_dict(state)
    loader = make_loader(train_ds, batch_size, shuffle=False, num_workers=0, seed=cfg.seed)
    if not least_squares_head(module.model, loader):
        return None
    results = trainer.validate(module, val_dl, verbose=False)
    metrics = {k: float(v) for k, v in (results[0] if results else {}).items()}
    if not metrics:
        return None
    trainer.save_checkpoint(str(best.with_name(EXACT_CKPT)))
    return metrics


def _enter_sandbox(cfg: Any, run_dir: Path) -> None:
    """Guardas del sandbox y carga del código experto (ADR-0025); irreversible."""
    from perceptron.domain.enums import Device
    from perceptron.sandbox import code
    from perceptron.sandbox.guard import Policy, install, runtime_roots

    (run_dir / code.CODE_FILE).write_text(cfg.code, encoding="utf-8")
    limits = cfg.sandbox
    # RLIMIT_AS limita memoria virtual: con CUDA se reserva mucho espacio de direcciones.
    memory = limits.memory_mb if (cfg.device is Device.CPU or sys.platform == "win32") else None
    install(
        Policy(
            write_roots=[run_dir],
            read_roots=[cfg.dataset_dir, *runtime_roots()],
            memory_mb=memory,
            cpu_seconds=limits.cpu_seconds,
        )
    )
    code.load(cfg.code)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        sys.stderr.write("uso: python -m perceptron.training.worker <run.json>\n")
        return EXIT_ERROR
    logging.basicConfig(level=os.environ.get("PERCEPTRON_WORKER_LOG", "WARNING"), stream=sys.stderr)
    return run(Path(args[0]))


if __name__ == "__main__":
    sys.exit(main())
