/**
 * Herramienta de etiquetado (SPEC §7.5, RF-LBL-01..03): cola de active learning, sugerencias
 * del modelo (aceptar con Enter), clases con atajos 1–9, multi-etiqueta, cajas y polígonos sobre la
 * imagen y segmentos sobre el audio.
 * Al aplicar, se crea una versión nueva del dataset lista para reentrenar.
 */
import { Link } from "@tanstack/react-router";
import { Check, Download, Sparkles, Wand2 } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import {
  AiBadge,
  Badge,
  Button,
  Card,
  CardTitle,
  EmptyState,
  ErrorNote,
  Field,
  Input,
  Select,
  Spinner,
} from "@/components/ui";
import { PolygonCanvas, SegmentMarker } from "@/features/labeling/ShapeTools";
import { useProjectId } from "@/features/projects/ProjectLayout";
import { downloadFromEngine, engineObjectUrl } from "@/lib/api/download";
import {
  type Box,
  type LabelSample,
  type LabelSet,
  type Polygon,
  type Segment,
  useAcceptSuggestions,
  useApplyLabels,
  useCreateLabelSet,
  useDatasets,
  useLabelQueue,
  useLabelSets,
  useLabelSummary,
  usePrelabelWithModel,
  usePrelabelZeroShot,
  useRuns,
  useSetLabels,
} from "@/lib/api/hooks";
import { formatNumber } from "@/lib/format";

const AUDIO = /\.(wav|flac|ogg|mp3)$/i;

function useSampleImage(labelsetId: string, sample: LabelSample | undefined) {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!sample?.path) return;
    let alive = true;
    let created: string | null = null;
    engineObjectUrl(`/api/v1/labelsets/${labelsetId}/samples/${sample.sample_id}/file`).then(
      (u) => {
        created = u;
        if (alive) setUrl(u);
        else URL.revokeObjectURL(u);
      },
      () => {
        if (alive) setUrl(null);
      },
    );
    return () => {
      alive = false;
      if (created) URL.revokeObjectURL(created);
    };
  }, [labelsetId, sample?.sample_id, sample?.path]);
  return sample?.path ? url : null;
}

/** Cajas: se dibujan arrastrando sobre la imagen (coordenadas normalizadas 0–1). */
function BoxCanvas({
  url,
  boxes,
  current,
  onAdd,
}: {
  url: string;
  boxes: Box[];
  current: string;
  onAdd: (b: Box) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [start, setStart] = useState<{ x: number; y: number } | null>(null);
  const [drag, setDrag] = useState<{ x: number; y: number } | null>(null);
  const pos = (e: React.PointerEvent) => {
    const r = ref.current?.getBoundingClientRect();
    if (!r) return { x: 0, y: 0 };
    return {
      x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)),
      y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)),
    };
  };
  const rect = (a: { x: number; y: number }, b: { x: number; y: number }) => ({
    x1: Math.min(a.x, b.x),
    y1: Math.min(a.y, b.y),
    x2: Math.max(a.x, b.x),
    y2: Math.max(a.y, b.y),
  });
  const live = start && drag ? rect(start, drag) : null;
  return (
    <div
      ref={ref}
      className="relative inline-block cursor-crosshair select-none"
      data-testid="box-canvas"
      onPointerDown={(e) => {
        setStart(pos(e));
        setDrag(pos(e));
      }}
      onPointerMove={(e) => start && setDrag(pos(e))}
      onPointerUp={(e) => {
        if (start) {
          const r = rect(start, pos(e));
          if (r.x2 - r.x1 > 0.01 && r.y2 - r.y1 > 0.01) onAdd({ ...r, label: current });
        }
        setStart(null);
        setDrag(null);
      }}
    >
      <img src={url} alt="" className="max-h-96 rounded-pt" draggable={false} />
      {[...boxes, ...(live ? [{ ...live, label: current }] : [])].map((b, i) => (
        <span
          key={i}
          className="absolute border-2 border-pt-lime"
          style={{
            left: `${b.x1 * 100}%`,
            top: `${b.y1 * 100}%`,
            width: `${(b.x2 - b.x1) * 100}%`,
            height: `${(b.y2 - b.y1) * 100}%`,
          }}
        >
          <span className="absolute -top-5 left-0 bg-pt-dark px-1 text-xs text-pt-lime">
            {b.label}
          </span>
        </span>
      ))}
    </div>
  );
}

function Annotator({ labelset }: { labelset: LabelSet }) {
  const { t, i18n } = useTranslation();
  const [strategy, setStrategy] = useState<"uncertainty" | "diversity" | "random">("uncertainty");
  const queue = useLabelQueue(labelset.id, strategy);
  const setLabels = useSetLabels(labelset.id);
  // La posición se reinicia sola cuando llega una tanda nueva de la cola.
  const [cursor, setCursor] = useState({ batch: 0, index: 0 });
  const index = cursor.batch === queue.dataUpdatedAt ? cursor.index : 0;
  const next = useCallback(
    () => setCursor({ batch: queue.dataUpdatedAt, index: index + 1 }),
    [queue.dataUpdatedAt, index],
  );
  const [multi, setMulti] = useState<string[]>([]);
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [polygons, setPolygons] = useState<Polygon[]>([]);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [boxClass, setBoxClass] = useState(labelset.classes[0] ?? "");
  const shapes = { box: boxes, mask: polygons, temporal_event: segments }[
    labelset.kind as "box" | "mask" | "temporal_event"
  ];
  const undoShape = () => {
    if (labelset.kind === "box") setBoxes((b) => b.slice(0, -1));
    else if (labelset.kind === "mask") setPolygons((b) => b.slice(0, -1));
    else setSegments((b) => b.slice(0, -1));
  };
  const samples = useMemo(() => queue.data ?? [], [queue.data]);
  const sample = samples[index];
  const image = useSampleImage(labelset.id, sample);
  const suggestion = sample?.item?.status === "suggested" ? sample.item : null;

  const save = useCallback(
    (update: {
      label?: string | string[] | null;
      boxes?: Box[];
      polygons?: Polygon[];
      segments?: Segment[];
    }) => {
      if (!sample) return;
      setLabels.mutate(
        [
          {
            sample_id: sample.sample_id,
            label: update.label ?? null,
            boxes: update.boxes ?? [],
            polygons: update.polygons ?? [],
            segments: update.segments ?? [],
          },
        ],
        {
          onSuccess: () => {
            setMulti([]);
            setBoxes([]);
            setPolygons([]);
            setSegments([]);
            next();
          },
        },
      );
    },
    [sample, setLabels, next],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!sample || labelset.kind !== "class") return;
      const target = e.target as HTMLElement;
      if (
        target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.tagName === "SELECT"
      )
        return;
      const n = Number.parseInt(e.key, 10);
      if (n >= 1 && n <= labelset.classes.length) save({ label: labelset.classes[n - 1] ?? null });
      else if (e.key === "Enter" && suggestion?.label) save({ label: suggestion.label });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [sample, suggestion, labelset, save]);

  if (queue.isPending) return <Spinner />;
  if (queue.error) return <ErrorNote error={queue.error} />;

  return (
    <Card>
      <CardTitle className="flex items-center justify-between gap-2">
        {t("labeling.queue")}
        <Select
          value={strategy}
          onChange={(e) => setStrategy(e.target.value as typeof strategy)}
          aria-label={t("labeling.strategy")}
          className="w-48"
        >
          {(["uncertainty", "diversity", "random"] as const).map((s) => (
            <option key={s} value={s}>
              {t(`labeling.strategies.${s}`)}
            </option>
          ))}
        </Select>
      </CardTitle>
      {!sample ? (
        <EmptyState>
          {samples.length ? t("labeling.queueDone") : t("labeling.queueEmpty")}
        </EmptyState>
      ) : (
        <div className="space-y-4">
          <p className="text-xs text-muted">
            {t("labeling.position", { current: index + 1, total: samples.length })} ·{" "}
            <span className="font-mono">{sample.sample_id}</span>
            {sample.split && ` · ${sample.split}`}
          </p>
          {sample.path ? (
            image ? (
              labelset.kind === "box" ? (
                <BoxCanvas
                  url={image}
                  boxes={boxes}
                  current={boxClass}
                  onAdd={(b) => setBoxes((x) => [...x, b])}
                />
              ) : labelset.kind === "mask" ? (
                <PolygonCanvas
                  url={image}
                  polygons={polygons}
                  current={boxClass}
                  onAdd={(p) => setPolygons((x) => [...x, p])}
                />
              ) : labelset.kind === "temporal_event" ? (
                <SegmentMarker
                  url={image}
                  segments={segments}
                  current={boxClass}
                  onAdd={(seg) => setSegments((x) => [...x, seg])}
                />
              ) : AUDIO.test(sample.path) ? (
                <audio
                  src={image}
                  controls
                  className="w-full max-w-xl"
                  aria-label={t("labeling.sampleAlt")}
                />
              ) : (
                <img src={image} alt={t("labeling.sampleAlt")} className="max-h-96 rounded-pt" />
              )
            ) : (
              <Spinner />
            )
          ) : sample.text ? (
            <blockquote className="whitespace-pre-wrap rounded-pt bg-canvas p-3 text-sm">
              {sample.text}
            </blockquote>
          ) : (
            <dl className="grid gap-1 text-sm sm:grid-cols-3">
              {Object.entries(sample.fields ?? {}).map(([k, v]) => (
                <div key={k}>
                  <dt className="text-xs text-muted">{k}</dt>
                  <dd>{String(v ?? "—")}</dd>
                </div>
              ))}
            </dl>
          )}
          {suggestion && (
            <p className="flex items-center gap-2 text-sm">
              <AiBadge />
              {t("labeling.suggestion", {
                label: String(suggestion.label),
                confidence: formatNumber((suggestion.confidence ?? 0) * 100, i18n.language, 0),
              })}
              <Button
                size="sm"
                variant="ai"
                onClick={() => save({ label: suggestion.label ?? null })}
              >
                <Check className="h-4 w-4" aria-hidden="true" />
                {t("labeling.acceptSuggestion")}
              </Button>
            </p>
          )}
          {labelset.kind === "class" && (
            <div className="flex flex-wrap gap-2" role="group" aria-label={t("labeling.classes")}>
              {labelset.classes.map((c, i) => (
                <Button
                  key={c}
                  variant="secondary"
                  loading={setLabels.isPending}
                  onClick={() => save({ label: c })}
                >
                  {i < 9 && <kbd className="mr-1 text-xs text-muted">{i + 1}</kbd>}
                  {c}
                </Button>
              ))}
            </div>
          )}
          {labelset.kind === "multilabel" && (
            <div className="flex flex-wrap items-center gap-3">
              {labelset.classes.map((c) => (
                <label key={c} className="flex items-center gap-1 text-sm">
                  <input
                    type="checkbox"
                    checked={multi.includes(c)}
                    onChange={() =>
                      setMulti((m) => (m.includes(c) ? m.filter((x) => x !== c) : [...m, c]))
                    }
                  />
                  {c}
                </label>
              ))}
              <Button disabled={!multi.length} onClick={() => save({ label: multi })}>
                {t("labeling.save")}
              </Button>
            </div>
          )}
          {shapes && (
            <div className="flex flex-wrap items-end gap-2">
              <Field
                label={t(labelset.kind === "box" ? "labeling.boxClass" : "labeling.shapeClass")}
              >
                <Select
                  value={boxClass}
                  onChange={(e) => setBoxClass(e.target.value)}
                  className="w-40"
                >
                  {labelset.classes.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </Select>
              </Field>
              <span className="text-xs text-muted">
                {t(`labeling.shapes.${labelset.kind}`, { count: shapes.length })}
              </span>
              <Button variant="secondary" disabled={!shapes.length} onClick={undoShape}>
                {t("labeling.undo")}
              </Button>
              <Button disabled={!shapes.length} onClick={() => save({ boxes, polygons, segments })}>
                {t("labeling.save")}
              </Button>
            </div>
          )}
          <div className="flex gap-2">
            <Button variant="ghost" size="sm" onClick={() => next()}>
              {t("labeling.skip")}
            </Button>
          </div>
          <ErrorNote error={setLabels.error} />
        </div>
      )}
    </Card>
  );
}

function LabelSetPanel({ labelset, projectId }: { labelset: LabelSet; projectId: string }) {
  const { t } = useTranslation();
  const summary = useLabelSummary(labelset.id);
  const runs = useRuns(projectId);
  const prelabel = usePrelabelWithModel(labelset.id);
  const zeroShot = usePrelabelZeroShot(labelset.id);
  const classKind = labelset.kind === "class" || labelset.kind === "multilabel";
  const accept = useAcceptSuggestions(labelset.id);
  const apply = useApplyLabels(labelset.id, projectId);
  const [runId, setRunId] = useState("");
  const [threshold, setThreshold] = useState(0.9);
  const [error, setError] = useState<unknown>(null);
  const done = (runs.data ?? []).filter((r) => r.status === "succeeded");
  const s = summary.data;
  const pct = s && s.total ? Math.round((s.accepted / s.total) * 100) : 0;

  return (
    <div className="space-y-4">
      <Card>
        <CardTitle className="flex items-center justify-between">
          {labelset.name ?? labelset.id}
          <Badge>{t(`labeling.kinds.${labelset.kind}`)}</Badge>
        </CardTitle>
        {s ? (
          <>
            <div
              className="mb-2 h-2 rounded bg-canvas"
              role="progressbar"
              aria-valuenow={pct}
              aria-valuemin={0}
              aria-valuemax={100}
            >
              <div className="h-2 rounded bg-primary" style={{ width: `${pct}%` }} />
            </div>
            <p className="text-sm">
              {t("labeling.progress", {
                accepted: s.accepted,
                total: s.total,
                suggested: s.suggested,
              })}
            </p>
            <p className="mt-1 text-xs text-muted">
              {Object.entries(s.by_class)
                .map(([c, n]) => `${c}: ${n}`)
                .join(" · ")}
            </p>
          </>
        ) : (
          <Spinner />
        )}
        <div className="mt-4 flex flex-wrap items-end gap-2">
          <Field label={t("labeling.prelabelRun")}>
            <Select value={runId} onChange={(e) => setRunId(e.target.value)} className="w-56">
              <option value="">—</option>
              {done.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.id.split("-").at(-1)} · {formatNumber(r.metrics?.val_loss ?? null, "es", 3)}
                </option>
              ))}
            </Select>
          </Field>
          <Button
            variant="ai"
            disabled={!runId}
            loading={prelabel.isPending}
            onClick={() => prelabel.mutate(runId)}
          >
            <Wand2 className="h-4 w-4" aria-hidden="true" />
            {t("labeling.prelabel")}
          </Button>
          {classKind && (
            <Button
              variant="ai"
              loading={zeroShot.isPending}
              title={t("labeling.zeroShotHint")}
              onClick={() => zeroShot.mutate(200)}
            >
              <Sparkles className="h-4 w-4" aria-hidden="true" />
              {t("labeling.zeroShot")}
            </Button>
          )}
          <Field label={t("labeling.threshold")}>
            <Input
              type="number"
              min={0.5}
              max={1}
              step={0.05}
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              className="w-24"
            />
          </Field>
          <Button
            variant="secondary"
            disabled={!s?.suggested}
            loading={accept.isPending}
            onClick={() => accept.mutate(threshold)}
          >
            <Sparkles className="h-4 w-4" aria-hidden="true" />
            {t("labeling.acceptAll")}
          </Button>
        </div>
        <div className="mt-4 flex flex-wrap gap-2">
          <Button disabled={!s?.accepted} loading={apply.isPending} onClick={() => apply.mutate()}>
            {t("labeling.apply")}
          </Button>
          {(
            ["csv", "jsonl", ...(labelset.kind === "box" ? ["coco", "yolo", "voc"] : [])] as const
          ).map((f) => (
            <Button
              key={f}
              variant="ghost"
              size="sm"
              onClick={() => {
                setError(null);
                downloadFromEngine(
                  `/api/v1/labelsets/${labelset.id}/export?format=${f}`,
                  `${labelset.id}.${f === "coco" ? "json" : f === "yolo" || f === "voc" ? "zip" : f}`,
                ).catch(setError);
              }}
            >
              <Download className="h-4 w-4" aria-hidden="true" />
              {f.toUpperCase()}
            </Button>
          ))}
        </div>
        {apply.data && (
          <p className="mt-3 text-sm" role="status">
            {t("labeling.applied", { samples: apply.data.num_samples })}{" "}
            <Link
              to="/projects/$projectId/train"
              params={{ projectId }}
              className="font-semibold underline"
            >
              {t("labeling.goTrain")}
            </Link>
          </p>
        )}
        <ErrorNote
          error={prelabel.error ?? zeroShot.error ?? accept.error ?? apply.error ?? error}
        />
      </Card>
      <Annotator labelset={labelset} />
    </div>
  );
}

export function LabelingPage() {
  const { t } = useTranslation();
  const projectId = useProjectId();
  const datasets = useDatasets(projectId);
  const [dvId, setDvId] = useState("");
  const chosen = dvId || datasets.data?.at(-1)?.id || "";
  const sets = useLabelSets(chosen || undefined);
  const create = useCreateLabelSet(chosen);
  const [selected, setSelected] = useState<string>("");
  const [kind, setKind] = useState<LabelSet["kind"]>("class");
  const [classes, setClasses] = useState("");
  const current = (sets.data ?? []).find((s) => s.id === selected) ?? sets.data?.at(-1);

  if (datasets.isPending) return <Spinner />;
  if (!datasets.data?.length) return <EmptyState>{t("train.noData")}</EmptyState>;
  return (
    <div className="space-y-4">
      <Card>
        <CardTitle>{t("labeling.title")}</CardTitle>
        <div className="flex flex-wrap items-end gap-3">
          <Field label={t("train.dataset")}>
            <Select value={chosen} onChange={(e) => setDvId(e.target.value)} className="w-72">
              {datasets.data.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.content_hash.slice(0, 10)} · {d.num_samples} · {d.target ?? "—"}
                </option>
              ))}
            </Select>
          </Field>
          {(sets.data ?? []).length > 0 && (
            <Field label={t("labeling.set")}>
              <Select
                value={current?.id ?? ""}
                onChange={(e) => setSelected(e.target.value)}
                className="w-56"
              >
                {(sets.data ?? []).map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.name ?? s.id}
                  </option>
                ))}
              </Select>
            </Field>
          )}
        </div>
        <form
          className="mt-4 flex flex-wrap items-end gap-3"
          onSubmit={(e) => {
            e.preventDefault();
            create.mutate(
              {
                kind,
                classes: classes
                  .split(",")
                  .map((c) => c.trim())
                  .filter(Boolean),
                name: null,
                target: null,
              },
              { onSuccess: (ls) => setSelected(ls.id) },
            );
          }}
        >
          <Field label={t("labeling.kind")}>
            <Select
              value={kind}
              onChange={(e) => setKind(e.target.value as typeof kind)}
              className="w-44"
            >
              {(["class", "multilabel", "box", "mask", "temporal_event"] as const).map((k) => (
                <option key={k} value={k}>
                  {t(`labeling.kinds.${k}`)}
                </option>
              ))}
            </Select>
          </Field>
          <Field label={t("labeling.newClasses")} hint={t("labeling.newClassesHint")}>
            <Input value={classes} onChange={(e) => setClasses(e.target.value)} className="w-72" />
          </Field>
          <Button type="submit" variant="secondary" loading={create.isPending}>
            {t("labeling.create")}
          </Button>
        </form>
        <ErrorNote error={create.error ?? sets.error} />
      </Card>
      {current && <LabelSetPanel key={current.id} labelset={current} projectId={projectId} />}
    </div>
  );
}
