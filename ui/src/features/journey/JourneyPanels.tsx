/**
 * Paneles del asistente guiado: cada uno muestra solo lo necesario para la acción actual.
 * La lógica (qué falta, qué datos pedir, cómo leer el resultado) vive en el motor; acá se
 * presenta y se envían las respuestas.
 */
import { useMutation } from "@tanstack/react-query";
import { Check, Circle, Download, Loader2 } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import {
  Badge,
  Button,
  Card,
  CardTitle,
  ErrorNote,
  Field,
  Input,
  Select,
  Textarea,
} from "@/components/ui";
import { getApiClient } from "@/lib/api/client";
import {
  type Schemas,
  unwrap,
  usePredictFile,
  usePredictRows,
  usePredictTexts,
} from "@/lib/api/hooks";
import { postForm } from "@/lib/api/upload";
import { formatNumber } from "@/lib/format";

import { factValue, fieldLabel } from "./journeyText";

export type Journey = Schemas["Journey"];
type Facts = Schemas["Facts"];
type InputForm = Schemas["InputForm"];
type Guide = Schemas["DataGuide"];

const SELECTS: Record<string, string[]> = {
  data_kind: ["table", "images", "text", "audio", "other"],
  problem: ["category", "value", "rule", "forecast", "anomaly"],
  error_costs: ["symmetric", "false_negative_worse", "false_positive_worse"],
};
const BOOLEANS = new Set(["labels_available", "extrapolate", "grouped"]);
const LISTS = new Set(["classes", "input_names"]);

/** Control de respaldo para el próximo dato que falta (también para corregir uno). */
export function FactControl({
  name,
  facts,
  busy,
  onSave,
}: {
  name: string;
  facts: Facts;
  busy: boolean;
  onSave: (facts: Facts, text: string) => void;
}) {
  const { t } = useTranslation();
  const [value, setValue] = useState("");
  const options =
    name === "positive_class" && facts.classes?.length
      ? facts.classes
      : BOOLEANS.has(name)
        ? ["yes", "no"]
        : SELECTS[name];
  const label = fieldLabel(t, name, facts);
  const submit = () => {
    const parsed: unknown = LISTS.has(name)
      ? value
          .split(",")
          .map((v) => v.trim())
          .filter(Boolean)
      : name === "success_target"
        ? Number(value)
        : BOOLEANS.has(name)
          ? value === "yes"
          : value.trim();
    const shown =
      options && name !== "positive_class"
        ? factValue(t, BOOLEANS.has(name) ? value === "yes" : value)
        : value;
    onSave({ [name]: parsed } as Facts, `${label} ${shown}`);
    setValue("");
  };
  return (
    <form
      className="space-y-2"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <Field label={label}>
        {options ? (
          <Select required value={value} onChange={(e) => setValue(e.target.value)}>
            <option value="">{t("journey.choose")}</option>
            {options.map((o) => (
              <option key={o} value={o}>
                {name === "positive_class" ? o : t(`journey.options.${o}`)}
              </option>
            ))}
          </Select>
        ) : (
          <Input
            required
            value={value}
            onChange={(e) => setValue(e.target.value)}
            type={name === "success_target" ? "number" : "text"}
            min={1}
            max={100}
            placeholder={
              ["prediction", "classes", "target_name", "input_names"].includes(name)
                ? t(`journey.placeholders.${name}`)
                : undefined
            }
          />
        )}
      </Field>
      <Button type="submit" size="sm" disabled={!value || busy}>
        {t("journey.answer")}
      </Button>
    </form>
  );
}

// ---------------------------------------------------------------------- datos

export function GuideView({ guide }: { guide: Guide }) {
  const { t } = useTranslation();
  const download = () => {
    const blob = new Blob([guide.template_csv ?? ""], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "plantilla.csv";
    a.click();
    URL.revokeObjectURL(url);
  };
  const columns = guide.example_columns ?? [];
  return (
    <div className="space-y-4 text-sm">
      <ol className="list-decimal space-y-1 pl-5">
        {(guide.steps ?? []).map((s) => (
          <li key={s}>{s}</li>
        ))}
      </ol>
      {columns.length > 0 && (
        <div>
          <p className="mb-1 font-semibold">{t("journey.data.example")}</p>
          <div className="overflow-auto">
            <table className="w-full border-collapse text-xs">
              <thead>
                <tr>
                  {columns.map((c, i) => (
                    <th
                      key={c}
                      className={`border border-line px-2 py-1 text-left ${i === columns.length - 1 ? "bg-pt-lime/20" : ""}`}
                    >
                      {c}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(guide.example_rows ?? []).map((row, r) => (
                  <tr key={r}>
                    {row.map((cell, i) => (
                      <td key={i} className="border border-line px-2 py-1">
                        {cell}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {guide.template_csv && (
            <Button size="sm" variant="secondary" className="mt-2" onClick={download}>
              <Download className="h-4 w-4" aria-hidden="true" />
              {t("journey.data.template")}
            </Button>
          )}
        </div>
      )}
      {(guide.example_tree ?? []).length > 0 && (
        <div>
          <p className="mb-1 font-semibold">{t("journey.data.tree")}</p>
          <pre className="overflow-auto rounded bg-canvas p-2 text-xs">
            {(guide.example_tree ?? []).join("\n")}
          </pre>
        </div>
      )}
      {(guide.tips ?? []).length > 0 && (
        <details open>
          <summary className="cursor-pointer font-semibold">{t("journey.data.tips")}</summary>
          <ul className="mt-1 list-disc space-y-1 pl-5">
            {(guide.tips ?? []).map((tip) => (
              <li key={tip}>{tip}</li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

export function Notes({
  title,
  items,
  tone,
}: {
  title: string;
  items: string[];
  tone: "bad" | "warn";
}) {
  if (!items.length) return null;
  return (
    <div
      role={tone === "bad" ? "alert" : "status"}
      className={`rounded-pt border p-3 text-sm ${tone === "bad" ? "border-bad" : "border-warn"}`}
    >
      <p className="font-semibold">{title}</p>
      <ul className="mt-1 list-disc space-y-1 pl-5">
        {items.map((i) => (
          <li key={i}>{i}</li>
        ))}
      </ul>
    </div>
  );
}

export function DataSummary({ summary }: { summary: Record<string, unknown> }) {
  const { t, i18n } = useTranslation();
  const classes = (summary.classes as { value: string; count: number }[] | undefined) ?? [];
  const range = summary.range as { low?: number; typical?: number; high?: number } | undefined;
  const inputs = (summary.inputs as string[] | undefined) ?? [];
  const n = (v?: number) => formatNumber(v ?? null, i18n.language, 4);
  return (
    <div className="space-y-2 text-sm">
      <p>{t("journey.review.rows", { count: Number(summary.rows ?? 0) })}</p>
      {typeof summary.target === "string" && (
        <p>{t("journey.review.target", { target: summary.target })}</p>
      )}
      {inputs.length > 0 && <p>{t("journey.review.inputs", { inputs: inputs.join(", ") })}</p>}
      {classes.length > 0 && (
        <div>
          <p className="font-semibold">{t("journey.review.classes")}</p>
          <ul className="flex flex-wrap gap-2">
            {classes.map((c) => (
              <li key={c.value}>
                <Badge>
                  {c.value}: {c.count}
                </Badge>
              </li>
            ))}
          </ul>
        </div>
      )}
      {range && (
        <p>
          {t("journey.review.range", {
            low: n(range.low),
            high: n(range.high),
            typical: n(range.typical),
          })}
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------- entrenamiento

const TRAIN_STAGES = [
  "design_pipeline",
  "design_architectures",
  "design_tournament",
  "design_strategy",
  "learning",
  "evaluating",
  "preparing_prediction",
];

export function TrainingStages({ journey }: { journey: Journey }) {
  const { t } = useTranslation();
  const stage =
    journey.stage === "queued"
      ? "learning"
      : journey.stage === "designing"
        ? "design_pipeline"
        : journey.stage;
  const stages =
    journey.stage === "searching_formula" ? [...TRAIN_STAGES, "searching_formula"] : TRAIN_STAGES;
  const current = stages.indexOf(stage ?? "");
  const progress = journey.progress ?? {};
  return (
    <ol className="space-y-2 text-sm" aria-label={t("journey.training.title")}>
      {stages.map((s, i) => {
        const done = current > i;
        const active = current === i;
        return (
          <li
            key={s}
            className={`flex items-center gap-2 ${active ? "font-semibold" : done ? "" : "text-muted"}`}
          >
            {done ? (
              <Check className="h-4 w-4 text-ok" aria-hidden="true" />
            ) : active ? (
              <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
            ) : (
              <Circle className="h-4 w-4" aria-hidden="true" />
            )}
            <span>{t(`journey.stages.${s}`)}</span>
            {active && s === "learning" && progress.total ? (
              <span className="text-muted">
                ·{" "}
                {t("journey.training.trials", {
                  done: Math.min((progress.done ?? 0) + 1, progress.total),
                  total: progress.total,
                })}
              </span>
            ) : null}
          </li>
        );
      })}
    </ol>
  );
}

// ---------------------------------------------------------------------- resultados

export function Results({ journey }: { journey: Journey }) {
  const { t } = useTranslation();
  const r = (journey.result ?? {}) as {
    meets_target?: boolean;
    explanation?: string[];
    next_steps?: string[];
    per_class?: { label: string; found: number; support: number }[];
    formula?: { formula: string; excel: string; python: string; better: boolean };
  };
  const [copied, setCopied] = useState<string | null>(null);
  const copy = (text: string, key: string) => {
    void navigator.clipboard?.writeText(text);
    setCopied(key);
  };
  return (
    <div className="space-y-3 text-sm">
      <p className={`font-semibold ${r.meets_target ? "text-ok" : "text-warn"}`}>
        {t(r.meets_target ? "journey.results.met" : "journey.results.notMet")}
      </p>
      <ul className="list-disc space-y-1 pl-5">
        {(r.explanation ?? []).map((e) => (
          <li key={e}>{e}</li>
        ))}
      </ul>
      {(r.per_class ?? []).length > 0 && (
        <ul className="space-y-1 text-xs text-muted">
          {(r.per_class ?? []).map((p) => (
            <li key={p.label}>
              {t("journey.results.perClass", {
                found: p.found,
                label: p.label,
                support: p.support,
              })}
            </li>
          ))}
        </ul>
      )}
      {r.formula && (
        <div className="rounded-pt border border-brand p-3">
          <p className="font-semibold">{t("journey.results.formula")}</p>
          <p className="font-mono">{r.formula.formula}</p>
          <div className="mt-2 flex gap-2">
            <Button
              size="sm"
              variant="secondary"
              onClick={() => copy(r.formula?.excel ?? "", "excel")}
            >
              {copied === "excel" ? t("journey.results.copied") : t("journey.results.copyExcel")}
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onClick={() => copy(r.formula?.python ?? "", "python")}
            >
              {copied === "python" ? t("journey.results.copied") : t("journey.results.copyPython")}
            </Button>
          </div>
        </div>
      )}
      {(r.next_steps ?? []).length > 0 && (
        <ul className="list-disc space-y-1 pl-5">
          {(r.next_steps ?? []).map((s) => (
            <li key={s}>{s}</li>
          ))}
        </ul>
      )}
      {journey.design_reason && (
        <details>
          <summary className="cursor-pointer text-muted">{t("journey.results.why")}</summary>
          <p className="mt-1">{journey.design_reason}</p>
        </details>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------- entradas del modelo

function CaseInputs({
  form,
  values,
  setValues,
  file,
  setFile,
}: {
  form: InputForm;
  values: Record<string, string>;
  setValues: (v: Record<string, string>) => void;
  file: File | null;
  setFile: (f: File | null) => void;
}) {
  const { t } = useTranslation();
  if (form.kind === "image" || form.kind === "audio")
    return (
      <Field label={t("journey.use.file")}>
        <input
          type="file"
          accept={form.kind === "image" ? "image/*" : "audio/*"}
          required
          aria-label={t("journey.use.file")}
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        {file && <span className="sr-only">{file.name}</span>}
      </Field>
    );
  if (form.kind === "text")
    return (
      <Field label={t("journey.use.text")}>
        <Textarea
          required
          value={values.__text ?? ""}
          onChange={(e) => setValues({ ...values, __text: e.target.value })}
        />
      </Field>
    );
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {(form.fields ?? []).map((f) => (
        <Field key={f.name} label={f.name}>
          {f.kind === "category" && (f.options ?? []).length > 0 ? (
            <Select
              value={values[f.name] ?? ""}
              onChange={(e) => setValues({ ...values, [f.name]: e.target.value })}
            >
              <option value="">{t("journey.choose")}</option>
              {(f.options ?? []).map((o) => (
                <option key={o}>{o}</option>
              ))}
            </Select>
          ) : f.kind === "boolean" ? (
            <Select
              value={values[f.name] ?? ""}
              onChange={(e) => setValues({ ...values, [f.name]: e.target.value })}
            >
              <option value="">{t("journey.choose")}</option>
              <option value="true">{t("journey.options.yes")}</option>
              <option value="false">{t("journey.options.no")}</option>
            </Select>
          ) : (
            <Input
              type={f.kind === "number" ? "number" : f.kind === "date" ? "date" : "text"}
              step="any"
              value={values[f.name] ?? ""}
              placeholder={f.example ?? undefined}
              onChange={(e) => setValues({ ...values, [f.name]: e.target.value })}
            />
          )}
        </Field>
      ))}
    </div>
  );
}

function rowOf(form: InputForm, values: Record<string, string>): Record<string, unknown> {
  return Object.fromEntries(
    (form.fields ?? []).map((f) => {
      const v = values[f.name] ?? "";
      if (v === "") return [f.name, null];
      if (f.kind === "number") return [f.name, Number(v)];
      if (f.kind === "boolean") return [f.name, v === "true"];
      return [f.name, v];
    }),
  );
}

export function ValidatePanel({
  journey,
  projectId,
  onUpdate,
}: {
  journey: Journey;
  projectId: string;
  onUpdate: (j: Journey) => void;
}) {
  const { t } = useTranslation();
  const form = journey.form as InputForm;
  const [values, setValues] = useState<Record<string, string>>({});
  const [file, setFile] = useState<File | null>(null);
  const [actual, setActual] = useState("");
  const classes = ((journey.contract?.outputs as { classes?: string[] } | undefined)?.classes ??
    []) as string[];
  const single = useMutation({
    mutationFn: async () => {
      if (form.kind === "image" || form.kind === "audio") {
        const body = new FormData();
        if (file) body.append("file", file);
        body.append("actual", actual);
        return postForm<Schemas["GuidedValidation"]>(
          `/api/v1/projects/${projectId}/journey/validate/file`,
          body,
        );
      }
      return unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/projects/{project_id}/journey/validate", {
          params: { path: { project_id: projectId } },
          body:
            form.kind === "text"
              ? { text: values.__text ?? "", actual }
              : { row: rowOf(form, values), actual },
        }),
      );
    },
    onSuccess: (r) => onUpdate(r.journey),
  });
  const batch = useMutation({
    mutationFn: async (f: File) => {
      const body = new FormData();
      body.append("file", f);
      return postForm<Schemas["GuidedValidation"]>(
        `/api/v1/projects/${projectId}/journey/validate/file`,
        body,
      );
    },
    onSuccess: (r) => onUpdate(r.journey),
  });
  const validations = journey.validations ?? [];
  const last = single.data;
  return (
    <Card>
      <CardTitle>{t("journey.validate.title")}</CardTitle>
      <p className="mb-3 text-sm text-muted">{t("journey.validate.help")}</p>
      <form
        className="space-y-3"
        onSubmit={(e) => {
          e.preventDefault();
          single.mutate();
        }}
      >
        <CaseInputs
          form={form}
          values={values}
          setValues={setValues}
          file={file}
          setFile={setFile}
        />
        <Field label={t("journey.validate.actual")}>
          {classes.length ? (
            <Select required value={actual} onChange={(e) => setActual(e.target.value)}>
              <option value="">{t("journey.choose")}</option>
              {classes.map((c) => (
                <option key={c}>{c}</option>
              ))}
            </Select>
          ) : (
            <Input
              required
              type="number"
              step="any"
              value={actual}
              onChange={(e) => setActual(e.target.value)}
            />
          )}
        </Field>
        <Button type="submit" loading={single.isPending} disabled={!actual}>
          {t("journey.validate.check")}
        </Button>
      </form>
      {last && (
        <p role="status" className="mt-3 text-sm">
          {t("journey.validate.predicted", { value: String(last.prediction) })} —{" "}
          <strong className={last.correct ? "text-ok" : "text-bad"}>
            {t(last.correct ? "journey.validate.correct" : "journey.validate.incorrect")}
          </strong>
        </p>
      )}
      {(form.kind === "fields" || form.kind === "text") && journey.target && (
        <Field label={t("journey.validate.batch", { target: journey.target })}>
          <input
            type="file"
            accept=".csv"
            aria-label={t("journey.validate.batch", { target: journey.target })}
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) batch.mutate(f);
              e.target.value = "";
            }}
          />
        </Field>
      )}
      {batch.data && (
        <p className="mt-2 text-sm">
          {t("journey.validate.batchDone", { count: batch.data.checked })}
        </p>
      )}
      {validations.length > 0 && (
        <p className="mt-3 text-sm font-semibold">
          {t("journey.validate.counts", {
            total: validations.length,
            correct: validations.filter((v) => (v as { correct?: boolean }).correct).length,
          })}
        </p>
      )}
      {journey.state === "ready_to_use" && (
        <p className="mt-2 text-sm text-ok">{t("journey.validate.ready")}</p>
      )}
      <p className="mt-3 text-xs text-muted">{t("journey.noGuarantee")}</p>
      <ErrorNote error={single.error ?? batch.error} />
    </Card>
  );
}

export function UsePanel({ journey }: { journey: Journey }) {
  const { t, i18n } = useTranslation();
  const form = journey.form as InputForm;
  const runId = journey.run_id ?? "";
  const [values, setValues] = useState<Record<string, string>>({});
  const [file, setFile] = useState<File | null>(null);
  const rows = usePredictRows(runId);
  const texts = usePredictTexts(runId);
  const files = usePredictFile(runId);
  const result = rows.data ?? texts.data ?? files.data;
  const p = result?.predictions?.[0] as { prediction?: unknown; confidence?: number } | undefined;
  const pending = rows.isPending || texts.isPending || files.isPending;
  return (
    <Card>
      <CardTitle>{t("journey.use.title")}</CardTitle>
      <p className="mb-3 text-sm text-muted">{t("journey.use.help")}</p>
      <form
        className="space-y-3"
        onSubmit={(e) => {
          e.preventDefault();
          if (form.kind === "fields") rows.mutate([rowOf(form, values)]);
          else if (form.kind === "text") texts.mutate([values.__text ?? ""]);
          else if (file) files.mutate(file);
        }}
      >
        <CaseInputs
          form={form}
          values={values}
          setValues={setValues}
          file={file}
          setFile={setFile}
        />
        <Button type="submit" loading={pending}>
          {t("journey.use.predict")}
        </Button>
      </form>
      {p && (
        <p role="status" className="mt-3">
          {t("journey.use.prediction", {
            value:
              typeof p.prediction === "number"
                ? formatNumber(p.prediction, i18n.language, 4)
                : String(p.prediction),
          })}
          {typeof p.confidence === "number" && (
            <span className="ml-2 text-muted">
              (
              {t("journey.use.confidence", {
                value: formatNumber(p.confidence * 100, i18n.language, 0),
              })}
              )
            </span>
          )}
        </p>
      )}
      <ErrorNote error={rows.error ?? texts.error ?? files.error} />
    </Card>
  );
}
