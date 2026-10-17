import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { Button, Card, ErrorNote, Field, Input, Select, Spinner, Textarea } from "@/components/ui";
import { useProjectId } from "@/features/projects/ProjectLayout";
import { getApiClient } from "@/lib/api/client";
import { unwrap, useProject, useUpload, type Schemas } from "@/lib/api/hooks";
import { postForm } from "@/lib/api/upload";

type Journey = Schemas["Journey"];
type Facts = Schemas["Facts"];
type Action = Schemas["JourneyAction"];
function normalize(j: Journey) {
  return {
    ...j,
    facts: j.facts ?? {},
    pending: j.pending ?? [],
    messages: j.messages ?? [],
    assumptions: j.assumptions ?? [],
    blockers: j.blockers ?? [],
    warnings: j.warnings ?? [],
  };
}
const choices: Record<string, string[]> = {
  modality: ["image", "other"],
  task: ["classification", "detection", "segmentation", "other"],
  grouped: ["yes", "no"],
  labels_available: ["yes", "no"],
  error_costs: ["misses", "false_alarms", "equal"],
};

function FactInput({
  name,
  facts,
  save,
  busy,
}: {
  name: string;
  facts: Facts;
  save: (facts: Facts) => void;
  busy: boolean;
}) {
  const { t } = useTranslation();
  const [value, setValue] = useState("");
  const options = name === "positive_class" ? (facts.classes ?? []) : choices[name];
  const submit = () => {
    const parsed =
      name === "classes"
        ? value
            .split(",")
            .map((v) => v.trim())
            .filter(Boolean)
        : name === "success_percent"
          ? Number(value)
          : ["grouped", "labels_available"].includes(name)
            ? value === "yes"
            : value;
    save({ [name]: parsed });
    setValue("");
  };
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
      className="space-y-3"
    >
      <Field label={t(`journey.questions.${name}`)}>
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
            type={name === "success_percent" ? "number" : "text"}
            min={1}
            max={100}
            placeholder={name === "classes" ? t("journey.classesExample") : undefined}
          />
        )}
      </Field>
      <Button type="submit" disabled={!value || busy}>
        {t("journey.answer")}
      </Button>
    </form>
  );
}

function Prediction({
  journey,
  projectId,
  onUpdate,
}: {
  journey: Journey;
  projectId: string;
  onUpdate: (j: Journey) => void;
}) {
  const { t } = useTranslation();
  const [mode, setMode] = useState<"predict" | "validate">("predict");
  const [file, setFile] = useState<File | null>(null);
  const [actual, setActual] = useState("");
  const classes = (journey.contract?.outputs as { classes?: string[] } | undefined)?.classes ?? [];
  const mutation = useMutation({
    mutationFn: async () => {
      const form = new FormData();
      if (file) form.append("file", file);
      if (mode === "validate") form.append("actual", actual);
      return postForm<Schemas["GuidedPrediction"]>(
        `/api/v1/projects/${projectId}/journey/${mode}`,
        form,
      );
    },
    onSuccess: (result) => onUpdate(result.journey),
  });
  if (journey.contract?.inputs && (journey.contract.inputs as { kind?: string }).kind !== "image")
    return null;
  const validations = journey.validations ?? [];
  const positive = journey.facts?.positive_class;
  return (
    <Card>
      <div className="mb-4 flex flex-wrap gap-2">
        {(["predict", "validate"] as const).map((m) => (
          <Button
            key={m}
            variant={mode === m ? "primary" : "secondary"}
            onClick={() => {
              setMode(m);
              mutation.reset();
            }}
          >
            {t(`journey.${m}`)}
          </Button>
        ))}
      </div>
      <p className="mb-4 text-sm text-muted">
        {t(mode === "validate" ? "journey.validationHelp" : "journey.predictionHelp")}
      </p>
      <form
        className="space-y-3"
        onSubmit={(e) => {
          e.preventDefault();
          mutation.mutate();
        }}
      >
        <Field label={t("journey.image")}>
          <input
            type="file"
            accept="image/*"
            required
            onChange={(e) => {
              setFile(e.target.files?.[0] ?? null);
              mutation.reset();
            }}
          />
        </Field>
        {mode === "validate" && (
          <Field label={t("journey.actual")}>
            <Select required value={actual} onChange={(e) => setActual(e.target.value)}>
              <option value="">{t("journey.choose")}</option>
              {classes.map((c) => (
                <option key={c}>{c}</option>
              ))}
            </Select>
          </Field>
        )}
        <Button
          type="submit"
          loading={mutation.isPending}
          disabled={!file || (mode === "validate" && !actual)}
        >
          {t(`journey.${mode}`)}
        </Button>
      </form>
      <ErrorNote error={mutation.error} />
      {mutation.data && (
        <p role="status" className="mt-4">
          {t("journey.predicted", { label: mutation.data.prediction })}
          {mutation.data.correct != null &&
            ` — ${t(mutation.data.correct ? "journey.correct" : "journey.incorrect")}`}
        </p>
      )}
      <p className="mt-3 text-sm text-muted">{t("journey.noGuarantee")}</p>
      {validations.length > 0 && (
        <p className="mt-3">
          {t("journey.counts", {
            total: validations.length,
            correct: validations.filter((v) => v.correct).length,
            misses: validations.filter((v) => v.actual === positive && !v.correct).length,
            alarms: validations.filter((v) => v.prediction === positive && !v.correct).length,
          })}
        </p>
      )}
    </Card>
  );
}

export function JourneyPage() {
  const projectId = useProjectId();
  const { t, i18n } = useTranslation();
  const project = useProject(projectId);
  const qc = useQueryClient();
  const key = ["journey", projectId];
  const query = useQuery({
    queryKey: key,
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/projects/{project_id}/journey", {
          params: { path: { project_id: projectId } },
        }),
      ),
    refetchInterval: 2000,
  });
  const update = (j: Journey) => qc.setQueryData(key, j);
  const [message, setMessage] = useState("");
  const [editField, setEditField] = useState<string | null>(null);
  const upload = useUpload(projectId);
  const action = useMutation({
    mutationFn: async (
      body: Omit<Action, "version" | "language" | "message"> & { message?: string },
    ) =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/projects/{project_id}/journey", {
          params: { path: { project_id: projectId } },
          body: {
            message: "",
            ...body,
            version: query.data?.version ?? 1,
            language: i18n.language.startsWith("en") ? "en" : "es",
          },
        }),
      ),
    onSuccess: (j) => {
      update(j);
      setMessage("");
      setEditField(null);
    },
    onSettled: () => void qc.invalidateQueries({ queryKey: key }),
  });
  const j = query.data && normalize(query.data);
  if (!j)
    return (
      <>
        <Spinner />
        <ErrorNote error={query.error} />
      </>
    );
  const busy = action.isPending || upload.isPending;
  const interviewing = j.state === "interview";
  const nextField = editField ?? j.pending[0];
  const canUpload = [
    "awaiting_data",
    "ready_to_train",
    "needs_improvement",
    "failed",
    "interrupted",
  ].includes(j.state);
  const canPredict =
    ["ready_to_validate", "ready_to_use", "needs_improvement"].includes(j.state) && j.contract;
  return (
    <div className="space-y-4">
      {j.notification && (
        <Card>
          <div role="status" className="flex items-center justify-between gap-3">
            <p>{t(`journey.notifications.${j.notification}`)}</p>
            <Button variant="secondary" onClick={() => action.mutate({ action: "acknowledge" })}>
              {t("journey.dismiss")}
            </Button>
          </div>
        </Card>
      )}
      <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,2fr)_minmax(280px,1fr)]">
        <Card>
          <h2 className="mb-4 text-xl font-semibold">{t("journey.title")}</h2>
          <p className="mb-4 text-muted">{project.data?.goal}</p>
          <div className="max-h-[55vh] space-y-3 overflow-auto" aria-live="polite">
            {j.messages.map((m, i) => (
              <div
                key={i}
                className={`rounded-pt p-3 ${m.role === "user" ? "bg-canvas" : "border border-line"}`}
              >
                <p className="text-xs text-muted">{t(`journey.${m.role}`)}</p>
                <p className="whitespace-pre-wrap">{m.text}</p>
              </div>
            ))}
          </div>
          {interviewing && (
            <form
              className="mt-4 space-y-3"
              onSubmit={(e) => {
                e.preventDefault();
                action.mutate({ action: "message", message: message || project.data?.goal || "" });
              }}
            >
              <Textarea
                aria-label={t("journey.message")}
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder={t("journey.message")}
              />
              <Button type="submit" loading={action.isPending}>
                {t(j.messages.length ? "journey.send" : "journey.begin")}
              </Button>
            </form>
          )}
          {j.assumptions.length > 0 && (
            <details className="mt-4">
              <summary>{t("journey.assumptions")}</summary>
              <ul>
                {j.assumptions.map((a, i) => (
                  <li key={i}>{a}</li>
                ))}
              </ul>
            </details>
          )}
          {interviewing && (
            <details className="mt-4">
              <summary>{t("journey.facts")}</summary>
              <dl className="mt-2 space-y-2">
                {Object.entries(j.facts)
                  .filter(([, v]) => v != null)
                  .map(([k, v]) => (
                    <div key={k}>
                      <dt className="text-sm text-muted">{t(`journey.questions.${k}`)}</dt>
                      <dd>
                        {Array.isArray(v)
                          ? v.join(", ")
                          : typeof v === "boolean"
                            ? t(v ? "journey.options.yes" : "journey.options.no")
                            : t(`journey.options.${String(v)}`, { defaultValue: String(v) })}
                        <Button variant="ghost" onClick={() => setEditField(k)}>
                          {t("journey.edit")}
                        </Button>
                      </dd>
                    </div>
                  ))}
              </dl>
            </details>
          )}
          <ErrorNote error={action.error ?? upload.error ?? query.error} />
        </Card>
        <Card>
          <h3 className="mb-2 font-semibold" role="status">
            {t(`journey.states.${j.state}`)}
          </h3>
          <p className="mb-4 text-sm text-muted">{t(`journey.stages.${j.stage}`)}</p>
          {j.llm_failed && interviewing && <p className="mb-3 text-sm">{t("journey.fallback")}</p>}
          {interviewing && nextField && (
            <FactInput
              key={nextField}
              name={nextField}
              facts={j.facts}
              busy={busy}
              save={(facts) =>
                action.mutate({
                  action: "message",
                  facts,
                  message: `${t(`journey.questions.${nextField}`)} ${Object.values(facts)
                    .map((v) =>
                      Array.isArray(v)
                        ? v.join(", ")
                        : typeof v === "boolean"
                          ? t(v ? "journey.options.yes" : "journey.options.no")
                          : t(`journey.options.${String(v)}`, { defaultValue: String(v) }),
                    )
                    .join(", ")}`,
                })
              }
            />
          )}
          {interviewing && !j.pending.length && (
            <div className="space-y-3">
              <p>
                {t("journey.summary", {
                  classes: j.facts.classes?.join(", "),
                  positive: j.facts.positive_class,
                  percent: j.facts.success_percent,
                  grouped: t(j.facts.grouped ? "journey.options.yes" : "journey.options.no"),
                  costs: t(`journey.options.${j.facts.error_costs}`),
                })}
              </p>
              <Button loading={busy} onClick={() => action.mutate({ action: "confirm" })}>
                {t("journey.confirm")}
              </Button>
            </div>
          )}
          {j.state === "unsupported" && (
            <>
              <p className="mb-3">{t("journey.unsupported")}</p>
              <Link to="/projects/$projectId/wizard" params={{ projectId }}>
                {t("journey.advanced")}
              </Link>
            </>
          )}
          {j.blockers.length > 0 && (
            <div className="my-3">
              <h4 className="font-semibold">{t("journey.blockers")}</h4>
              <ul>
                {j.blockers.map((b) => (
                  <li key={b}>{b}</li>
                ))}
              </ul>
            </div>
          )}
          {j.warnings.length > 0 && (
            <div className="my-3">
              <h4 className="font-semibold">{t("journey.warnings")}</h4>
              <ul>
                {j.warnings.map((w) => (
                  <li key={w}>{w}</li>
                ))}
              </ul>
            </div>
          )}
          {canUpload && (
            <div className="space-y-3">
              <p>{t("journey.dataHelp", { classes: j.facts.classes?.join(", ") })}</p>
              <pre className="overflow-auto rounded bg-canvas p-2 text-xs">
                {j.facts.classes
                  ?.map((c) => `${c}/${j.facts.grouped ? "objeto_01/" : ""}imagen_01.jpg`)
                  .join("\n")}
              </pre>
              {j.facts.grouped && <p>{t("journey.groupHelp")}</p>}
              {!j.facts.labels_available && <p>{t("journey.labelHelp")}</p>}
              <Field label={t("journey.upload")}>
                <input
                  type="file"
                  accept=".zip"
                  disabled={busy}
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (!file) return;
                    upload.mutate([file], {
                      onSuccess: (data) =>
                        action.mutate({ action: "prepare", source_id: data.source.id }),
                    });
                    e.target.value = "";
                  }}
                />
              </Field>
              {upload.isPending && <p role="status">{t("journey.uploading")}</p>}
            </div>
          )}
          {["ready_to_train", "failed", "interrupted"].includes(j.state) &&
            j.dataset_id &&
            !j.blockers.length && (
              <Button
                className="mt-4"
                loading={busy}
                onClick={() => action.mutate({ action: "train" })}
              >
                {t("journey.train")}
              </Button>
            )}
          {["training", "reviewing_data"].includes(j.state) && (
            <Button disabled={j.stop_requested} onClick={() => action.mutate({ action: "stop" })}>
              {t(j.stop_requested ? "journey.stopping" : "journey.stop")}
            </Button>
          )}
          {j.error && (
            <p role="alert" className="mt-3">
              {j.error}
            </p>
          )}
          {!["training", "reviewing_data", "interview"].includes(j.state) && (
            <Button
              variant="secondary"
              className="mt-4"
              disabled={busy}
              onClick={() => action.mutate({ action: "revise" })}
            >
              {t("journey.revise")}
            </Button>
          )}
          {j.result && (
            <div className="mt-4 space-y-3">
              <p>
                {t("journey.counts", {
                  total: j.result.total,
                  correct: j.result.correct,
                  misses: j.result.misses,
                  alarms: j.result.false_alarms,
                })}
              </p>
              <p>{t(j.result.meets_target ? "journey.targetMet" : "journey.improve")}</p>
            </div>
          )}
        </Card>
      </div>
      {canPredict && <Prediction journey={j} projectId={projectId} onUpdate={update} />}
    </div>
  );
}
