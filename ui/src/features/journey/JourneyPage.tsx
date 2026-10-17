/**
 * Experiencia principal: un asistente guía al usuario sin conocimientos de ML desde la
 * descripción del problema hasta usar el modelo. Cada estado muestra solo lo necesario;
 * las herramientas técnicas quedan en «Opciones avanzadas».
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import {
  Button,
  Card,
  CardTitle,
  ErrorNote,
  Field,
  Select,
  Spinner,
  Textarea,
} from "@/components/ui";
import { useProjectId } from "@/features/projects/ProjectLayout";
import { getApiClient } from "@/lib/api/client";
import { type Schemas, unwrap, useUpload } from "@/lib/api/hooks";

import {
  DataSummary,
  FactControl,
  GuideView,
  type Journey,
  Notes,
  Results,
  TrainingStages,
  UsePanel,
  ValidatePanel,
} from "./JourneyPanels";
import { factValue, fieldLabel } from "./journeyText";

type Action = Omit<Schemas["JourneyAction"], "version" | "language" | "message"> & {
  message?: string;
};
type Step = "case" | "data" | "review" | "training" | "results" | "validate" | "use";

const STEPS: Step[] = ["case", "data", "review", "training", "validate", "use"];
const ACTIVE = new Set(["reviewing_data", "training"]);
const MODEL = new Set(["ready_to_validate", "needs_improvement", "ready_to_use"]);
const RETRY = new Set(["ready_to_train", "failed", "interrupted", "needs_improvement"]);

function stepOf(j: Journey): Step {
  if (j.state === "interview" || j.state === "unsupported") return "case";
  if (j.state === "awaiting_data") return "data";
  if (j.state === "reviewing_data" || j.state === "ready_to_train") return "review";
  if (j.state === "training") return "training";
  if (j.state === "ready_to_use") return "use";
  if (MODEL.has(j.state)) return "validate";
  return j.dataset_id ? "review" : "data";
}

function Stepper({ current }: { current: Step }) {
  const { t } = useTranslation();
  const index = STEPS.indexOf(current);
  return (
    <ol className="flex flex-wrap gap-2 text-xs" aria-label={t("journey.title")}>
      {STEPS.map((s, i) => (
        <li
          key={s}
          aria-current={i === index ? "step" : undefined}
          className={`rounded-full border px-3 py-1 ${
            i === index
              ? "border-brand bg-brand text-white"
              : i < index
                ? "border-brand text-brand"
                : "border-line text-muted"
          }`}
        >
          {i + 1}. {t(`journey.steps.${s}`)}
        </li>
      ))}
    </ol>
  );
}

function Conversation({ journey }: { journey: Journey }) {
  const { t } = useTranslation();
  return (
    <div className="max-h-[50vh] space-y-3 overflow-auto" aria-live="polite">
      {(journey.messages ?? []).map((m, i) => (
        <div
          key={i}
          className={`rounded-pt p-3 ${m.role === "user" ? "ml-8 bg-canvas" : "mr-8 border border-line"}`}
        >
          <p className="text-xs text-muted">{t(`journey.${m.role}`)}</p>
          <p className="whitespace-pre-wrap">{m.text}</p>
        </div>
      ))}
    </div>
  );
}

export function JourneyPage() {
  const projectId = useProjectId();
  const { t, i18n } = useTranslation();
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
    refetchInterval: (q) => (q.state.data && ACTIVE.has(q.state.data.state) ? 2000 : false),
  });
  const update = (j: Journey) => qc.setQueryData(key, j);
  const [message, setMessage] = useState("");
  const [editing, setEditing] = useState<string | null>(null);
  const [target, setTarget] = useState("");
  const upload = useUpload(projectId);
  const action = useMutation({
    mutationFn: async (body: Action) =>
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
      setEditing(null);
    },
    onError: () => void qc.invalidateQueries({ queryKey: key }),
  });
  const j = query.data;
  if (!j)
    return (
      <>
        <Spinner />
        <ErrorNote error={query.error} />
      </>
    );
  const facts = j.facts ?? {};
  const pending = j.pending ?? [];
  const busy = action.isPending || upload.isPending;
  const step = stepOf(j);
  const next = editing ?? pending[0];
  const send = (body: Action) => action.mutate(body);
  const sendFacts = (f: Schemas["Facts"], text: string) =>
    send({ action: "message", message: text, facts: f });
  const advancedLink = (
    <Link to="/projects/$projectId/wizard" params={{ projectId }} className="text-brand underline">
      {t("journey.toAdvanced")}
    </Link>
  );
  const uploadData = (files: File[]) =>
    upload.mutate(files, {
      onSuccess: (data) => send({ action: "prepare", source_id: data.source.id }),
    });

  return (
    <div className="mx-auto max-w-3xl space-y-4">
      <Stepper current={step} />
      {j.notification && (
        <Card>
          <div role="status" className="flex items-center justify-between gap-3">
            <p>{t(`journey.notifications.${j.notification}`)}</p>
            <Button variant="secondary" size="sm" onClick={() => send({ action: "acknowledge" })}>
              {t("journey.dismiss")}
            </Button>
          </div>
        </Card>
      )}
      {j.error && (
        <p role="alert" className="rounded-pt border border-bad p-3 text-sm">
          {j.error}
        </p>
      )}

      {step === "case" && (
        <Card>
          <CardTitle>{t("journey.title")}</CardTitle>
          {!(j.messages ?? []).length && (
            <p className="mb-3 text-sm text-muted">{t("journey.example")}</p>
          )}
          <Conversation journey={j} />
          {j.state === "unsupported" ? (
            <div className="mt-4 space-y-3">
              <p className="font-semibold">{t("journey.unsupportedTitle")}</p>
              <p className="text-sm">{j.unsupported}</p>
              <div className="flex flex-wrap items-center gap-3">
                {advancedLink}
                <Button
                  variant="secondary"
                  size="sm"
                  disabled={busy}
                  onClick={() => send({ action: "revise" })}
                >
                  {t("journey.revise")}
                </Button>
              </div>
            </div>
          ) : (
            <>
              <form
                className="mt-4 space-y-2"
                onSubmit={(e) => {
                  e.preventDefault();
                  if (message.trim()) send({ action: "message", message });
                }}
              >
                <Textarea
                  aria-label={t("journey.message")}
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  placeholder={t("journey.message")}
                />
                <Button type="submit" loading={action.isPending} disabled={!message.trim()}>
                  {t((j.messages ?? []).length ? "journey.send" : "journey.begin")}
                </Button>
              </form>
              {j.llm_failed && <p className="mt-3 text-sm text-warn">{t("journey.fallback")}</p>}
              {next && (j.messages ?? []).length > 0 && (
                <div className="mt-4 rounded-pt border border-line p-3">
                  <FactControl
                    key={next}
                    name={next}
                    facts={facts}
                    busy={busy}
                    onSave={sendFacts}
                  />
                </div>
              )}
              {Object.values(facts).some((v) => v != null) && (
                <details className="mt-4" open={!pending.length}>
                  <summary className="cursor-pointer font-semibold">{t("journey.facts")}</summary>
                  <dl className="mt-2 space-y-1 text-sm">
                    {Object.entries(facts)
                      .filter(([, v]) => v != null)
                      .map(([k, v]) => (
                        <div key={k} className="flex flex-wrap items-baseline gap-2">
                          <dt className="text-muted">{fieldLabel(t, k, facts)}</dt>
                          <dd className="font-medium">{factValue(t, v)}</dd>
                          <Button variant="ghost" size="sm" onClick={() => setEditing(k)}>
                            {t("journey.edit")}
                          </Button>
                        </div>
                      ))}
                  </dl>
                </details>
              )}
              {(j.assumptions ?? []).length > 0 && (
                <details className="mt-3 text-sm">
                  <summary className="cursor-pointer">{t("journey.assumptions")}</summary>
                  <ul className="mt-1 list-disc pl-5">
                    {(j.assumptions ?? []).map((a) => (
                      <li key={a}>{a}</li>
                    ))}
                  </ul>
                </details>
              )}
              {!pending.length && (j.messages ?? []).length > 0 && (
                <div className="mt-4 space-y-2">
                  <p className="text-sm">{t("journey.confirmHint")}</p>
                  <Button loading={busy} onClick={() => send({ action: "confirm" })}>
                    {t("journey.confirm")}
                  </Button>
                </div>
              )}
            </>
          )}
        </Card>
      )}

      {step === "data" && j.guide && (
        <Card>
          <CardTitle>{t("journey.data.title")}</CardTitle>
          <GuideView guide={j.guide} />
          <Notes title={t("journey.blockers")} items={j.blockers ?? []} tone="bad" />
          <div className="mt-4">
            <Field label={t("journey.data.upload")}>
              <input
                type="file"
                multiple={j.guide.layout === "table"}
                accept={(j.guide.accept ?? []).join(",")}
                disabled={busy}
                aria-label={t("journey.data.upload")}
                onChange={(e) => {
                  const files = Array.from(e.target.files ?? []);
                  if (files.length) uploadData(files);
                  e.target.value = "";
                }}
              />
            </Field>
            {busy && <p role="status">{t("journey.data.uploading")}</p>}
          </div>
          {j.source_id && (j.columns ?? []).length > 0 && j.guide.layout === "table" && (
            <form
              className="mt-4 flex flex-wrap items-end gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                send({ action: "prepare", source_id: j.source_id ?? "", target });
              }}
            >
              <Field label={t("journey.data.chooseTarget")}>
                <Select required value={target} onChange={(e) => setTarget(e.target.value)}>
                  <option value="">{t("journey.choose")}</option>
                  {(j.columns ?? []).map((c) => (
                    <option key={c}>{c}</option>
                  ))}
                </Select>
              </Field>
              <Button type="submit" size="sm" disabled={!target || busy}>
                {t("journey.data.reload")}
              </Button>
            </form>
          )}
          <Button
            variant="ghost"
            size="sm"
            className="mt-4"
            disabled={busy}
            onClick={() => send({ action: "revise" })}
          >
            {t("journey.revise")}
          </Button>
        </Card>
      )}

      {step === "review" && (
        <Card>
          <CardTitle>{t("journey.review.title")}</CardTitle>
          {j.state === "reviewing_data" ? (
            <p role="status" className="flex items-center gap-2">
              <Spinner /> {t("journey.stages.reviewing_data")}
            </p>
          ) : (
            <div className="space-y-3">
              {j.data_summary && <DataSummary summary={j.data_summary} />}
              <Notes title={t("journey.blockers")} items={j.blockers ?? []} tone="bad" />
              <Notes title={t("journey.warnings")} items={j.warnings ?? []} tone="warn" />
              {!(j.blockers ?? []).length && RETRY.has(j.state) && (
                <>
                  <p className="text-sm text-muted">{t("journey.review.trainHint")}</p>
                  <Button loading={busy} onClick={() => send({ action: "train" })}>
                    {j.state === "ready_to_train"
                      ? t("journey.review.train")
                      : t("journey.training.retry")}
                  </Button>
                </>
              )}
              <div>
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={busy}
                  onClick={() => send({ action: "revise" })}
                >
                  {t("journey.review.otherData")}
                </Button>
              </div>
            </div>
          )}
        </Card>
      )}

      {step === "training" && (
        <Card>
          <CardTitle>{t("journey.training.title")}</CardTitle>
          <TrainingStages journey={j} />
          <p className="mt-3 text-sm text-muted">{t("journey.training.leave")}</p>
          <Button
            variant="secondary"
            size="sm"
            className="mt-3"
            disabled={j.stop_requested || busy}
            onClick={() => send({ action: "stop" })}
          >
            {t(j.stop_requested ? "journey.training.stopping" : "journey.training.stop")}
          </Button>
        </Card>
      )}

      {(step === "validate" || step === "use") && j.result && (
        <Card>
          <CardTitle>{t("journey.results.title")}</CardTitle>
          <Results journey={j} />
          <div className="mt-3 flex flex-wrap gap-2">
            {j.state === "needs_improvement" && (
              <Button size="sm" loading={busy} onClick={() => send({ action: "train" })}>
                {t("journey.results.retrain")}
              </Button>
            )}
            <Button
              variant="ghost"
              size="sm"
              disabled={busy}
              onClick={() => send({ action: "revise" })}
            >
              {t("journey.revise")}
            </Button>
          </div>
        </Card>
      )}
      {(step === "validate" || step === "use") && j.form && j.run_id && (
        <>
          {step === "validate" && (
            <ValidatePanel journey={j} projectId={projectId} onUpdate={update} />
          )}
          {step === "use" && <UsePanel journey={j} />}
          {step === "use" && (
            <details>
              <summary className="cursor-pointer text-sm text-muted">
                {t("journey.validate.title")}
              </summary>
              <div className="mt-2">
                <ValidatePanel journey={j} projectId={projectId} onUpdate={update} />
              </div>
            </details>
          )}
        </>
      )}

      {step !== "case" && (j.messages ?? []).length > 0 && (
        <details className="text-sm">
          <summary className="cursor-pointer text-muted">{t("journey.showConversation")}</summary>
          <div className="mt-2">
            <Conversation journey={j} />
          </div>
        </details>
      )}
      <ErrorNote error={action.error ?? upload.error ?? query.error} />
    </div>
  );
}
