/**
 * Experiencia principal (ADR-0042): una conversación con el asistente de punta a punta. El
 * asistente pregunta, acusa cada respuesta, cuenta cada etapa (datos, revisión, entrenamiento,
 * validación) y siempre se le puede escribir. Lo necesario para la acción actual aparece
 * adjunto al final del hilo; las herramientas técnicas quedan en «Opciones avanzadas».
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { Bell } from "lucide-react";
import { Fragment, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import { Button, Card, ErrorNote, Field, Select, Spinner } from "@/components/ui";
import { useProjectId } from "@/features/projects/ProjectLayout";
import { getApiClient } from "@/lib/api/client";
import { type Schemas, unwrap, useUpload } from "@/lib/api/hooks";
import { postForm } from "@/lib/api/upload";

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
import { FIRST_FILES, FIRST_MESSAGE } from "./journeyStart";
import { Bubble, Chips, Composer, DropZone, GoalCards, Typing } from "./JourneyThread";
import { factValue, fieldLabel, type QuickReply, quickReplies } from "./journeyText";

type Action = Omit<Schemas["JourneyAction"], "version" | "language" | "message"> & {
  message?: string;
};
type Step = "case" | "data" | "review" | "training" | "validate" | "use";

const STEPS: Step[] = ["case", "data", "review", "training", "validate", "use"];
const ACTIVE = new Set(["reviewing_data", "training"]);
const MODEL = new Set(["ready_to_validate", "needs_improvement", "ready_to_use"]);
const TRAIN = new Set(["ready_to_train", "failed", "interrupted", "needs_improvement"]);

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
    <ol className="flex flex-wrap gap-x-3 gap-y-1 text-xs" aria-label={t("journey.progress")}>
      {STEPS.map((s, i) => (
        <li
          key={s}
          aria-current={i === index ? "step" : undefined}
          className={
            i === index ? "font-semibold text-brand" : i < index ? "text-ink" : "text-muted"
          }
        >
          {i < index ? "✓ " : ""}
          {t(`journey.steps.${s}`)}
        </li>
      ))}
    </ol>
  );
}

/** Aviso en la pestaña y, si la persona lo permitió, notificación del sistema al terminar. */
function useFinishedNotice(j: Journey | undefined) {
  const { t } = useTranslation();
  const original = useRef<string | null>(null);
  const shown = useRef<string | null>(null);
  const notification = j?.notification ?? null;
  useEffect(() => {
    original.current ??= document.title;
    const base = original.current;
    document.title = notification ? `● ${base}` : base;
    if (!notification || shown.current === notification) return;
    shown.current = notification;
    try {
      if (
        typeof Notification !== "undefined" &&
        Notification.permission === "granted" &&
        document.hidden
      )
        new Notification(base, { body: t(`journey.notifications.${notification}`) });
    } catch {
      // Sin notificaciones del sistema: queda el aviso en la pestaña y en la conversación.
    }
  }, [notification, t]);
  useEffect(
    () => () => {
      if (original.current) document.title = original.current;
    },
    [],
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
  const [sending, setSending] = useState<string | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [showFacts, setShowFacts] = useState(false);
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
            version: qc.getQueryData<Journey>(key)?.version ?? 1,
            language: i18n.language.startsWith("en") ? "en" : "es",
          },
        }),
      ),
    onMutate: (body) => {
      if (body.action === "message" && body.message) setSending(body.message);
    },
    onSuccess: (j) => {
      update(j);
      setEditing(null);
    },
    onError: () => void qc.invalidateQueries({ queryKey: key }),
    onSettled: () => setSending(null),
  });
  const send = (body: Action) => action.mutate(body);
  const say = (text: string) => send({ action: "message", message: text });
  // Archivos antes de contar el caso: se suben, el asistente los lee y propone objetivos.
  const files = useMutation({
    mutationFn: async ({ list }: { list: File[]; message?: string }) => {
      const form = new FormData();
      for (const f of list) form.append("files", f, f.webkitRelativePath || f.name);
      return postForm<Schemas["DataSource"]>(`/api/v1/projects/${projectId}/uploads`, form);
    },
    onMutate: ({ message }) => {
      if (message) setSending(message);
    },
    onSuccess: (source, { message }) =>
      send({ action: "files", source_id: source.id ?? "", message: message ?? "" }),
    onError: () => setSending(null),
  });
  const attach = (list: File[], message?: string) =>
    files.mutate(message ? { list, message } : { list });
  const j = query.data;
  useFinishedNotice(j);

  // Lo que la persona escribió (o adjuntó) en Inicio se envía acá, con el «escribiendo…».
  const firstSent = useRef(false);
  useEffect(() => {
    if (!j || firstSent.current || (j.messages ?? []).length) return;
    let first: string | null;
    try {
      first = sessionStorage.getItem(FIRST_MESSAGE(projectId));
      sessionStorage.removeItem(FIRST_MESSAGE(projectId));
    } catch {
      first = null;
    }
    const attached = FIRST_FILES.get(projectId);
    FIRST_FILES.delete(projectId);
    if (attached?.length) {
      firstSent.current = true;
      attach(attached, first ?? undefined);
    } else if (first) {
      firstSent.current = true;
      say(first);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [j, projectId]);

  // El aviso de fin se da por visto cuando la persona tiene la conversación a la vista.
  const notification = j?.notification;
  useEffect(() => {
    if (!notification) return;
    const seen = () => {
      if (!document.hidden && !action.isPending) send({ action: "acknowledge" });
    };
    seen();
    document.addEventListener("visibilitychange", seen);
    return () => document.removeEventListener("visibilitychange", seen);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [notification]);

  const end = useRef<HTMLDivElement>(null);
  const count = (j?.messages ?? []).length;
  useEffect(() => {
    end.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [count, sending, j?.state]);

  if (!j)
    return (
      <>
        <Spinner />
        <ErrorNote error={query.error} />
      </>
    );

  const facts = j.facts ?? {};
  const pending = j.pending ?? [];
  const messages = j.messages ?? [];
  const reading = files.isPending || (action.isPending && action.variables?.action === "files");
  const busy = action.isPending || upload.isPending || files.isPending;
  const step = stepOf(j);
  const last = messages.at(-1);
  const lastSaid = last?.role === "assistant" ? last : undefined;
  const interviewing = j.state === "interview";
  const telling = interviewing || j.state === "unsupported";
  const canTrain = TRAIN.has(j.state) && Boolean(j.dataset_id) && !(j.blockers ?? []).length;
  // Los objetivos que propuso al leer los archivos, si todavía no eligió ninguno.
  const goals =
    telling && !j.goal_id
      ? (lastSaid?.goals ?? [])
          .map((id) => (j.goals ?? []).find((g) => g.id === id))
          .filter((g) => g !== undefined)
      : [];

  // Respuestas rápidas: las que propuso el LLM (texto libre) o las del dato que se pregunta.
  const field = editing ?? (interviewing ? lastSaid?.field : null);
  const replies: QuickReply[] = !field
    ? []
    : !editing && lastSaid?.options?.length
      ? lastSaid.options.map((label) => ({ label }))
      : quickReplies(t, field, facts);
  const pick = (r: QuickReply) =>
    send({ action: "message", message: r.label, ...(r.facts ? { facts: r.facts } : {}) });
  const choose = (id: string) => {
    setSending((j.goals ?? []).find((g) => g.id === id)?.title ?? null);
    send({ action: "choose", goal_id: id });
  };

  const offered = lastSaid?.action;
  const actionButtons = (
    <div className="mt-3 flex flex-wrap gap-2">
      {offered === "confirm" && interviewing && !pending.length && (
        <>
          <Button size="sm" loading={busy} onClick={() => send({ action: "confirm" })}>
            {t("journey.confirm")}
          </Button>
          <Button size="sm" variant="secondary" onClick={() => setShowFacts(true)}>
            {t("journey.correct")}
          </Button>
        </>
      )}
      {offered === "train" && canTrain && (
        <Button size="sm" loading={busy} onClick={() => send({ action: "train" })}>
          {j.state === "ready_to_train" ? t("journey.review.train") : t("journey.training.retry")}
        </Button>
      )}
      {offered === "revise" && !ACTIVE.has(j.state) && (
        <Button
          size="sm"
          variant="secondary"
          disabled={busy}
          onClick={() => send({ action: "revise" })}
        >
          {t("journey.revise")}
        </Button>
      )}
      {offered === "stop" && j.state === "training" && (
        <Button
          size="sm"
          variant="secondary"
          disabled={busy || j.stop_requested}
          onClick={() => send({ action: "stop" })}
        >
          {t("journey.training.stop")}
        </Button>
      )}
      {offered === "advanced" && (
        <Link
          to="/projects/$projectId/wizard"
          params={{ projectId }}
          className="text-sm text-brand underline"
        >
          {t("journey.toAdvanced")}
        </Link>
      )}
    </div>
  );

  const factsPanel = Object.values(facts).some((v) => v != null) && step === "case" && (
    <details className="text-sm" open={showFacts || Boolean(editing) || undefined}>
      <summary className="cursor-pointer text-muted">{t("journey.facts")}</summary>
      <dl className="mt-2 space-y-1">
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
      {editing && !replies.length && (
        <div className="mt-2">
          <FactControl
            key={editing}
            name={editing}
            facts={facts}
            busy={busy}
            onSave={(f, text) => send({ action: "message", message: text, facts: f })}
          />
        </div>
      )}
    </details>
  );

  // Lo que hace falta para la acción actual va pegado al mensaje que lo presentó: las
  // preguntas libres posteriores (y sus respuestas) quedan debajo, como en cualquier chat.
  const anchor = messages.findLastIndex((m) => m.step === step && !m.chat);
  const attachment = (() => {
    if (step === "data" && j.guide)
      return (
        <Card>
          <GuideView guide={j.guide} />
          <div className="mt-4">
            <DropZone
              label={t("journey.data.upload")}
              hint={t("journey.data.drop", { formats: (j.guide.accept ?? []).join(", ") })}
              accept={(j.guide.accept ?? []).join(",")}
              multiple={j.guide.layout === "table"}
              disabled={busy}
              onFiles={(files) =>
                upload.mutate(files, {
                  onSuccess: (data) => send({ action: "prepare", source_id: data.source.id }),
                })
              }
            />
            {upload.isPending && <p role="status">{t("journey.data.uploading")}</p>}
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
        </Card>
      );
    if (step === "review")
      return j.state === "reviewing_data" ? (
        <p role="status" className="flex items-center gap-2 text-sm text-muted">
          <Spinner /> {t("journey.stages.reviewing_data")}
        </p>
      ) : (
        <Card>
          {j.data_summary && <DataSummary summary={j.data_summary} />}
          <div className="mt-3 space-y-3">
            <Notes title={t("journey.warnings")} items={j.warnings ?? []} tone="warn" />
          </div>
          {canTrain && offered !== "train" && (
            <Button
              className="mt-3"
              size="sm"
              loading={busy}
              onClick={() => send({ action: "train" })}
            >
              {j.state === "ready_to_train"
                ? t("journey.review.train")
                : t("journey.training.retry")}
            </Button>
          )}
        </Card>
      );
    if (step === "training")
      return (
        <Card>
          <TrainingStages journey={j} />
          <div className="mt-3 flex flex-wrap gap-2">
            {typeof Notification !== "undefined" && Notification.permission === "default" && (
              <Button
                size="sm"
                variant="secondary"
                onClick={() => void Notification.requestPermission()}
              >
                <Bell className="h-4 w-4" aria-hidden="true" />
                {t("journey.training.notify")}
              </Button>
            )}
            <Button
              size="sm"
              variant="ghost"
              disabled={j.stop_requested || busy}
              onClick={() => send({ action: "stop" })}
            >
              {t(j.stop_requested ? "journey.training.stopping" : "journey.training.stop")}
            </Button>
          </div>
        </Card>
      );
    if ((step === "validate" || step === "use") && j.form && j.run_id)
      return (
        <div className="space-y-3">
          {j.result && <Results journey={j} compact />}
          {step === "validate" ? (
            <ValidatePanel journey={j} projectId={projectId} onUpdate={update} />
          ) : (
            <>
              <UsePanel journey={j} projectId={projectId} />
              <details>
                <summary className="cursor-pointer text-sm text-muted">
                  {t("journey.validate.title")}
                </summary>
                <div className="mt-2">
                  <ValidatePanel journey={j} projectId={projectId} onUpdate={update} />
                </div>
              </details>
            </>
          )}
        </div>
      );
    return null;
  })();

  return (
    <div className="flex max-w-3xl flex-col gap-4">
      <Stepper current={step} />
      <section aria-label={t("journey.conversation")} aria-live="polite" className="space-y-3">
        {!messages.length && !sending && (
          <Bubble turn={{ role: "assistant", text: t("journey.welcome") }} />
        )}
        {messages.map((m, i) => (
          <Fragment key={i}>
            <Bubble turn={m}>{m === lastSaid && !sending && actionButtons}</Bubble>
            {i === anchor && attachment}
          </Fragment>
        ))}
        {sending && !(last?.role === "user" && last.text === sending) && (
          <Bubble turn={{ role: "user", text: sending }} />
        )}
        {files.isPending && (
          <Bubble
            turn={{
              role: "user",
              text: (files.variables?.list ?? []).map((f) => f.name).join(", "),
            }}
          />
        )}
        {reading ? <Typing label={t("journey.files.reading")} /> : sending && <Typing />}
        {!sending && !reading && <GoalCards goals={goals} disabled={busy} onPick={choose} />}
        {!sending && <Chips replies={replies} disabled={busy} onPick={pick} />}
        {factsPanel}
        {anchor < 0 && !sending && attachment}
        <div ref={end} className="scroll-mb-40" />
      </section>
      <div className="sticky bottom-0 space-y-1 bg-canvas pb-4 pt-2">
        <Composer
          placeholder={t(`journey.composer.${step}`)}
          disabled={busy}
          onSend={say}
          {...(telling ? { onFiles: (list: File[]) => attach(list) } : {})}
          autoFocus={step === "case"}
        />
        {j.llm_failed && <p className="px-2 text-xs text-muted">{t("journey.noAi")}</p>}
        <ErrorNote error={action.error ?? upload.error ?? files.error ?? query.error} />
      </div>
    </div>
  );
}
