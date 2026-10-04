/**
 * Wizard adaptativo (ADR-0040): ficha del caso editable, entrevista con el asistente y el plan
 * (chequeos y defaults sugeridos). El LLM solo propone: cada cambio a la ficha se acepta o se
 * descarta; sin LLM la ficha se completa a mano y el plan se calcula igual.
 */
import { AlertTriangle, Info, OctagonAlert, Sparkles } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { Badge, Button, Card, ErrorNote, Field, Input, Select, Textarea } from "@/components/ui";
import {
  type BriefPatch,
  type IntakeTurn,
  type UseCaseBrief,
  type WizardPlan,
  useDraftIntake,
} from "@/lib/api/hooks";

type SaveBrief = (brief: UseCaseBrief, origin?: "user" | "copilot") => void;

const PROBLEMS = ["value", "category", "anomaly", "forecast", "rule", "other"] as const;
const COSTS = ["symmetric", "false_negative_worse", "false_positive_worse"] as const;
const DEPLOYMENTS = ["desktop", "server", "edge", "spreadsheet"] as const;
const FLAGS = [
  "has_time",
  "has_entities",
  "labels_available",
  "independent_inputs",
  "extrapolate",
  "explainability",
] as const;
type Flag = (typeof FLAGS)[number];

const SEVERITY_ICON = { info: Info, warning: AlertTriangle, high: OctagonAlert } as const;
const SEVERITY_TONE = { info: "neutral", warning: "warn", high: "bad" } as const;

function valueLabel(t: (k: string) => string, field: string, value: unknown): string {
  if (value === true) return t("wizard.brief.yes");
  if (value === false) return t("wizard.brief.no");
  if (field === "problem" || field === "error_costs" || field === "deployment")
    return t(`wizard.brief.${field}.${String(value)}`);
  return String(value);
}

export function BriefPanel({
  projectId,
  brief,
  save,
}: {
  projectId: string;
  brief: UseCaseBrief;
  save: SaveBrief;
}) {
  const { t } = useTranslation();
  const set = (patch: Partial<UseCaseBrief>) => save({ ...brief, ...patch });
  const flag = (f: Flag) => (brief[f] === true ? "yes" : brief[f] === false ? "no" : "");

  return (
    <div className="space-y-4">
      <IntakeChat projectId={projectId} brief={brief} save={save} />
      <Card>
        <h3 className="mb-1 font-semibold">{t("wizard.brief.title")}</h3>
        <p className="mb-3 text-xs text-muted">{t("wizard.brief.hint")}</p>
        <div className="grid gap-3 sm:grid-cols-2">
          <Field label={t("wizard.brief.problem.label")}>
            <Select
              value={brief.problem ?? ""}
              onChange={(e) =>
                set({
                  problem: (e.target.value || null) as NonNullable<UseCaseBrief["problem"]> | null,
                })
              }
            >
              <option value="">—</option>
              {PROBLEMS.map((p) => (
                <option key={p} value={p}>
                  {t(`wizard.brief.problem.${p}`)}
                </option>
              ))}
            </Select>
          </Field>
          {brief.problem === "other" ? (
            <Field label={t("wizard.brief.problemOther")}>
              <Input
                defaultValue={brief.problem_other ?? ""}
                onBlur={(e) => set({ problem_other: e.target.value || null })}
              />
            </Field>
          ) : (
            <Field label={t("wizard.brief.prediction")}>
              <Input
                defaultValue={brief.prediction ?? ""}
                onBlur={(e) => set({ prediction: e.target.value || null })}
              />
            </Field>
          )}
          <Field label={t("wizard.brief.error_costs.label")}>
            <Select
              value={brief.error_costs ?? ""}
              onChange={(e) =>
                set({
                  error_costs: (e.target.value || null) as NonNullable<
                    UseCaseBrief["error_costs"]
                  > | null,
                })
              }
            >
              <option value="">—</option>
              {COSTS.map((c) => (
                <option key={c} value={c}>
                  {t(`wizard.brief.error_costs.${c}`)}
                </option>
              ))}
            </Select>
          </Field>
          {brief.error_costs && brief.error_costs !== "symmetric" && (
            <Field label={t("wizard.brief.ratio")}>
              <Input
                type="number"
                min={1}
                step="any"
                defaultValue={brief.error_cost_ratio ?? ""}
                onBlur={(e) =>
                  set({ error_cost_ratio: e.target.value === "" ? null : Number(e.target.value) })
                }
              />
            </Field>
          )}
          {FLAGS.map((f) => (
            <Field key={f} label={t(`wizard.brief.${f}`)}>
              <Select
                value={flag(f)}
                onChange={(e) =>
                  set({ [f]: e.target.value === "" ? null : e.target.value === "yes" })
                }
              >
                <option value="">{t("wizard.brief.unknown")}</option>
                <option value="yes">{t("wizard.brief.yes")}</option>
                <option value="no">{t("wizard.brief.no")}</option>
              </Select>
            </Field>
          ))}
          <Field label={t("wizard.brief.deployment.label")}>
            <Select
              value={brief.deployment ?? ""}
              onChange={(e) =>
                set({
                  deployment: (e.target.value || null) as NonNullable<
                    UseCaseBrief["deployment"]
                  > | null,
                })
              }
            >
              <option value="">—</option>
              {DEPLOYMENTS.map((d) => (
                <option key={d} value={d}>
                  {t(`wizard.brief.deployment.${d}`)}
                </option>
              ))}
            </Select>
          </Field>
        </div>
        {(brief.assumptions ?? []).length > 0 && (
          <div className="mt-3 text-sm">
            <p className="text-muted">{t("wizard.brief.assumptions")}</p>
            <ul className="list-disc pl-5">
              {(brief.assumptions ?? []).map((a) => (
                <li key={a.text}>
                  {a.text}{" "}
                  <span className="text-xs text-muted">({Math.round(a.confidence * 100)} %)</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </Card>
    </div>
  );
}

function IntakeChat({
  projectId,
  brief,
  save,
}: {
  projectId: string;
  brief: UseCaseBrief;
  save: SaveBrief;
}) {
  const { t } = useTranslation();
  const intake = useDraftIntake(projectId);
  const [message, setMessage] = useState("");
  const [history, setHistory] = useState<IntakeTurn[]>([]);
  const [pending, setPending] = useState<BriefPatch | null>(null);

  const send = () => {
    const text = message.trim();
    if (!text) return;
    intake.mutate(
      { message: text, history },
      {
        onSuccess: (reply) => {
          const next = reply.patch.next_question;
          setHistory((h) => [
            ...h,
            { role: "user", content: text },
            ...(next ? [{ role: "assistant" as const, content: next }] : []),
          ]);
          setPending(reply.patch);
          setMessage("");
        },
      },
    );
  };
  const accept = () => {
    if (!pending) return;
    const next: UseCaseBrief = { ...brief };
    for (const c of pending.changes ?? []) (next as Record<string, unknown>)[c.field] = c.value;
    next.assumptions = [...(brief.assumptions ?? []), ...(pending.assumptions ?? [])].slice(-20);
    save(next, "copilot");
    setPending(null);
  };
  const question = [...history].reverse().find((h) => h.role === "assistant")?.content;

  return (
    <Card className="border-copilot bg-copilot-bg/30">
      <h3 className="mb-1 flex items-center gap-2 font-semibold">
        <Sparkles className="h-4 w-4" aria-hidden="true" />
        {t("wizard.intake.title")}
      </h3>
      <p className="mb-3 text-xs text-muted">{t("wizard.intake.hint")}</p>
      {question && (
        <p className="mb-2 text-sm" aria-label={t("wizard.intake.question")}>
          <Badge tone="brand" className="mr-2">
            IA
          </Badge>
          {question}
        </p>
      )}
      <Textarea
        rows={3}
        value={message}
        aria-label={t("wizard.intake.message")}
        placeholder={t("wizard.intake.placeholder")}
        onChange={(e) => setMessage(e.target.value)}
      />
      <div className="mt-2">
        <Button variant="ai" loading={intake.isPending} disabled={!message.trim()} onClick={send}>
          {t("wizard.intake.send")}
        </Button>
      </div>
      <ErrorNote error={intake.error} />
      {pending && (pending.changes ?? []).length > 0 && (
        <div className="mt-3 rounded-pt border border-copilot p-3 text-sm">
          <p className="mb-2 font-semibold">{t("wizard.intake.proposed")}</p>
          <ul className="space-y-1">
            {(pending.changes ?? []).map((c) => (
              <li key={c.field}>
                <strong>{t(`wizard.brief.fieldName.${c.field}`)}</strong>:{" "}
                {valueLabel(t, c.field, c.value)}{" "}
                <span className="text-xs text-muted">— {c.rationale}</span>
              </li>
            ))}
          </ul>
          <div className="mt-3 flex gap-2">
            <Button size="sm" variant="ai" onClick={accept}>
              {t("wizard.intake.accept")}
            </Button>
            <Button size="sm" variant="ghost" onClick={() => setPending(null)}>
              {t("wizard.intake.reject")}
            </Button>
          </div>
        </div>
      )}
    </Card>
  );
}

/** Chequeos del plan de un paso (o de todos), con su severidad. */
export function PlanChecks({ plan, step }: { plan: WizardPlan; step?: string }) {
  const { t } = useTranslation();
  const checks = (plan.checks ?? []).filter((c) => !step || c.step === step);
  if (checks.length === 0) return null;
  return (
    <ul className="mb-4 space-y-2" aria-label={t("wizard.plan.checks")}>
      {checks.map((c) => {
        const Icon = SEVERITY_ICON[c.severity];
        return (
          <li key={c.code} className="flex items-start gap-2 text-sm">
            <Icon className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
            <span>
              <Badge tone={SEVERITY_TONE[c.severity]} className="mr-2">
                {t(`severity.${c.severity}`)}
              </Badge>
              {c.message}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

/** Default que sugiere el plan para un campo, con su «por qué» y «Usar». */
export function PlanSuggestion({
  plan,
  field,
  current,
  label,
  onUse,
}: {
  plan: WizardPlan;
  field: "task" | "target_metric" | "architecture_hint";
  current?: string | null;
  label?: (value: string) => string;
  onUse?: (value: string) => void;
}) {
  const { t } = useTranslation();
  const d = (plan.defaults ?? []).find((x) => x.key === field);
  if (!d || d.value === current) return null;
  return (
    <p className="mt-2 flex flex-wrap items-center gap-2 rounded-pt border border-line p-2 text-sm">
      <Badge tone="brand">{t("wizard.plan.suggested")}</Badge>
      {onUse && <strong>{label ? label(d.value) : d.value}</strong>}
      <span className="text-muted">{d.reason}</span>
      {onUse && (
        <Button size="sm" variant="secondary" onClick={() => onUse(d.value)}>
          {t("wizard.plan.use")}
        </Button>
      )}
    </p>
  );
}
