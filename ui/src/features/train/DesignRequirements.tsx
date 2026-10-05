/**
 * Requisitos de diseño del escenario y evaluación de cada propuesta (ADR-0041): qué exige el
 * caso (p. ej. «red preentrenada» con pocas imágenes), cuál propuesta lo cumple mejor y un
 * aviso si se elige otra. La recomendación nunca bloquea.
 */
import { AlertTriangle, Check, X } from "lucide-react";
import { useTranslation } from "react-i18next";

import { Badge } from "@/components/ui";
import type { ArchProposals, Schemas } from "@/lib/api/hooks";

type Requirements = Schemas["DesignRequirements"];
type Assessment = Schemas["DesignAssessment"];
type Proposal = ArchProposals["proposals"][number];

export function RequirementsPanel({
  requirements,
}: {
  requirements?: Requirements | null | undefined;
}) {
  const { t } = useTranslation();
  const items = requirements?.items ?? [];
  if (items.length === 0) return null;
  return (
    <section
      aria-label={t("design.title")}
      className="rounded-pt border border-line bg-canvas p-4 text-sm"
    >
      <h3 className="font-semibold">{t("design.title")}</h3>
      <p className="mt-1 text-xs text-muted">{t("design.intro")}</p>
      <ul className="mt-3 space-y-2">
        {items.map((r) => (
          <li key={r.code} className="flex flex-wrap items-start gap-2">
            <Badge tone={r.level === "must" ? "warn" : "neutral"}>
              {t(`design.level.${r.level}`)}
            </Badge>
            <span>
              <span className="font-semibold">{t(`design.code.${r.code}`)}</span>
              <span className="text-muted"> — {r.message}</span>
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}

export function AssessmentChecks({ assessment }: { assessment?: Assessment | null | undefined }) {
  const { t } = useTranslation();
  if (!assessment) return null;
  return (
    <div className="mb-2 space-y-1">
      {assessment.recommended && <Badge tone="brand">{t("design.recommended")}</Badge>}
      <ul className="flex flex-wrap gap-x-3 gap-y-1 text-xs" aria-label={t("design.checks")}>
        {(assessment.checks ?? []).map((c) => (
          <li key={c.code} className={`flex items-center gap-1 ${c.met ? "text-ok" : "text-bad"}`}>
            {c.met ? (
              <Check className="h-3 w-3" aria-hidden="true" />
            ) : (
              <X className="h-3 w-3" aria-hidden="true" />
            )}
            <span>{t(`design.code.${c.code}`)}</span>
            <span className="sr-only">{t(c.met ? "design.met" : "design.unmet")}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Aviso al elegir una propuesta que no es la recomendada: qué requisitos deja sin cumplir. */
export function NotRecommendedNote({
  proposals,
  chosenId,
}: {
  proposals?: Proposal[] | undefined;
  chosenId?: string | null | undefined;
}) {
  const { t } = useTranslation();
  const chosen = proposals?.find((p) => p.archspec.id === chosenId);
  const best = proposals?.find((p) => p.assessment?.recommended);
  if (!chosen?.assessment || !best || chosen === best || chosen.assessment.recommended) return null;
  const unmet = (chosen.assessment.checks ?? [])
    .filter((c) => !c.met)
    .map((c) => t(`design.code.${c.code}`));
  return (
    <p role="status" className="flex items-start gap-2 rounded-pt border border-warn p-3 text-sm">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warn" aria-hidden="true" />
      <span>
        {t("design.notRecommended", { best: best.title })}
        {unmet.length > 0 && ` ${t("design.unmetList", { list: unmet.join(", ") })}`}
      </span>
    </p>
  );
}
