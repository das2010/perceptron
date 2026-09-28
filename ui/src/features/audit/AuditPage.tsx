import { useState } from "react";
import { useTranslation } from "react-i18next";

import { Badge, Card, CardTitle, EmptyState, ErrorNote, Spinner, Table, Td, Th } from "@/components/ui";
import { useProjectId } from "@/features/projects/ProjectLayout";
import { useAudit, type LLMCall } from "@/lib/api/hooks";
import { formatDate, formatNumber } from "@/lib/format";

/** Auditoría LLM (RF-PRV-03): exactamente qué salió, a qué proveedor y modelo. */
export function AuditPage() {
  const { t, i18n } = useTranslation();
  const projectId = useProjectId();
  const { data, isPending, error } = useAudit(projectId);
  const [open, setOpen] = useState<LLMCall | null>(null);
  const calls = data ?? [];
  const total = calls.reduce((acc, c) => acc + (c.cost_usd ?? 0), 0);

  return (
    <div className="space-y-4">
      <Card>
        <CardTitle>{t("audit.title")}</CardTitle>
        <p className="mb-3 text-sm text-muted">
          {t("audit.summary", { count: calls.length, cost: formatNumber(total, i18n.language, 4) })}
        </p>
        {isPending && <Spinner />}
        <ErrorNote error={error} />
        {!isPending && calls.length === 0 && <EmptyState>{t("audit.empty")}</EmptyState>}
        {calls.length > 0 && (
          <Table>
            <thead>
              <tr>
                <Th>{t("audit.when")}</Th>
                <Th>{t("audit.purpose")}</Th>
                <Th>{t("audit.model")}</Th>
                <Th>{t("audit.privacy")}</Th>
                <Th>{t("audit.status")}</Th>
                <Th>{t("audit.cost")}</Th>
                <Th />
              </tr>
            </thead>
            <tbody>
              {calls.map((c) => (
                <tr key={c.id}>
                  <Td className="text-xs">{formatDate(c.created_at, i18n.language)}</Td>
                  <Td>{t(`purpose.${c.purpose}`)}</Td>
                  <Td className="text-xs">
                    {c.provider}/{c.model}
                  </Td>
                  <Td>
                    <Badge>{c.privacy_level}</Badge>
                  </Td>
                  <Td>
                    <Badge tone={c.status === "ok" ? "ok" : c.status === "error" ? "bad" : "warn"}>
                      {c.status}
                      {c.cache_hit ? " · cache" : ""}
                    </Badge>
                  </Td>
                  <Td className="text-xs">${formatNumber(c.cost_usd, i18n.language, 4)}</Td>
                  <Td>
                    <button type="button" className="text-xs underline" onClick={() => setOpen(c)}>
                      {t("audit.payload")}
                    </button>
                  </Td>
                </tr>
              ))}
            </tbody>
          </Table>
        )}
      </Card>
      {open && (
        <Card>
          <CardTitle>{t("audit.payloadOf", { id: open.id })}</CardTitle>
          {(open.redactions ?? []).length > 0 && (
            <p className="mb-2 text-xs text-muted">
              {t("audit.redactions")}: {(open.redactions ?? []).join(", ")}
            </p>
          )}
          <pre className="max-h-96 overflow-auto rounded-pt bg-canvas p-3 text-xs">
            {JSON.stringify(open.payload, null, 2)}
          </pre>
        </Card>
      )}
    </div>
  );
}
