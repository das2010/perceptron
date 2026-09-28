import { AlertTriangle, Info, OctagonAlert } from "lucide-react";
import { useMemo } from "react";
import { useTranslation } from "react-i18next";

import { EChart } from "@/components/charts/EChart";
import { Badge, Card, CardTitle, ErrorNote, Spinner, Table, Td, Th } from "@/components/ui";
import { useProfile } from "@/lib/api/hooks";
import { formatNumber, formatPercent } from "@/lib/format";

const SEVERITY_ICON = { info: Info, warning: AlertTriangle, high: OctagonAlert } as const;
const SEVERITY_TONE = { info: "neutral", warning: "warn", high: "bad" } as const;

/** Dataset Profile Card (RF-PRF-07): lo mismo que ve el LLM en L1, en forma legible. */
export function ProfileView({ datasetVersionId }: { datasetVersionId: string }) {
  const { t, i18n } = useTranslation();
  const { data: card, isPending, error } = useProfile(datasetVersionId);

  const classes = useMemo(() => card?.target?.classes ?? [], [card]);
  const option = useMemo(
    () => ({
      xAxis: { type: "category", data: classes.map((c) => c.value) },
      yAxis: { type: "value" },
      series: [{ type: "bar", data: classes.map((c) => c.count) }],
    }),
    [classes],
  );

  if (isPending) return <Spinner label={t("profile.computing")} />;
  if (error) return <ErrorNote error={error} />;
  if (!card) return null;

  return (
    <div className="space-y-4">
      <Card>
        <CardTitle>{t("profile.title")}</CardTitle>
        <dl className="grid gap-3 text-sm sm:grid-cols-4">
          <div>
            <dt className="text-muted">{t("data.modality")}</dt>
            <dd className="font-semibold">{t(`modality.${card.modality}`)}</dd>
          </div>
          <div>
            <dt className="text-muted">{t("data.samples")}</dt>
            <dd className="font-semibold">{card.num_samples.toLocaleString(i18n.language)}</dd>
          </div>
          <div>
            <dt className="text-muted">{t("profile.splits")}</dt>
            <dd>
              {Object.entries(card.split_counts ?? {})
                .map(([k, v]) => `${t(`split.${k}`)} ${v}`)
                .join(" · ")}
            </dd>
          </div>
          <div>
            <dt className="text-muted">{t("data.target")}</dt>
            <dd>
              {card.target ? (
                <>
                  <span className="font-semibold">{card.target.name}</span>{" "}
                  <Badge>{t(`task.${card.target.task_hint}`)}</Badge>
                </>
              ) : (
                "—"
              )}
            </dd>
          </div>
        </dl>
      </Card>

      {(card.alerts ?? []).length > 0 && (
        <Card>
          <CardTitle>{t("profile.alerts")}</CardTitle>
          <ul className="space-y-2">
            {(card.alerts ?? []).map((a, i) => {
              const Icon = SEVERITY_ICON[a.severity];
              return (
                <li key={i} className="flex items-start gap-2 text-sm">
                  <Icon className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
                  <span>
                    <Badge tone={SEVERITY_TONE[a.severity]} className="mr-2">
                      {t(`severity.${a.severity}`)}
                    </Badge>
                    {a.message}
                  </span>
                </li>
              );
            })}
          </ul>
        </Card>
      )}

      {classes.length > 0 && (
        <Card>
          <CardTitle>{t("profile.classes")}</CardTitle>
          <EChart option={option} height={220} label={t("profile.classes")} />
          {card.target?.imbalance_ratio != null && (
            <p className="text-xs text-muted">
              {t("profile.imbalance", {
                ratio: formatNumber(card.target.imbalance_ratio, i18n.language, 2),
              })}
            </p>
          )}
        </Card>
      )}

      {(card.columns ?? []).length > 0 && (
        <Card>
          <CardTitle>{t("profile.columns")}</CardTitle>
          <Table>
            <thead>
              <tr>
                <Th>{t("profile.column")}</Th>
                <Th>{t("profile.type")}</Th>
                <Th>{t("profile.nulls")}</Th>
                <Th>{t("profile.unique")}</Th>
                <Th>{t("profile.summary")}</Th>
              </tr>
            </thead>
            <tbody>
              {(card.columns ?? []).map((c) => (
                <tr key={c.name}>
                  <Td className="font-semibold">{c.name}</Td>
                  <Td>{t(`semantic.${c.semantic}`)}</Td>
                  <Td>{formatPercent(c.null_fraction, i18n.language)}</Td>
                  <Td>{c.n_unique.toLocaleString(i18n.language)}</Td>
                  <Td className="text-xs text-muted">
                    {c.numeric
                      ? t("profile.numericSummary", {
                          mean: formatNumber(c.numeric.mean, i18n.language, 3),
                          std: formatNumber(c.numeric.std, i18n.language, 3),
                        })
                      : c.categorical
                        ? c.categorical.top
                            .map((x) => x.value)
                            .slice(0, 4)
                            .join(", ")
                        : ""}
                  </Td>
                </tr>
              ))}
            </tbody>
          </Table>
        </Card>
      )}
    </div>
  );
}
