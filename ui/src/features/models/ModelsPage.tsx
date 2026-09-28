import { Link, useNavigate } from "@tanstack/react-router";
import { Crown, RotateCcw, Rocket, Swords } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import {
  Badge,
  Button,
  Card,
  CardTitle,
  EmptyState,
  ErrorNote,
  Spinner,
  Table,
  Td,
  Th,
} from "@/components/ui";
import {
  type ChallengeResult,
  useChallenge,
  useCreateDeployment,
  usePromote,
  useRollback,
} from "@/features/monitoring/hooks";
import { useProjectId } from "@/features/projects/ProjectLayout";
import { useModels } from "@/lib/api/hooks";
import { formatDate, formatNumber } from "@/lib/format";

/** Modelos registrados y su ciclo de vida: champion, challengers, rollback (RF-MON-06). */
export function ModelsPage() {
  const { t, i18n } = useTranslation();
  const projectId = useProjectId();
  const navigate = useNavigate();
  const { data, isPending, error } = useModels(projectId);
  const promote = usePromote(projectId);
  const rollback = useRollback(projectId);
  const challenge = useChallenge(projectId);
  const deploy = useCreateDeployment(projectId);
  const [result, setResult] = useState<ChallengeResult | null>(null);
  const models = data ?? [];
  const champion = models.find((m) => m.stage === "production");
  const hasRetired = models.some((m) => m.retired_at);

  return (
    <Card>
      <CardTitle className="flex flex-wrap items-center justify-between gap-2">
        {t("models.title")}
        {hasRetired && (
          <Button
            size="sm"
            variant="ghost"
            loading={rollback.isPending}
            onClick={() => rollback.mutate("")}
          >
            <RotateCcw className="h-4 w-4" aria-hidden="true" />
            {t("models.rollback")}
          </Button>
        )}
      </CardTitle>
      {isPending && <Spinner />}
      <ErrorNote
        error={error ?? promote.error ?? rollback.error ?? challenge.error ?? deploy.error}
      />
      {result && (
        <p role="status" className="mb-3 rounded-pt border border-line p-3 text-sm">
          {t(result.promoted ? "models.challengeWon" : "models.challengeLost", {
            metric: result.metric,
            challenger: formatNumber(result.challenger_value, i18n.language, 4),
            champion:
              result.champion_value === null || result.champion_value === undefined
                ? "—"
                : formatNumber(result.champion_value, i18n.language, 4),
            n: result.n_samples,
          })}
        </p>
      )}
      {!isPending && models.length === 0 && <EmptyState>{t("models.empty")}</EmptyState>}
      {models.length > 0 && (
        <Table>
          <thead>
            <tr>
              <Th>{t("models.version")}</Th>
              <Th>{t("models.stage")}</Th>
              <Th>{t("models.metrics")}</Th>
              <Th>{t("models.run")}</Th>
              <Th>{t("data.created")}</Th>
              <Th />
            </tr>
          </thead>
          <tbody>
            {models.map((m) => {
              const card = (m.model_card ?? {}) as { test_metrics?: Record<string, number> };
              const metrics = card.test_metrics ?? {};
              const isChampion = m.stage === "production";
              return (
                <tr key={m.id}>
                  <Td className="font-mono text-xs">{m.id}</Td>
                  <Td>
                    <Badge tone={isChampion ? "brand" : "neutral"}>
                      {isChampion && <Crown className="h-3 w-3" aria-hidden="true" />}
                      {t(`stage.${m.stage}`)}
                    </Badge>
                  </Td>
                  <Td className="text-xs">
                    {Object.entries(metrics)
                      .slice(0, 3)
                      .map(([k, v]) => `${k} ${formatNumber(v, i18n.language, 3)}`)
                      .join(" · ")}
                  </Td>
                  <Td>
                    <Link
                      to="/projects/$projectId/runs/$runId"
                      params={{ projectId, runId: m.run_id }}
                      className="font-mono text-xs underline"
                    >
                      {m.run_id.split("-").at(-1)}
                    </Link>
                  </Td>
                  <Td className="text-xs text-muted">{formatDate(m.created_at, i18n.language)}</Td>
                  <Td className="space-x-1 whitespace-nowrap text-right">
                    {isChampion ? (
                      <Button
                        size="sm"
                        variant="ghost"
                        loading={deploy.isPending}
                        onClick={() =>
                          deploy.mutate(
                            { mv: m.id, body: { name: "default", sample_rate: 1 } },
                            {
                              onSuccess: () =>
                                void navigate({
                                  to: "/projects/$projectId/monitoring",
                                  params: { projectId },
                                }),
                            },
                          )
                        }
                      >
                        <Rocket className="h-4 w-4" aria-hidden="true" />
                        {t("models.deploy")}
                      </Button>
                    ) : (
                      <>
                        {champion && (
                          <Button
                            size="sm"
                            variant="ghost"
                            loading={challenge.isPending && challenge.variables === m.id}
                            onClick={() =>
                              challenge.mutate(m.id, {
                                onSuccess: (r) => setResult(r as ChallengeResult),
                              })
                            }
                          >
                            <Swords className="h-4 w-4" aria-hidden="true" />
                            {t("models.challenge")}
                          </Button>
                        )}
                        <Button
                          size="sm"
                          variant="ghost"
                          loading={promote.isPending && promote.variables === m.id}
                          onClick={() => promote.mutate(m.id)}
                        >
                          <Crown className="h-4 w-4" aria-hidden="true" />
                          {t("models.promote")}
                        </Button>
                      </>
                    )}
                  </Td>
                </tr>
              );
            })}
          </tbody>
        </Table>
      )}
    </Card>
  );
}
