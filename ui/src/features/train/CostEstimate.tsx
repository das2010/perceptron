/** Complejidad y costo de la arquitectura elegida por dispositivo (RF-PRF-08). */
import { useMutation } from "@tanstack/react-query";
import { Gauge } from "lucide-react";
import { useTranslation } from "react-i18next";

import { Badge, Button, ErrorNote, Table, Td, Th } from "@/components/ui";
import { getApiClient } from "@/lib/api/client";
import { type Schemas, unwrap } from "@/lib/api/hooks";
import { formatNumber } from "@/lib/format";

type Estimate = Schemas["CostEstimate"];

export function CostEstimate({
  projectId,
  archspecId,
  datasetVersionId,
}: {
  projectId: string;
  archspecId: string;
  datasetVersionId: string;
}) {
  const { t, i18n } = useTranslation();
  const estimate = useMutation({
    mutationFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/projects/{project_id}/arch/estimate", {
          params: { path: { project_id: projectId } },
          body: { archspec_id: archspecId, dataset_version_id: datasetVersionId },
        }),
      ) as Estimate,
  });
  const e = estimate.data;
  const n = (v: number | null | undefined, d = 1) => formatNumber(v ?? null, i18n.language, d);

  return (
    <div className="mt-4">
      <Button variant="secondary" loading={estimate.isPending} onClick={() => estimate.mutate()}>
        <Gauge className="h-4 w-4" aria-hidden="true" />
        {t("cost.estimate")}
      </Button>
      <ErrorNote error={estimate.error} />
      {e && (
        <div className="mt-3 space-y-2 text-sm" aria-label={t("cost.title")}>
          <p>
            {t("cost.size", {
              train: n(e.n_train, 0),
              total: n(e.num_samples, 0),
              disk: n(e.size_bytes / 2 ** 20),
            })}
            {e.train_tensor_mb != null && ` ${t("cost.tensors", { mb: n(e.train_tensor_mb) })}`}
          </p>
          <p>
            {t("cost.model", {
              params: n(e.num_params, 0),
              memory: n(e.memory_mb),
              batch: e.batch_size,
            })}
          </p>
          <Table>
            <thead>
              <tr>
                <Th>{t("cost.device")}</Th>
                <Th>{t("cost.memory")}</Th>
                <Th>{t("cost.epoch")}</Th>
              </tr>
            </thead>
            <tbody>
              {e.devices.map((d, i) => (
                <tr key={`${d.device}-${i}`}>
                  <Td>
                    <span className="font-semibold uppercase">{d.device}</span>{" "}
                    <span className="text-xs text-muted">{d.name}</span>
                  </Td>
                  <Td>
                    {d.memory_gb != null ? `${n(d.memory_gb)} GB` : "—"}{" "}
                    {d.fits != null && (
                      <Badge tone={d.fits ? "ok" : "bad"}>
                        {t(d.fits ? "cost.fits" : "cost.doesNotFit")}
                      </Badge>
                    )}
                  </Td>
                  <Td>{d.epoch_time_s != null ? `${n(d.epoch_time_s)} s` : "—"}</Td>
                </tr>
              ))}
            </tbody>
          </Table>
        </div>
      )}
    </div>
  );
}
