/** Caché de modelos preentrenados (RF-TRN-11): qué hay descargado, cuánto ocupa,
 * verificación de checksums y predescarga para trabajar sin conexión. */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Download, ShieldCheck, Trash2 } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import {
  Badge,
  Button,
  Card,
  CardTitle,
  EmptyState,
  ErrorNote,
  Field,
  Select,
  Spinner,
  Table,
  Td,
  Th,
} from "@/components/ui";
import { getApiClient } from "@/lib/api/client";
import { type Schemas, unwrap } from "@/lib/api/hooks";
import { formatDate } from "@/lib/format";

type CacheReport = Schemas["CacheReport"];
type VerifyReport = Schemas["VerifyReport"];
const KEY = ["system", "models-cache"] as const;

const size = (bytes: number) =>
  bytes >= 1024 ** 3
    ? `${(bytes / 1024 ** 3).toFixed(2)} GB`
    : `${(bytes / 1024 ** 2).toFixed(1)} MB`;

export function ModelsCacheCard() {
  const { t, i18n } = useTranslation();
  const qc = useQueryClient();
  const [choice, setChoice] = useState("");
  const report = useQuery({
    queryKey: KEY,
    queryFn: async () =>
      unwrap(await (await getApiClient()).GET("/api/v1/system/models-cache")) as CacheReport,
  });
  const remove = useMutation({
    mutationFn: async (id: string) =>
      unwrap(
        await (
          await getApiClient()
        ).DELETE("/api/v1/system/models-cache/{model_id}", {
          params: { path: { model_id: id } },
        }),
      ) as CacheReport,
    onSuccess: (data) => qc.setQueryData(KEY, data),
  });
  const check = useMutation({
    mutationFn: async () =>
      unwrap(
        await (await getApiClient()).POST("/api/v1/system/models-cache/verify"),
      ) as VerifyReport,
  });
  const fetchModel = useMutation({
    mutationFn: async (model: string) =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/system/models-cache/prefetch", { body: { model } }),
      ),
  });
  const data = report.data;
  const available = data?.available ?? [];

  return (
    <Card className="mt-6">
      <CardTitle className="flex items-center gap-2">
        {t("modelsCache.title")}
        {data?.offline && <Badge tone="warn">{t("modelsCache.offline")}</Badge>}
      </CardTitle>
      {report.isPending && <Spinner />}
      {data && (
        <>
          <p className="mb-3 text-sm text-muted">
            {t("modelsCache.summary", { size: size(data.total_bytes), count: data.models.length })}
            <span className="ml-1 font-mono text-xs break-all">{data.path}</span>
          </p>
          {data.models.length === 0 ? (
            <EmptyState>{t("modelsCache.empty")}</EmptyState>
          ) : (
            <Table>
              <thead>
                <tr>
                  <Th>{t("modelsCache.model")}</Th>
                  <Th>{t("modelsCache.source")}</Th>
                  <Th>{t("modelsCache.size")}</Th>
                  <Th>{t("modelsCache.lastUsed")}</Th>
                  <Th />
                </tr>
              </thead>
              <tbody>
                {data.models.map((m) => (
                  <tr key={`${m.source}:${m.id}`}>
                    <Td className="font-mono text-xs">{m.id}</Td>
                    <Td>{m.source}</Td>
                    <Td>{size(m.size_bytes)}</Td>
                    <Td>{m.last_used ? formatDate(m.last_used, i18n.language) : "—"}</Td>
                    <Td>
                      <Button
                        variant="ghost"
                        size="icon"
                        aria-label={t("modelsCache.delete", { model: m.id })}
                        loading={remove.isPending && remove.variables === m.id}
                        onClick={() => remove.mutate(m.id)}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </Td>
                  </tr>
                ))}
              </tbody>
            </Table>
          )}
          <div className="mt-4 flex flex-wrap items-end gap-2">
            <Button variant="secondary" loading={check.isPending} onClick={() => check.mutate()}>
              <ShieldCheck className="h-4 w-4" aria-hidden="true" />
              {t("modelsCache.verify")}
            </Button>
            {available.length > 0 && !data.offline && (
              <>
                <Field label={t("modelsCache.prefetch")} hint={t("modelsCache.prefetchHint")}>
                  <Select
                    value={choice}
                    onChange={(e) => setChoice(e.target.value)}
                    className="w-72"
                  >
                    <option value="">—</option>
                    {available.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </Select>
                </Field>
                <Button
                  disabled={!choice}
                  loading={fetchModel.isPending}
                  onClick={() => fetchModel.mutate(choice)}
                >
                  <Download className="h-4 w-4" aria-hidden="true" />
                  {t("modelsCache.download")}
                </Button>
              </>
            )}
          </div>
          {check.data && (
            <p className="mt-2 text-sm" role="status">
              {check.data.corrupt.length === 0
                ? t("modelsCache.verified", { count: check.data.checked })
                : t("modelsCache.corrupt", { files: check.data.corrupt.join(", ") })}
            </p>
          )}
          {fetchModel.isSuccess && (
            <p className="mt-2 text-sm text-muted" role="status">
              {t("modelsCache.downloading")}
            </p>
          )}
        </>
      )}
      <ErrorNote error={report.error ?? remove.error ?? check.error ?? fetchModel.error} />
    </Card>
  );
}
