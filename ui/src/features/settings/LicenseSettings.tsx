/** Licencia (ADR-0035) y telemetría opt-in (D7) en Configuración. */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { KeyRound, Radio } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import {
  Badge,
  Button,
  Card,
  CardTitle,
  ErrorNote,
  Field,
  Spinner,
  Textarea,
} from "@/components/ui";
import { getApiClient } from "@/lib/api/client";
import { type Schemas, unwrap } from "@/lib/api/hooks";
import { formatDate } from "@/lib/format";

type LicenseView = Schemas["LicenseView"];
type TelemetryStatus = Schemas["TelemetryStatus"];
const TONE = {
  valid: "ok",
  missing: "neutral",
  expired: "bad",
  invalid: "bad",
  untrusted: "bad",
  not_yet_valid: "warn",
} as const;

export function LicenseCard() {
  const { t, i18n } = useTranslation();
  const qc = useQueryClient();
  const [content, setContent] = useState("");
  const license = useQuery({
    queryKey: ["system", "license"],
    queryFn: async () =>
      unwrap(await (await getApiClient()).GET("/api/v1/system/license")) as LicenseView,
  });
  const install = useMutation({
    mutationFn: async (text: string) =>
      unwrap(
        await (await getApiClient()).PUT("/api/v1/system/license", { body: { content: text } }),
      ),
    onSuccess: (view) => {
      qc.setQueryData(["system", "license"], view);
      setContent("");
    },
  });
  const v = license.data;
  return (
    <Card>
      <CardTitle className="flex items-center gap-2">
        <KeyRound className="h-4 w-4" aria-hidden="true" />
        {t("license.title")}
      </CardTitle>
      {license.isPending && <Spinner />}
      <ErrorNote error={license.error ?? install.error} />
      {v && (
        <dl className="mb-4 grid grid-cols-2 gap-x-4 gap-y-1 text-sm sm:grid-cols-4">
          <dt className="text-muted">{t("license.status")}</dt>
          <dd>
            <Badge tone={TONE[v.status as keyof typeof TONE] ?? "neutral"}>
              {t(`license.st.${v.status}`)}
            </Badge>
          </dd>
          {v.licensee && (
            <>
              <dt className="text-muted">{t("license.licensee")}</dt>
              <dd>{v.licensee}</dd>
              <dt className="text-muted">{t("license.edition")}</dt>
              <dd>{v.edition}</dd>
              <dt className="text-muted">{t("license.limits")}</dt>
              <dd>
                {t("license.limitsValue", {
                  seats: v.seats ?? "∞",
                  servers: v.servers ?? "∞",
                  gpus: v.gpus ?? "∞",
                })}
              </dd>
              <dt className="text-muted">{t("license.expires")}</dt>
              <dd>
                {v.expires_at ? formatDate(v.expires_at, i18n.language) : t("license.perpetual")}
              </dd>
            </>
          )}
        </dl>
      )}
      {v && !v.enforce && <p className="mb-3 text-xs text-muted">{t("license.noEnforce")}</p>}
      <form
        className="space-y-2"
        onSubmit={(e) => {
          e.preventDefault();
          install.mutate(content);
        }}
      >
        <Field label={t("license.paste")} hint={t("license.pasteHint")}>
          <Textarea
            rows={4}
            className="font-mono text-xs"
            value={content}
            onChange={(e) => setContent(e.target.value)}
          />
        </Field>
        <Button type="submit" disabled={!content.trim()} loading={install.isPending}>
          {t("license.install")}
        </Button>
      </form>
    </Card>
  );
}

export function TelemetryCard() {
  const { t } = useTranslation();
  const qc = useQueryClient();
  const status = useQuery({
    queryKey: ["system", "telemetry"],
    queryFn: async () =>
      unwrap(await (await getApiClient()).GET("/api/v1/system/telemetry")) as TelemetryStatus,
  });
  const consent = useMutation({
    mutationFn: async (optIn: boolean) =>
      unwrap(
        await (await getApiClient()).PUT("/api/v1/system/telemetry", { body: { opt_in: optIn } }),
      ),
    onSuccess: (s) => qc.setQueryData(["system", "telemetry"], s),
  });
  const s = status.data;
  return (
    <Card>
      <CardTitle className="flex items-center gap-2">
        <Radio className="h-4 w-4" aria-hidden="true" />
        {t("telemetry.title")}
      </CardTitle>
      <p className="mb-3 text-sm text-muted">{t("telemetry.hint")}</p>
      <ErrorNote error={status.error ?? consent.error} />
      {s && (
        <>
          <p className="mb-3 flex items-center gap-2 text-sm">
            <Badge tone={s.opt_in ? "ok" : "neutral"}>
              {s.opt_in ? t("telemetry.on") : t("telemetry.off")}
            </Badge>
            {!s.endpoint_configured && (
              <span className="text-xs text-muted">{t("telemetry.noEndpoint")}</span>
            )}
          </p>
          <details className="mb-3 text-sm">
            <summary className="cursor-pointer">{t("telemetry.preview")}</summary>
            <pre className="mt-2 max-h-64 overflow-auto rounded-pt border border-line p-2 text-xs">
              {JSON.stringify(s.preview, null, 2)}
            </pre>
          </details>
          <Button
            variant={s.opt_in ? "ghost" : "secondary"}
            loading={consent.isPending}
            onClick={() => consent.mutate(!s.opt_in)}
          >
            {s.opt_in ? t("telemetry.disable") : t("telemetry.enable")}
          </Button>
        </>
      )}
    </Card>
  );
}
