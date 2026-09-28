/** Políticas por workspace (RF-SRV-06, RF-PRV-02): topes de privacidad y proveedores LLM. */
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
  Input,
  Select,
} from "@/components/ui";
import { useAuthConfig } from "@/features/auth/session";

import { useUpdateWorkspacePolicy, type Workspace } from "./hooks";

const LEVELS = ["L0", "L1", "L2", "L3"] as const;
type Level = (typeof LEVELS)[number];

function PolicyForm({ ws }: { ws: Workspace }) {
  const { t } = useTranslation();
  const update = useUpdateWorkspacePolicy();
  const [max, setMax] = useState<Level>((ws.max_privacy_level as Level | undefined) ?? "L3");
  const [local, setLocal] = useState<Level | "">(
    (ws.local_llm_max_privacy as Level | null | undefined) ?? "",
  );
  const [providers, setProviders] = useState((ws.allowed_llm_providers ?? []).join(", "));
  const restricted = ws.allowed_llm_providers !== null && ws.allowed_llm_providers !== undefined;
  return (
    <form
      className="grid gap-3 border-b border-line pb-4 sm:grid-cols-4"
      onSubmit={(e) => {
        e.preventDefault();
        const list = providers
          .split(",")
          .map((p) => p.trim())
          .filter(Boolean);
        update.mutate({
          workspaceId: ws.id,
          policy: {
            max_privacy_level: max,
            local_llm_max_privacy: local || null,
            clear_allowed_providers: list.length === 0,
            ...(list.length ? { allowed_llm_providers: list } : {}),
          },
        });
      }}
    >
      <p className="font-semibold sm:col-span-4">
        {ws.name}{" "}
        {restricted ? (
          <Badge tone="warn">{t("policies.restricted")}</Badge>
        ) : (
          <Badge>{t("policies.anyProvider")}</Badge>
        )}
      </p>
      <Field label={t("policies.maxPrivacy")} hint={t("policies.maxPrivacyHint")}>
        <Select value={max} onChange={(e) => setMax(e.target.value as Level)}>
          {LEVELS.map((l) => (
            <option key={l} value={l}>
              {l}
            </option>
          ))}
        </Select>
      </Field>
      <Field label={t("policies.localMax")} hint={t("policies.localMaxHint")}>
        <Select value={local} onChange={(e) => setLocal(e.target.value as Level | "")}>
          <option value="">{t("policies.sameAsGeneral")}</option>
          {LEVELS.map((l) => (
            <option key={l} value={l}>
              {l}
            </option>
          ))}
        </Select>
      </Field>
      <Field label={t("policies.providers")} hint={t("policies.providersHint")}>
        <Input value={providers} onChange={(e) => setProviders(e.target.value)} />
      </Field>
      <div className="flex items-end">
        <Button type="submit" loading={update.isPending}>
          {t("policies.save")}
        </Button>
      </div>
      <div className="sm:col-span-4">
        <ErrorNote error={update.error} />
      </div>
    </form>
  );
}

export function PoliciesTab({ workspaces }: { workspaces: Workspace[] }) {
  const { t } = useTranslation();
  const config = useAuthConfig(true);
  const sso = config.data?.oidc_providers ?? [];
  return (
    <div className="space-y-4">
      <Card>
        <CardTitle>{t("policies.title")}</CardTitle>
        <div className="space-y-4">
          {workspaces.map((ws) => (
            <PolicyForm key={ws.id} ws={ws} />
          ))}
        </div>
      </Card>
      <Card>
        <CardTitle>{t("policies.sso")}</CardTitle>
        {sso.length === 0 ? (
          <EmptyState>{t("policies.noSso")}</EmptyState>
        ) : (
          <ul className="space-y-1 text-sm">
            {sso.map((p) => (
              <li key={p.id}>
                <Badge>{p.kind}</Badge> {p.display_name}
              </li>
            ))}
          </ul>
        )}
        <p className="mt-2 text-xs text-muted">{t("policies.ssoHint")}</p>
      </Card>
    </div>
  );
}
