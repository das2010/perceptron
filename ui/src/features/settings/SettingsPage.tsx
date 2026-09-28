import { useState } from "react";
import { useTranslation } from "react-i18next";

import {
  Badge,
  Button,
  Card,
  CardTitle,
  ErrorNote,
  Field,
  Input,
  PageHeader,
  Select,
  Spinner,
  Table,
  Td,
  Th,
} from "@/components/ui";
import {
  useLlmProfiles,
  useProviders,
  useSaveProvider,
  useSetActiveProfile,
  useTestLlm,
  type ProviderView,
} from "@/lib/api/hooks";

/** La clave es write-only: se envía una vez y el Engine la guarda en el keychain (RF-LLM-08). */
function KeyForm({ provider }: { provider: ProviderView }) {
  const { t } = useTranslation();
  const save = useSaveProvider();
  const [key, setKey] = useState("");
  return (
    <form
      className="flex items-end gap-2"
      onSubmit={(e) => {
        e.preventDefault();
        save.mutate(
          {
            name: provider.name,
            body: {
              kind: provider.kind,
              base_url: provider.base_url,
              api_key_ref: provider.api_key_ref,
              api_key: key,
            },
          },
          { onSuccess: () => setKey("") },
        );
      }}
    >
      <Input
        type="password"
        autoComplete="off"
        placeholder={t("settings.llm.keyPlaceholder")}
        aria-label={t("settings.llm.keyFor", { name: provider.name })}
        value={key}
        onChange={(e) => setKey(e.target.value)}
        className="w-56"
      />
      <Button size="sm" type="submit" disabled={!key} loading={save.isPending}>
        {t("common.save")}
      </Button>
      <ErrorNote error={save.error} />
    </form>
  );
}

function LlmSettings() {
  const { t } = useTranslation();
  const providers = useProviders();
  const profiles = useLlmProfiles();
  const setActive = useSetActiveProfile();
  const test = useTestLlm();

  return (
    <div className="space-y-4">
      <Card>
        <CardTitle>{t("settings.llm.profile")}</CardTitle>
        {profiles.isPending && <Spinner />}
        {profiles.data && (
          <div className="flex flex-wrap items-end gap-3">
            <Field label={t("settings.llm.active")} hint={t("settings.llm.activeHint")}>
              <Select
                value={profiles.data.active}
                onChange={(e) => setActive.mutate(e.target.value)}
              >
                {Object.keys(profiles.data.profiles).map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </Select>
            </Field>
            <Button variant="ai" loading={test.isPending} onClick={() => test.mutate(undefined)}>
              {t("settings.llm.test")}
            </Button>
          </div>
        )}
        {test.data && (
          <p className="mt-3 text-sm" role="status">
            {test.data.ok
              ? t("settings.llm.testOk", {
                  provider: test.data.provider,
                  model: test.data.model,
                  seconds: test.data.latency_s,
                })
              : t("settings.llm.testFail", { error: test.data.error })}
          </p>
        )}
        <ErrorNote error={setActive.error ?? test.error} />
      </Card>

      <Card>
        <CardTitle>{t("settings.llm.providers")}</CardTitle>
        {providers.isPending && <Spinner />}
        <ErrorNote error={providers.error} />
        {providers.data && (
          <Table>
            <thead>
              <tr>
                <Th>{t("settings.llm.provider")}</Th>
                <Th>{t("settings.llm.where")}</Th>
                <Th>{t("settings.llm.models")}</Th>
                <Th>{t("settings.llm.key")}</Th>
              </tr>
            </thead>
            <tbody>
              {providers.data.map((p) => (
                <tr key={p.name}>
                  <Td className="font-semibold">{p.name}</Td>
                  <Td>
                    <Badge>{p.local ? t("settings.llm.local") : t("settings.llm.cloud")}</Badge>
                  </Td>
                  <Td className="text-xs">{Object.keys(p.models).join(", ")}</Td>
                  <Td>
                    {p.local ? (
                      <span className="text-xs text-muted">{t("settings.llm.noKey")}</span>
                    ) : (
                      <div className="flex flex-col gap-1">
                        <Badge tone={p.has_key ? "ok" : "warn"}>
                          {p.has_key ? t("settings.llm.hasKey") : t("settings.llm.missingKey")}
                        </Badge>
                        <KeyForm provider={p} />
                      </div>
                    )}
                  </Td>
                </tr>
              ))}
            </tbody>
          </Table>
        )}
      </Card>
    </div>
  );
}

export function SettingsPage() {
  const { t } = useTranslation();
  return (
    <>
      <PageHeader title={t("nav.settings")} description={t("settings.description")} />
      <LlmSettings />
    </>
  );
}
