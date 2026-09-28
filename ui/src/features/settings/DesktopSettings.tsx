/** Configuración del desktop: variante de PyTorch del runtime embebido (RF-TRN-02, ADR-0026). */
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { Button, Card, CardTitle, ErrorNote, Field, Select, Spinner } from "@/components/ui";
import { resetApiClient } from "@/lib/api/client";
import { getPlatform } from "@/lib/platform/bridge";

import { UpdatesCard } from "./Updates";

const VARIANTS = ["cpu", "cuda", "rocm", "xpu"] as const;

export function DesktopSettings() {
  const { t } = useTranslation();
  const runtime = getPlatform().runtime;
  const qc = useQueryClient();
  const state = useQuery({
    queryKey: ["desktop", "runtime"],
    enabled: Boolean(runtime),
    queryFn: () => runtime?.state() ?? Promise.resolve(null),
  });
  const [variant, setVariant] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  if (!runtime) return <UpdatesCard />;
  const current = state.data?.torch_variant ?? "cpu";
  const chosen = variant ?? current;

  const apply = async () => {
    setBusy(true);
    setError(null);
    try {
      await runtime.setTorchVariant(chosen);
      resetApiClient(); // el Engine reinició con otro puerto y token
      await qc.invalidateQueries();
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <Card className="mt-6">
        <CardTitle>{t("settings.desktop.title")}</CardTitle>
        {state.isPending ? (
          <Spinner />
        ) : (
          <dl className="mb-4 grid gap-1 text-sm sm:grid-cols-[12rem_1fr]">
            <dt className="text-muted">{t("settings.desktop.variant")}</dt>
            <dd className="font-semibold">{current}</dd>
            <dt className="text-muted">{t("settings.desktop.index")}</dt>
            <dd className="font-mono text-xs break-all">{state.data?.torch_index ?? "—"}</dd>
            <dt className="text-muted">{t("settings.desktop.driver")}</dt>
            <dd>{state.data?.nvidia_driver ?? "—"}</dd>
          </dl>
        )}
        <div className="flex flex-wrap items-end gap-2">
          <Field label={t("settings.desktop.change")} hint={t("settings.desktop.changeHint")}>
            <Select value={chosen} onChange={(e) => setVariant(e.target.value)} className="w-40">
              {VARIANTS.map((v) => (
                <option key={v} value={v}>
                  {t(`settings.desktop.variants.${v}`)}
                </option>
              ))}
            </Select>
          </Field>
          <Button disabled={chosen === current} loading={busy} onClick={() => void apply()}>
            {t("settings.desktop.apply")}
          </Button>
        </div>
        {busy && <p className="mt-2 text-xs text-muted">{t("settings.desktop.applying")}</p>}
        <ErrorNote error={error} />
      </Card>
      <UpdatesCard />
    </>
  );
}
