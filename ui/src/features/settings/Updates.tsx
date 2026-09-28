/** Actualización automática firmada del desktop (Tauri updater, ADR-0037). */
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { Download, X } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { Button, Card, CardTitle, ErrorNote, Spinner } from "@/components/ui";
import { getPlatform, type UpdateInfo, type UpdateProgress } from "@/lib/platform/bridge";

const UPDATE_KEY = ["desktop", "update"] as const;

/** Una consulta por sesión: el aviso del encabezado y la tarjeta comparten el resultado. */
function useUpdateCheck() {
  const updater = getPlatform().updater;
  return useQuery<UpdateInfo | null>({
    queryKey: UPDATE_KEY,
    enabled: Boolean(updater),
    staleTime: Infinity,
    retry: false,
    queryFn: () => updater?.check() ?? Promise.resolve(null),
  });
}

const mb = (bytes: number) => (bytes / 1_048_576).toFixed(1);

export function UpdatesCard() {
  const { t, i18n } = useTranslation();
  const updater = getPlatform().updater;
  const qc = useQueryClient();
  const check = useUpdateCheck();
  const [progress, setProgress] = useState<UpdateProgress | null>(null);
  const [installing, setInstalling] = useState(false);
  const [error, setError] = useState<unknown>(null);
  if (!updater) return null;
  const update = check.data;

  const recheck = () => {
    setError(null);
    void qc.invalidateQueries({ queryKey: UPDATE_KEY });
  };

  const install = async () => {
    setInstalling(true);
    setError(null);
    const stop = await updater.onProgress(setProgress);
    try {
      await updater.install(); // si sale bien, la app se reinicia
    } catch (e) {
      setError(e);
      setInstalling(false);
      // el backend descarta la actualización pendiente al intentar instalarla
      void qc.invalidateQueries({ queryKey: UPDATE_KEY });
    } finally {
      stop();
    }
  };

  return (
    <Card className="mt-6">
      <CardTitle>{t("settings.updates.title")}</CardTitle>
      {check.isFetching ? (
        <Spinner />
      ) : update ? (
        <div className="space-y-2 text-sm">
          <p className="font-semibold">
            {t("settings.updates.available", { version: update.version })}
          </p>
          <p className="text-muted">
            {t("settings.updates.current")}: {update.current_version}
            {update.date &&
              ` · ${t("settings.updates.published", {
                date: new Date(update.date).toLocaleDateString(i18n.resolvedLanguage),
              })}`}
          </p>
          {update.notes && (
            <pre className="max-h-48 overflow-y-auto whitespace-pre-wrap rounded-pt bg-canvas p-3 text-xs">
              {update.notes}
            </pre>
          )}
          <p className="text-xs text-muted">{t("settings.updates.safe")}</p>
          <Button loading={installing} onClick={() => void install()}>
            <Download className="h-4 w-4" aria-hidden="true" />
            {t("settings.updates.install")}
          </Button>
          {installing && (
            <p className="text-xs text-muted" role="status">
              {t("settings.updates.installing")}
              {progress?.total
                ? ` ${t("settings.updates.downloaded", {
                    done: mb(progress.downloaded),
                    total: mb(progress.total),
                  })}`
                : ""}
            </p>
          )}
        </div>
      ) : (
        <div className="flex flex-wrap items-center gap-3 text-sm">
          {check.isSuccess && <p>{t("settings.updates.upToDate")}</p>}
          <Button variant="secondary" onClick={recheck}>
            {t("settings.updates.check")}
          </Button>
        </div>
      )}
      <ErrorNote error={error ?? check.error} />
    </Card>
  );
}

/** Aviso en el encabezado cuando hay una versión nueva (se consulta al abrir la app). */
export function UpdateNotice() {
  const { t } = useTranslation();
  const check = useUpdateCheck();
  const [dismissed, setDismissed] = useState<string | null>(null);
  const version = check.data?.version;
  if (!version || dismissed === version) return null;
  return (
    <div
      role="status"
      className="flex items-center justify-between gap-3 border-b border-line bg-pt-lime/20 px-4 py-1.5 text-sm"
    >
      <span>{t("settings.updates.notice", { version })}</span>
      <span className="flex items-center gap-2">
        <Link to="/settings" className="font-semibold underline">
          {t("settings.updates.see")}
        </Link>
        <Button
          variant="ghost"
          size="icon"
          aria-label={t("settings.updates.dismiss")}
          onClick={() => setDismissed(version)}
        >
          <X className="h-4 w-4" />
        </Button>
      </span>
    </div>
  );
}
