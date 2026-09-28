/**
 * Puerta de la UI web del Team Server (RF-SRV-05): si el Engine es un servidor, pide login
 * antes de mostrar la app. En el desktop (Tauri) y contra un Engine local no hace nada.
 */
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, type ReactNode } from "react";
import { useTranslation } from "react-i18next";

import { Spinner } from "@/components/ui";
import { UNAUTHORIZED_EVENT } from "@/lib/api/client";
import { isTauri } from "@/lib/platform/bridge";

import { LoginPage } from "./LoginPage";
import { authKeys, SessionContext, useAuthConfig, useMe } from "./session";

export function AuthGate({ children }: { children: ReactNode }) {
  const { t } = useTranslation();
  const qc = useQueryClient();
  const web = !isTauri();
  const config = useAuthConfig(web);
  const server = web && config.data?.mode === "server";
  const me = useMe(server);

  useEffect(() => {
    if (!server) return;
    const onUnauthorized = () => void qc.invalidateQueries({ queryKey: authKeys.me });
    window.addEventListener(UNAUTHORIZED_EVENT, onUnauthorized);
    return () => window.removeEventListener(UNAUTHORIZED_EVENT, onUnauthorized);
  }, [qc, server]);

  if (!web) return <>{children}</>;
  if (config.isPending || (server && me.isPending)) {
    return (
      <div className="flex h-full items-center justify-center">
        <Spinner label={t("auth.checking")} />
      </div>
    );
  }
  // Sin Team Server (o Engine caído: lo informa la propia app) no hay sesión que pedir.
  if (!server) return <>{children}</>;
  if (!me.data) return <LoginPage config={config.data ?? null} />;
  return <SessionContext.Provider value={me.data}>{children}</SessionContext.Provider>;
}
