/** Login del Team Server (RF-SRV-01): local y SSO OIDC (Entra ID, Google, genérico). */
import { KeyRound, LogIn } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { Button, Card, Field, Input } from "@/components/ui";
import { ApiError } from "@/lib/api/hooks";

import { type AuthConfig, useLogin } from "./session";

function ssoHref(provider: string): string {
  const next = `${window.location.pathname}${window.location.search}`.replace(
    /[?&]sso_error=1/,
    "",
  );
  return `/api/v1/auth/oidc/${encodeURIComponent(provider)}/login?next=${encodeURIComponent(next || "/")}`;
}

export function LoginPage({ config }: { config?: AuthConfig | null }) {
  const { t } = useTranslation();
  const login = useLogin();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const providers = config?.oidc_providers ?? [];
  const passwordLogin = config?.password_login ?? true;
  const ssoFailed = new URLSearchParams(window.location.search).has("sso_error");
  const error = login.error;
  const message =
    error instanceof ApiError && error.status === 429
      ? t("auth.locked")
      : error
        ? t("auth.invalid")
        : ssoFailed
          ? t("auth.ssoFailed")
          : null;

  return (
    <div className="flex h-full items-center justify-center bg-canvas p-6">
      <Card className="w-full max-w-sm">
        <div className="mb-6 flex flex-col items-center gap-2">
          <span className="rounded-full bg-pt-dark px-5 py-1 text-xl font-semibold text-pt-lime">
            {t("app.name")}
          </span>
          <p className="text-sm text-muted">{t("auth.subtitle")}</p>
        </div>
        {providers.length > 0 && (
          <div className="mb-4 space-y-2">
            {providers.map((p) => (
              <a
                key={p.id}
                href={ssoHref(p.id)}
                className="flex h-10 w-full items-center justify-center gap-2 rounded-pt border border-line bg-card text-sm font-semibold hover:bg-canvas"
              >
                <KeyRound className="h-4 w-4" aria-hidden="true" />
                {t("auth.ssoWith", { provider: p.display_name })}
              </a>
            ))}
          </div>
        )}
        {providers.length > 0 && passwordLogin && (
          <p className="mb-4 text-center text-xs text-muted">{t("auth.or")}</p>
        )}
        {message && !passwordLogin && (
          <p role="alert" className="text-sm text-bad">
            {message}
          </p>
        )}
        {passwordLogin && (
          <form
            className="space-y-4"
            onSubmit={(e) => {
              e.preventDefault();
              login.mutate({ email, password });
            }}
          >
            <Field label={t("auth.email")}>
              <Input
                type="email"
                autoComplete="username"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                autoFocus
              />
            </Field>
            <Field label={t("auth.password")}>
              <Input
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </Field>
            {message && (
              <p role="alert" className="text-sm text-bad">
                {message}
              </p>
            )}
            <Button type="submit" className="w-full" loading={login.isPending}>
              <LogIn className="h-4 w-4" aria-hidden="true" />
              {t("auth.login")}
            </Button>
          </form>
        )}
      </Card>
    </div>
  );
}
