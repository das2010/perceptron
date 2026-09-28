import { Link, Outlet } from "@tanstack/react-router";
import {
  FolderKanban,
  Home,
  LogOut,
  PanelRightClose,
  PanelRightOpen,
  Settings,
  ShieldCheck,
} from "lucide-react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui";
import { canAdminister, useLogout, useSession } from "@/features/auth/session";
import { CopilotPanel } from "@/features/copilot/CopilotPanel";
import { useProjects } from "@/lib/api/hooks";
import { SUPPORTED_LANGUAGES } from "@/lib/i18n";

import { useUiStore } from "./store";
import { useTheme, type ThemePreference } from "./theme";

const THEMES: ThemePreference[] = ["system", "light", "dark"];
const THEME_LABEL = { system: "themeSystem", light: "themeLight", dark: "themeDark" } as const;

const navLink =
  "flex items-center gap-2 rounded-pt px-3 py-2 text-sm hover:bg-canvas data-[status=active]:bg-canvas data-[status=active]:font-semibold";

function ProjectNav() {
  const { t } = useTranslation();
  const { data } = useProjects();
  const projects = Array.isArray(data) ? data : [];
  return (
    <div className="mt-6">
      <p className="px-3 text-xs font-semibold uppercase text-muted">{t("nav.projects")}</p>
      <ul className="mt-2 space-y-1">
        {projects.slice(0, 12).map((p) => (
          <li key={p.id}>
            <Link to="/projects/$projectId" params={{ projectId: p.id }} className={navLink}>
              <FolderKanban className="h-4 w-4 shrink-0" aria-hidden="true" />
              <span className="truncate">{p.name}</span>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Usuario del Team Server y logout (en el desktop no hay sesión). */
function UserMenu() {
  const { t } = useTranslation();
  const me = useSession();
  const logout = useLogout();
  if (!me) return null;
  return (
    <div className="flex items-center gap-2">
      <span className="text-muted" title={me.user.email}>
        {me.user.display_name || me.user.email}
      </span>
      <Button
        variant="ghost"
        size="icon"
        aria-label={t("auth.logout")}
        onClick={() => logout.mutate()}
      >
        <LogOut className="h-4 w-4" />
      </Button>
    </div>
  );
}

export function Layout() {
  const { t, i18n } = useTranslation();
  const { theme, setTheme } = useTheme();
  const { copilotOpen, toggleCopilot } = useUiStore();
  const me = useSession();

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center justify-between border-b border-line bg-card px-4 py-2">
        <Link
          to="/"
          className="rounded-full bg-pt-dark px-4 py-1 text-lg font-semibold text-pt-lime"
        >
          {t("app.name")}
        </Link>
        <div className="flex items-center gap-3 text-sm">
          <label>
            <span className="sr-only">{t("settings.language")}</span>
            <select
              className="rounded-pt border border-line bg-canvas px-2 py-1"
              value={i18n.resolvedLanguage}
              onChange={(e) => void i18n.changeLanguage(e.target.value)}
            >
              {SUPPORTED_LANGUAGES.map((lng) => (
                <option key={lng} value={lng}>
                  {lng.toUpperCase()}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span className="sr-only">{t("settings.theme")}</span>
            <select
              className="rounded-pt border border-line bg-canvas px-2 py-1"
              value={theme}
              onChange={(e) => setTheme(e.target.value as ThemePreference)}
            >
              {THEMES.map((th) => (
                <option key={th} value={th}>
                  {t(`settings.${THEME_LABEL[th]}`)}
                </option>
              ))}
            </select>
          </label>
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleCopilot}
            aria-pressed={copilotOpen}
            aria-label={t("copilot.toggle")}
          >
            {copilotOpen ? (
              <PanelRightClose className="h-4 w-4" />
            ) : (
              <PanelRightOpen className="h-4 w-4" />
            )}
          </Button>
          <UserMenu />
        </div>
      </header>
      <div className="flex min-h-0 flex-1">
        <nav
          aria-label={t("nav.main")}
          className="w-60 shrink-0 overflow-y-auto border-r border-line bg-card p-3"
        >
          <ul className="space-y-1">
            <li>
              <Link to="/" className={navLink} activeOptions={{ exact: true }}>
                <Home className="h-4 w-4" aria-hidden="true" />
                {t("nav.home")}
              </Link>
            </li>
            <li>
              <Link to="/settings" className={navLink}>
                <Settings className="h-4 w-4" aria-hidden="true" />
                {t("nav.settings")}
              </Link>
            </li>
            {canAdminister(me) && (
              <li>
                <Link to="/admin" className={navLink}>
                  <ShieldCheck className="h-4 w-4" aria-hidden="true" />
                  {t("nav.admin")}
                </Link>
              </li>
            )}
          </ul>
          <ProjectNav />
        </nav>
        <main className="min-w-0 flex-1 overflow-y-auto p-6">
          <Outlet />
        </main>
        {copilotOpen && <CopilotPanel />}
      </div>
    </div>
  );
}
