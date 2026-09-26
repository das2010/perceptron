import { useTranslation } from "react-i18next";

import { EngineStatus } from "@/features/system/EngineStatus";
import { SUPPORTED_LANGUAGES } from "@/lib/i18n";

import styles from "./App.module.css";
import { useTheme, type ThemePreference } from "./theme";

const THEMES: ThemePreference[] = ["system", "light", "dark"];
const THEME_LABEL = { system: "themeSystem", light: "themeLight", dark: "themeDark" } as const;

export function App() {
  const { t, i18n } = useTranslation();
  const { theme, setTheme } = useTheme();

  return (
    <div className={styles.shell}>
      <header className={styles.header}>
        <span className={styles.logo}>{t("app.name")}</span>
        <div className={styles.controls}>
          <label>
            <span className={styles.srOnly}>{t("settings.language")}</span>
            <select
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
            <span className={styles.srOnly}>{t("settings.theme")}</span>
            <select value={theme} onChange={(e) => setTheme(e.target.value as ThemePreference)}>
              {THEMES.map((th) => (
                <option key={th} value={th}>
                  {t(`settings.${THEME_LABEL[th]}`)}
                </option>
              ))}
            </select>
          </label>
        </div>
      </header>
      <main className={styles.main}>
        <h1 className={styles.tagline}>{t("app.tagline")}</h1>
        <EngineStatus />
      </main>
    </div>
  );
}
