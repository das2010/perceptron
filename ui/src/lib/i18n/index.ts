import i18n from "i18next";
import { initReactI18next } from "react-i18next";

import en from "./en.json";
import es from "./es.json";

export const SUPPORTED_LANGUAGES = ["es", "en"] as const;
export type Language = (typeof SUPPORTED_LANGUAGES)[number];

export const resources = {
  es: { translation: es },
  en: { translation: en },
} as const;

// Español por defecto (SPEC §11.2). Sin strings hardcodeados en componentes.
void i18n.use(initReactI18next).init({
  resources,
  lng: "es",
  fallbackLng: "es",
  supportedLngs: SUPPORTED_LANGUAGES,
  interpolation: { escapeValue: false },
});

export default i18n;
