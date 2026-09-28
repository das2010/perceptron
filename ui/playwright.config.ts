import { defineConfig, devices } from "@playwright/test";

/**
 * E2E de la UI contra el Engine real (SPEC §15.1). Levanta `perceptron serve` y Vite; el LLM
 * queda apagado (camino por reglas, determinístico). Objetivo O1: el recorrido guiado de
 * UC-01 termina en menos de 15 minutos.
 */
const workspace = process.env.PERCEPTRON_E2E_WORKSPACE ?? "../.e2e-workspace";

export default defineConfig({
  testDir: "./e2e",
  timeout: 15 * 60_000,
  expect: { timeout: 30_000 },
  retries: 0,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: "http://127.0.0.1:5173",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    locale: "es-AR",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      command: `uv run perceptron serve --port 8765 --workspace "${workspace}"`,
      cwd: "..",
      url: "http://127.0.0.1:8765/api/v1/system/health",
      timeout: 300_000,
      reuseExistingServer: !process.env.CI,
      env: { PERCEPTRON_OFFLINE: "1", PERCEPTRON_LLM__ENABLED: "false" },
    },
    {
      command: "pnpm dev --host 127.0.0.1",
      url: "http://127.0.0.1:5173",
      timeout: 120_000,
      reuseExistingServer: !process.env.CI,
    },
  ],
});
