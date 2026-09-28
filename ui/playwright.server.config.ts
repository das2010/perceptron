import { defineConfig, devices } from "@playwright/test";

/**
 * E2E de la UI web del Team Server (Capa 5, RF-SRV-05): `perceptron-server serve` sirve la UI
 * compilada (`pnpm -C ui build`) y la API en el mismo origen, con login y RBAC. En CI la base
 * es PostgreSQL (`PERCEPTRON_DATABASE_URL`); si no, SQLite en el workspace.
 */
const workspace = process.env.PERCEPTRON_E2E_SERVER_WORKSPACE ?? "../.e2e-server-workspace";
const port = 8766;

export default defineConfig({
  testDir: "./e2e-server",
  timeout: 20 * 60_000,
  expect: { timeout: 30_000 },
  retries: 0,
  workers: 1,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    locale: "es-AR",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: `uv run perceptron-server serve --host 127.0.0.1 --port ${port}`,
    cwd: "..",
    url: `http://127.0.0.1:${port}/api/v1/system/health`,
    timeout: 300_000,
    reuseExistingServer: !process.env.CI,
    gracefulShutdown: { signal: "SIGTERM", timeout: 5_000 },
    env: {
      PERCEPTRON_WORKSPACE_DIR: workspace,
      PERCEPTRON_OFFLINE: "1",
      PERCEPTRON_LLM__ENABLED: "false",
      PERCEPTRON_LOGGING__TO_FILE: "false",
      PERCEPTRON_SERVER__SECRET_KEY: "e2e-clave-de-firma-solo-para-pruebas-000",
      PERCEPTRON_SERVER__COOKIE_SECURE: "false",
      PERCEPTRON_SERVER__SPA_DIR: "ui/dist",
      PERCEPTRON_SERVER__BOOTSTRAP_ADMIN_EMAIL: "admin@preteco.test",
      PERCEPTRON_SERVER__BOOTSTRAP_ADMIN_PASSWORD: "clave-admin-de-e2e",
      ...(process.env.PERCEPTRON_DATABASE_URL
        ? { PERCEPTRON_DATABASE_URL: process.env.PERCEPTRON_DATABASE_URL }
        : {}),
    },
  },
});
