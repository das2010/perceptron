import path from "node:path";
import { fileURLToPath } from "node:url";

import { expect, type Page, test } from "@playwright/test";

import { withDiagnostics } from "../e2e/diagnostics";

const here = path.dirname(fileURLToPath(import.meta.url));
const DEMAND = path.resolve(here, "../../fixtures/uc07_demand/demanda.csv");
const PASSWORD = "clave-del-equipo-e2e";

withDiagnostics();

async function login(page: Page, email: string, password: string) {
  await page.goto("/");
  await page.getByLabel("Email").fill(email);
  await page.getByLabel("Contraseña").fill(password);
  await page.getByRole("button", { name: "Ingresar" }).click();
  await expect(page.getByRole("button", { name: "Cerrar sesión" })).toBeVisible();
}

async function logout(page: Page) {
  await page.getByRole("button", { name: "Cerrar sesión" }).click();
  await expect(page.getByRole("button", { name: "Ingresar" })).toBeVisible();
}

async function createUser(page: Page, email: string, role: "editor" | "viewer") {
  await page.getByLabel("Email").fill(email);
  await page.getByLabel("Nombre").fill(email.split("@")[0] ?? email);
  await page.getByLabel("Contraseña inicial").fill(PASSWORD);
  await page.getByLabel("Rol en el workspace").selectOption(role);
  await page.getByRole("button", { name: "Crear usuario" }).click();
  await expect(page.getByRole("cell", { name: email })).toBeVisible();
}

/**
 * Aceptación de la Capa 5 (SPEC §14): UC-07 completo solo desde el navegador contra el Team
 * Server, y dos usuarios con roles distintos colaborando en el mismo proyecto.
 */
test("UC-07 desde el navegador con roles Editor y Viewer", async ({ page }) => {
  const started = Date.now();
  const editor = `analista-${started}@preteco.test`;
  const viewer = `consulta-${started}@preteco.test`;
  const projectName = `Demanda web ${started}`;

  // El Admin da de alta al equipo desde la consola.
  await login(page, "admin@preteco.test", "clave-admin-de-e2e");
  await page.getByRole("link", { name: "Administración" }).click();
  await createUser(page, editor, "editor");
  await createUser(page, viewer, "viewer");
  await logout(page);

  // La Editora completa UC-07: datos → perfil → entrenar → evaluar → registrar.
  await login(page, editor, PASSWORD);
  await expect(page.getByText("Engine operativo")).toBeVisible();
  await page.getByRole("button", { name: "Nuevo proyecto" }).click();
  await page.getByLabel("Nombre").fill(projectName);
  await page.getByLabel("Objetivo").fill("Pronosticar la demanda semanal por SKU");
  await page.getByRole("button", { name: "Crear" }).click();

  await expect(page.getByText("Agregar datos")).toBeVisible();
  await page.getByTestId("file-input").setInputFiles(DEMAND);
  await expect(page.getByText("Vista previa de demanda.csv")).toBeVisible();
  await page.getByLabel("Columna objetivo").selectOption("unidades");
  await page.getByRole("button", { name: "Crear versión de datos" }).click();
  await expect(page.getByText("Perfil del dataset")).toBeVisible({ timeout: 120_000 });

  await page.getByRole("link", { name: "Seguir: entrenar" }).click();
  await page.getByRole("button", { name: "Proponer preparación" }).click();
  await page.getByRole("button", { name: "Proponer arquitecturas" }).click();
  await expect(page.getByText("Por reglas").first()).toBeVisible({ timeout: 120_000 });
  await page.getByLabel("Intentos (trials)").fill("2");
  await page.getByLabel("Épocas máximas por intento").fill("10");
  await page.getByRole("button", { name: "Recomendar estrategia" }).click();
  await page.getByRole("button", { name: "Entrenar ahora" }).click();

  await expect(page.getByText("Entrenamiento en vivo")).toBeVisible();
  const live = page.locator("div", { has: page.getByText("Entrenamiento en vivo") }).last();
  await expect(live.getByText("Terminado")).toBeVisible({ timeout: 15 * 60_000 });

  await page
    .getByRole("link", { name: /^t\d{3}$/ })
    .first()
    .click();
  const evaluate = page.getByRole("button", { name: "Evaluar en test" });
  await expect(evaluate).toBeVisible({ timeout: 60_000 });
  await evaluate.click();
  await expect(page.getByText("smape", { exact: true })).toBeVisible({ timeout: 120_000 });
  await page.getByRole("button", { name: "Registrar modelo" }).click();
  await expect(page.getByRole("button", { name: "Modelo registrado" })).toBeVisible();
  await logout(page);

  // El Viewer ve el mismo proyecto y su modelo, en solo lectura.
  await login(page, viewer, PASSWORD);
  await page.getByRole("link", { name: projectName }).first().click();
  await expect(page.getByText("Solo lectura")).toBeVisible();
  await page.getByRole("link", { name: "Modelos", exact: true }).click();
  await expect(page.getByRole("cell", { name: /^mdl_/ }).first()).toBeVisible();
  // Las escrituras las rechaza el servidor (RBAC), no solo la UI.
  const denied = await page.evaluate(async (name) => {
    const csrf = /pt_csrf=([^;]+)/.exec(document.cookie)?.[1] ?? "";
    const res = await fetch("/api/v1/projects", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-CSRF-Token": csrf },
      body: JSON.stringify({ name }),
    });
    return res.status;
  }, `${projectName} (viewer)`);
  expect(denied).toBe(403);

  expect(Date.now() - started).toBeLessThan(20 * 60_000);
});
