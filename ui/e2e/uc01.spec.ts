import path from "node:path";
import { fileURLToPath } from "node:url";

import { expect, test } from "@playwright/test";

const here = path.dirname(fileURLToPath(import.meta.url));
const CHURN = path.resolve(here, "../../fixtures/uc01_churn/churn.csv");

/** UC-01 guiado sin código (SPEC §14, Capa 3): datos → perfil → entrenar → evaluar → registrar. */
test("UC-01: de un CSV a un modelo registrado desde la UI", async ({ page }) => {
  const started = Date.now();
  await page.goto("/");
  await expect(page.getByText("Engine operativo")).toBeVisible();

  await page.getByRole("button", { name: "Nuevo proyecto" }).click();
  await page.getByLabel("Nombre").fill(`Churn E2E ${started}`);
  await page.getByLabel("Objetivo").fill("Anticipar qué clientes se van a dar de baja");
  await page.getByRole("button", { name: "Crear" }).click();

  await expect(page.getByText("Agregar datos")).toBeVisible();
  await page.getByTestId("file-input").setInputFiles(CHURN);
  await expect(page.getByText("Vista previa de churn.csv")).toBeVisible();
  await page.getByLabel("Columna objetivo").selectOption("churn");
  await page.getByRole("button", { name: "Crear versión de datos" }).click();
  await expect(page.getByText("Perfil del dataset")).toBeVisible({ timeout: 120_000 });

  await page.getByRole("link", { name: "Seguir: entrenar" }).click();
  await page.getByRole("button", { name: "Proponer preparación" }).click();
  await page.getByRole("button", { name: "Proponer arquitecturas" }).click();
  await expect(page.getByText("Por reglas").first()).toBeVisible({ timeout: 120_000 });
  await page.getByLabel("Intentos (trials)").fill("3");
  await page.getByLabel("Épocas máximas por intento").fill("5");
  await page.getByRole("button", { name: "Recomendar estrategia" }).click();
  await page.getByRole("button", { name: "Entrenar ahora" }).click();

  await expect(page.getByText("Entrenamiento en vivo")).toBeVisible();
  await expect(page.getByText("Terminado").first()).toBeVisible({ timeout: 10 * 60_000 });

  await page
    .getByRole("link", { name: /^t\d{3}$/ })
    .first()
    .click();
  await page.getByRole("button", { name: "Evaluar en test" }).click();
  await expect(page.getByText("roc_auc", { exact: true })).toBeVisible({ timeout: 120_000 });
  await page.getByRole("button", { name: "Registrar modelo" }).click();
  await expect(page.getByRole("button", { name: "Modelo registrado" })).toBeVisible();

  expect(Date.now() - started).toBeLessThan(15 * 60_000); // O1
});
