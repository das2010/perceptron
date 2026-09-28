import path from "node:path";
import { fileURLToPath } from "node:url";

import { expect, type Page, test } from "@playwright/test";

import { withDiagnostics } from "./diagnostics";

const here = path.dirname(fileURLToPath(import.meta.url));
const DEFECTS = path.resolve(here, "../../fixtures/uc04_defects");

withDiagnostics();

async function next(page: Page, step: string) {
  await page.getByRole("button", { name: "Siguiente" }).click();
  await expect(page.locator('[aria-current="step"]')).toHaveText(step);
}

/**
 * UC-04 guiado sin código por el wizard (SPEC §14, Capa 3): carpeta de imágenes por clase →
 * perfil → propuesta por reglas (revisada en el editor visual y como código) → HPO → entrenar
 * → evaluar en el test sellado → registrar.
 */
test("UC-04: de una carpeta de imágenes a un modelo registrado con el wizard", async ({ page }) => {
  const started = Date.now();
  await page.goto("/");
  await expect(page.getByText("Engine operativo")).toBeVisible();

  await page.getByRole("button", { name: "Nuevo proyecto" }).click();
  await page.getByLabel("Nombre").fill(`Defectos E2E ${started}`);
  await page.getByLabel("Objetivo").fill("Detectar piezas con defectos en la línea");
  await page.getByRole("button", { name: "Crear" }).click();
  await page.getByRole("link", { name: "Wizard" }).click();

  // 1. Objetivo
  await expect(page.locator('[aria-current="step"]')).toHaveText("1. Objetivo");
  await page.getByLabel("¿Qué querés lograr?").fill("Separar piezas buenas de piezas con defectos");
  await next(page, "2. Datos");

  // 2. Datos: carpeta con una subcarpeta por clase (y las anotaciones COCO en la raíz).
  await page.getByTestId("folder-input").setInputFiles(DEFECTS);
  await expect(page.getByText("Vista previa de uc04_defects")).toBeVisible({ timeout: 60_000 });
  await page.getByRole("button", { name: "Crear versión de datos" }).click();
  await expect(page.getByLabel("Versión de datos")).not.toHaveValue("", { timeout: 120_000 });
  await next(page, "3. Calidad");
  await expect(page.getByText("Perfil del dataset")).toBeVisible({ timeout: 120_000 });
  await next(page, "4. Etiquetado");
  await next(page, "5. Tarea y métrica");
  await expect(page.getByLabel("Tarea")).toHaveValue("classification");
  await next(page, "6. Arquitectura");

  // 6. Arquitectura por reglas, revisada en el editor visual (RF-ARC-05) y como código (RF-ARC-07).
  await page.getByRole("button", { name: "Proponer arquitecturas" }).click();
  await expect(page.getByText("Por reglas").first()).toBeVisible({ timeout: 120_000 });
  await page.getByRole("link", { name: "Abrir en el editor visual" }).first().click();
  await expect(page.getByText("Válida", { exact: true })).toBeVisible({ timeout: 60_000 });
  await expect(page.getByText(/parámetros · ~/)).toBeVisible();
  await page.getByRole("button", { name: "Ver como código" }).click();
  await expect(page.locator(".monaco-editor .view-lines")).toContainText("torch", {
    timeout: 60_000,
  });
  await page.getByRole("link", { name: "Wizard" }).click();
  await expect(page.locator('[aria-current="step"]')).toHaveText("6. Arquitectura");
  await expect(page.getByText(/Arquitectura elegida/)).toBeVisible();
  await next(page, "7. Búsqueda de hiperparámetros");

  // 7. HPO con presupuesto chico para el CI.
  await page.getByLabel("Intentos (trials)").fill("3");
  await page.getByLabel("Épocas máximas por intento").fill("10");
  await page.getByRole("button", { name: "Recomendar estrategia" }).click();
  await expect(page.getByText(/poda/)).toBeVisible({ timeout: 60_000 });
  await next(page, "8. Presupuesto y hardware");
  await next(page, "9. Revisión y lanzamiento");
  await expect(
    page.getByText("Se van a entrenar hasta 3 intentos de hasta 10 épocas", { exact: false }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Entrenar ahora" }).click();

  await expect(page.getByText("Entrenamiento en vivo")).toBeVisible();
  const live = page.locator("div", { has: page.getByText("Entrenamiento en vivo") }).last();
  await expect(live.getByText("Terminado")).toBeVisible({ timeout: 10 * 60_000 });

  await page
    .getByRole("link", { name: /^t\d{3}$/ })
    .first()
    .click();
  const evaluate = page.getByRole("button", { name: "Evaluar en test" });
  await expect(evaluate).toBeVisible({ timeout: 60_000 });
  await evaluate.click();
  await expect(page.getByText("accuracy", { exact: true })).toBeVisible({ timeout: 120_000 });
  await page.getByRole("button", { name: "Registrar modelo" }).click();
  await expect(page.getByRole("button", { name: "Modelo registrado" })).toBeVisible();

  expect(Date.now() - started).toBeLessThan(15 * 60_000); // O1
});
