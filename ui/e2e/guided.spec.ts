import path from "node:path";
import { fileURLToPath } from "node:url";

import { expect, test, type Page } from "@playwright/test";

import { withDiagnostics } from "./diagnostics";

const here = path.dirname(fileURLToPath(import.meta.url));
const CHURN = path.resolve(here, "../../fixtures/uc01_churn/churn.csv");

withDiagnostics();

async function answer(page: Page, label: string, value: string, select = true) {
  const field = page.getByLabel(label);
  await expect(field).toBeVisible();
  if (select) await field.selectOption(value);
  else await field.fill(value);
  await page.getByRole("button", { name: "Responder" }).click();
  await expect(page.getByLabel(label)).toBeHidden();
}

/**
 * Experiencia principal (ADR-0042) sin LLM: el asistente pregunta con los controles de respaldo,
 * explica los datos, revisa, diseña, entrena y evalúa solo, avisa y guía la validación.
 */
test("asistente guiado: de la descripción del problema a validar el modelo", async ({ page }) => {
  await page.goto("/");
  await page
    .getByLabel("¿Qué querés predecir?")
    .fill("Quiero anticipar qué clientes se van a dar de baja");
  await page.getByRole("button", { name: "Empezar" }).click();

  // 1. El caso: una pregunta por vez; sin LLM se responde con el control de respaldo.
  await expect(page.getByText("Quiero anticipar qué clientes se van a dar de baja")).toBeVisible();
  await answer(page, "¿Qué datos tenés?", "table");
  await answer(page, "¿Qué querés obtener?", "category");
  await answer(
    page,
    "¿Qué querés predecir, con tus palabras?",
    "si el cliente se da de baja",
    false,
  );
  await answer(page, "¿Ya tenés ejemplos con la respuesta correcta?", "yes");
  await answer(page, "¿Qué error es peor?", "symmetric");
  await answer(page, "¿Qué porcentaje de aciertos te sirve, como mínimo?", "60", false);
  await expect(page.getByRole("link", { name: "Datos", exact: true })).toBeHidden();
  await page.getByRole("button", { name: "Sí, es esto" }).click();

  // 2. Los datos: guía con ejemplo y plantilla; se sube el CSV.
  await expect(page.getByRole("heading", { name: "Qué datos necesito" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Descargar plantilla" })).toBeVisible();
  await page.getByLabel("Subir los datos").setInputFiles(CHURN);

  // 3. Revisión (si no detecta la columna resultado, se elige de la lista).
  const train = page.getByRole("button", { name: "Preparar y entrenar" });
  const target = page.getByLabel("¿Cuál columna tiene el resultado?");
  await expect(train.or(target)).toBeVisible({ timeout: 120_000 });
  if (await target.isVisible()) {
    await target.selectOption("churn");
    await page.getByRole("button", { name: "Revisar otra vez con esta columna" }).click();
  }
  await expect(page.getByText("Resultado a predecir: «churn»")).toBeVisible({ timeout: 120_000 });
  await train.click();

  // 4. Entrenamiento automático con etapas; aviso al terminar.
  await expect(page.getByRole("heading", { name: "Trabajando en tu modelo" })).toBeVisible();
  await expect(page.getByText("Tu modelo terminó de entrenar. Mirá el resultado.")).toBeVisible({
    timeout: 13 * 60_000,
  });

  // 5–6. Resultado explicado y validación con un caso nuevo: solo los campos que usa el modelo.
  await expect(page.getByRole("heading", { name: "Resultado" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Validar con casos nuevos" })).toBeVisible();
  await expect(page.getByLabel("customer_id")).toBeHidden();
  await expect(page.getByLabel("churn", { exact: true })).toBeHidden();
  await page.getByLabel("edad", { exact: true }).fill("41");
  await page.getByLabel("antiguedad_meses", { exact: true }).fill("3");
  await page.getByLabel("Resultado real").selectOption({ index: 1 });
  await page.getByRole("button", { name: "Comparar" }).click();
  await expect(page.getByText(/El modelo dijo:/)).toBeVisible({ timeout: 60_000 });
  await expect(
    page.getByText("1 de 1 casos nuevos bien").or(page.getByText("0 de 1 casos nuevos bien")),
  ).toBeVisible();
});
