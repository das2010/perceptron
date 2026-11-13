import path from "node:path";
import { fileURLToPath } from "node:url";

import { expect, test, type Page } from "@playwright/test";

import { withDiagnostics } from "./diagnostics";

const here = path.dirname(fileURLToPath(import.meta.url));
const CHURN = path.resolve(here, "../../fixtures/uc01_churn/churn.csv");

withDiagnostics();

const thread = (page: Page) => page.getByRole("region", { name: "Conversación con el asistente" });
const bubbles = (page: Page) => thread(page).locator("p.whitespace-pre-wrap");

/** Elige una respuesta rápida y espera la respuesta del asistente. */
async function pick(page: Page, label: string) {
  const before = await bubbles(page).count();
  await thread(page).getByRole("button", { name: label, exact: true }).click();
  await expect(bubbles(page)).toHaveCount(before + 2);
}

/** Escribe en la caja de texto (Enter envía) y espera la respuesta del asistente. */
async function write(page: Page, text: string) {
  const before = await bubbles(page).count();
  const box = page.getByLabel(/Contame más|Preguntame/);
  await box.fill(text);
  await box.press("Enter");
  await expect(bubbles(page)).toHaveCount(before + 2);
}

/**
 * Experiencia principal (ADR-0042) sin LLM: el asistente conversa con preguntas redactadas y
 * respuestas rápidas, explica los datos, revisa, diseña, entrena y evalúa solo, y guía la
 * validación, todo en el mismo hilo.
 */
test("asistente guiado: de la descripción del problema a validar el modelo", async ({ page }) => {
  await page.goto("/");
  await page
    .getByLabel("¿Qué querés predecir?")
    .fill("Quiero anticipar qué clientes se van a dar de baja");
  await page.getByRole("button", { name: "Empezar" }).click();

  // 1. El caso: el asistente saluda, pregunta de a una cosa y acusa cada respuesta.
  await expect(
    thread(page).getByText("Quiero anticipar qué clientes se van a dar de baja"),
  ).toBeVisible();
  await expect(thread(page).getByText(/¡Hola! Te ayudo a armarlo paso a paso/)).toBeVisible();
  await pick(page, "Una planilla (Excel o CSV)");
  await pick(page, "Elegir entre respuestas posibles");
  await write(page, "si el cliente se da de baja");
  await pick(page, "Sí, ya está la respuesta");
  await pick(page, "Los dos me preocupan igual");
  await pick(page, "7 de cada 10 (70 %)");
  await expect(thread(page).getByText(/A ver si entendí bien/)).toBeVisible();
  await expect(page.getByRole("link", { name: "Datos", exact: true })).toBeHidden();
  await page.getByRole("button", { name: "Sí, es esto" }).click();

  // 2. Los datos: el asistente lo cuenta y adjunta la guía con ejemplo y plantilla.
  await expect(thread(page).getByText(/Ahora necesito tus datos/)).toBeVisible();
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
  await expect(thread(page).getByText(/Revisé tus datos: \d+ ejemplos/)).toBeVisible({
    timeout: 120_000,
  });
  await expect(page.getByText("Resultado a predecir: «churn»")).toBeVisible();
  await train.click();

  // 4. Entrenamiento automático con etapas; el resultado llega al mismo hilo.
  await expect(thread(page).getByText(/¡Arranco!/)).toBeVisible();
  await expect(page.getByRole("heading", { name: "Validar con casos nuevos" })).toBeVisible({
    timeout: 13 * 60_000,
  });
  await expect(thread(page).getByText(/Sobre \d+ casos que el modelo nunca vio/)).toBeVisible();

  // 5–6. Validación con un caso nuevo: solo los campos que usa el modelo, con nombres legibles.
  await expect(page.getByLabel("Customer id")).toBeHidden();
  await expect(page.getByLabel("Churn", { exact: true })).toBeHidden();
  await page.getByLabel("Edad", { exact: true }).fill("41");
  await page.getByLabel("Antiguedad meses", { exact: true }).fill("3");
  await page.getByLabel("Resultado real").selectOption({ index: 1 });
  await page.getByRole("button", { name: "Comparar" }).click();
  await expect(page.getByText(/El modelo dijo:/)).toBeVisible({ timeout: 60_000 });
  await expect(
    page.getByText("1 de 1 casos nuevos bien").or(page.getByText("0 de 1 casos nuevos bien")),
  ).toBeVisible();

  // Se le puede preguntar al asistente en cualquier etapa (sin IA, dice qué se puede hacer).
  await write(page, "¿Qué hago ahora?");
  await expect(
    thread(page).getByText(/Probalo con casos nuevos|Podés subir datos nuevos/),
  ).toBeVisible();
});
