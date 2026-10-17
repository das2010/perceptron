import { QueryClient } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { App } from "@/app/App";
import { Providers } from "@/app/providers";
import { createTestRouter } from "@/app/router";
import { resetApiClient } from "@/lib/api/client";
import { mockEngine, project } from "@/test/engine";

import { FIRST_MESSAGE } from "./JourneyPage";

const base = {
  version: 3,
  schema_version: 2,
  state: "interview",
  stage: "interview",
  confirmed: false,
  llm_failed: false,
  stop_requested: false,
  facts: {},
  pending: ["data_kind"],
  messages: [],
  assumptions: [],
  blockers: [],
  warnings: [],
};

type Handler = (r: Request) => unknown;

function mount(journey: Record<string, unknown>, extra: Record<string, Handler> = {}) {
  const routes = mockEngine({
    "GET /api/v1/projects": () => [project()],
    "GET /api/v1/projects/prj_1": () => project(),
    "GET /api/v1/projects/prj_1/journey": () => journey,
    "POST /api/v1/projects/prj_1/journey": () => journey,
    ...extra,
  });
  render(
    <Providers client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <App router={createTestRouter("/projects/prj_1")} />
    </Providers>,
  );
  return routes;
}

async function posted(calls: Request[], suffix = "/journey") {
  await waitFor(() =>
    expect(calls.some((c) => c.method === "POST" && c.url.endsWith(suffix))).toBe(true),
  );
  const post = calls.filter((c) => c.method === "POST" && c.url.endsWith(suffix)).at(-1);
  return (await post?.clone().json()) as Record<string, unknown>;
}

beforeEach(() => {
  resetApiClient();
  sessionStorage.clear();
});
afterEach(() => vi.unstubAllGlobals());

const interview = {
  ...base,
  facts: { data_kind: "table" },
  pending: ["problem", "prediction"],
  messages: [
    { role: "user", text: "Quiero estimar el precio de las casas", step: "case" },
    {
      role: "assistant",
      text: "Perfecto. ¿Qué te gustaría que te diga el sistema para cada casa?",
      step: "case",
      field: "problem",
    },
  ],
};

it("entrevista: el asistente pregunta y ofrece respuestas rápidas; elegir una responde", async () => {
  let release = () => {};
  const { calls } = mount(interview, {
    "POST /api/v1/projects/prj_1/journey": async () => {
      await new Promise<void>((r) => (release = r));
      return interview;
    },
  });
  expect(await screen.findByText("Quiero estimar el precio de las casas")).toBeInTheDocument();
  expect(screen.getByText(/¿Qué te gustaría que te diga el sistema/)).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Preparar y entrenar" })).not.toBeInTheDocument();
  expect(screen.getByText("Opciones avanzadas").closest("details")).not.toHaveAttribute("open");

  await userEvent.click(screen.getByRole("button", { name: "Estimar un número" }));
  // Mientras responde: el mensaje propio y el «escribiendo…».
  expect(
    await screen.findByRole("status", { name: "El asistente está escribiendo" }),
  ).toBeInTheDocument();
  expect(screen.getByText("Estimar un número", { selector: "p" })).toBeInTheDocument();
  const body = await posted(calls);
  expect(body).toMatchObject({
    action: "message",
    version: 3,
    message: "Estimar un número",
    facts: { problem: "value" },
  });
  release();
});

it("las respuestas propuestas por la IA se envían como texto, sin forzar el dato", async () => {
  const { calls } = mount({
    ...interview,
    messages: [
      ...interview.messages.slice(0, 1),
      {
        role: "assistant",
        text: "¿Ya figura el precio de venta de casas vendidas?",
        step: "case",
        field: "labels_available",
        options: ["Sí, tengo el precio de cada venta", "No, solo las características"],
      },
    ],
  });
  await userEvent.click(
    await screen.findByRole("button", { name: "Sí, tengo el precio de cada venta" }),
  );
  const body = await posted(calls);
  expect(body.message).toBe("Sí, tengo el precio de cada venta");
  expect(body.facts).toBeUndefined();
});

it("la caja de texto envía con Enter y agrega líneas con Shift+Enter", async () => {
  const { calls } = mount(interview);
  const box = await screen.findByLabelText(/Contame más/);
  await userEvent.type(box, "Son casas de la ciudad{Shift>}{Enter}{/Shift}y alrededores");
  expect(calls.some((c) => c.method === "POST")).toBe(false);
  await userEvent.type(box, "{Enter}");
  const body = await posted(calls);
  expect(body).toMatchObject({
    action: "message",
    message: "Son casas de la ciudad\ny alrededores",
  });
});

it("con todo respondido, confirma lo entendido o deja corregirlo", async () => {
  const { calls } = mount({
    ...base,
    pending: [],
    facts: { data_kind: "table", problem: "value", success_target: 10 },
    messages: [
      { role: "user", text: "Estimar precios", step: "case" },
      {
        role: "assistant",
        text: "A ver si entendí bien: tenés una planilla y querés estimar esto: precio. ¿Es así?",
        step: "case",
        action: "confirm",
      },
    ],
  });
  await userEvent.click(await screen.findByRole("button", { name: "Corregir algo" }));
  expect(screen.getByText("Lo que entendí").closest("details")).toHaveAttribute("open");
  expect(screen.getByText("Una tabla (planilla, CSV, Excel)")).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Sí, es esto" }));
  expect(await posted(calls)).toMatchObject({ action: "confirm" });
});

it("el primer mensaje escrito en Inicio se envía al abrir la conversación", async () => {
  sessionStorage.setItem(FIRST_MESSAGE("prj_1"), "Quiero saber qué clientes se van");
  const { calls } = mount(base);
  expect(await posted(calls)).toMatchObject({
    action: "message",
    message: "Quiero saber qué clientes se van",
  });
  expect(sessionStorage.getItem(FIRST_MESSAGE("prj_1"))).toBeNull();
});

it("datos: el asistente lo cuenta en el hilo y adjunta la guía y la subida", async () => {
  const { calls } = mount({
    ...base,
    state: "awaiting_data",
    stage: "awaiting_data",
    confirmed: true,
    pending: [],
    guide: {
      layout: "table",
      accept: [".csv", ".xlsx"],
      steps: ["Una fila por casa."],
      example_columns: ["superficie", "precio"],
      example_rows: [["120", "95000"]],
      example_tree: [],
      template_csv: "superficie,precio\n",
      tips: ["No incluyas datos que se conocen después."],
    },
    messages: [
      { role: "user", text: "Estimar precios", step: "case" },
      { role: "assistant", text: "¡Bien! Ahora necesito tus datos.", step: "data" },
    ],
  });
  expect(await screen.findByText("¡Bien! Ahora necesito tus datos.")).toBeInTheDocument();
  expect(screen.getByText("Una fila por casa.")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Descargar plantilla" })).toBeInTheDocument();
  expect(screen.getByLabelText("Subir los datos")).toHaveAttribute("accept", ".csv,.xlsx");
  // Siempre se le puede preguntar algo al asistente.
  await userEvent.type(
    screen.getByPlaceholderText("Preguntame lo que quieras sobre cómo preparar los datos…"),
    "¿Sirve un Excel?{Enter}",
  );
  expect(await posted(calls)).toMatchObject({ action: "message", message: "¿Sirve un Excel?" });
});

it("entrenamiento: etapas reales y progreso, con opción de detener", async () => {
  mount({
    ...base,
    state: "training",
    stage: "learning",
    confirmed: true,
    pending: [],
    progress: { done: 2, total: 8 },
    messages: [{ role: "assistant", text: "¡Arranco!", step: "training" }],
  });
  expect(await screen.findByText("Aprendiendo de tus datos")).toBeInTheDocument();
  expect(screen.getByText(/Prueba 3 de 8/)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Detener" })).toBeInTheDocument();
  expect(screen.queryByLabelText("Subir los datos")).not.toBeInTheDocument();
});

const model = {
  ...base,
  confirmed: true,
  pending: [],
  stage: "complete",
  run_id: "run_1",
  target: "precio",
  facts: { data_kind: "table", problem: "value", success_target: 10 },
  contract: { outputs: { task: "regression" } },
  form: {
    kind: "fields",
    fields: [
      { name: "superficie", kind: "number", options: [], example: "120" },
      { name: "barrio", kind: "category", options: ["Centro", "Norte"], example: "Centro" },
    ],
  },
  result: { meets_target: true, explanation: ["El error típico es de 4 %."], next_steps: [] },
  validations: [],
  messages: [
    {
      role: "assistant",
      text: "El error típico es de 4 %. Cumple el objetivo que definiste.",
      step: "validate",
    },
  ],
};

it("validación: el resultado va en el hilo y se compara un caso nuevo", async () => {
  const { calls } = mount(
    { ...model, state: "ready_to_validate", notification: "training_finished" },
    {
      "POST /api/v1/projects/prj_1/journey/validate": () => ({
        prediction: 100000,
        actual: 98000,
        correct: true,
        checked: 1,
        journey: { ...model, state: "ready_to_validate" },
      }),
    },
  );
  expect(await screen.findByText(/El error típico es de 4 %. Cumple/)).toBeInTheDocument();
  // El aviso de fin se da por visto al tener la conversación a la vista.
  expect(await posted(calls)).toMatchObject({ action: "acknowledge" });
  await userEvent.type(screen.getByLabelText("Superficie"), "130");
  await userEvent.selectOptions(screen.getByLabelText("Barrio"), "Norte");
  await userEvent.type(screen.getByLabelText("Resultado real"), "98000");
  await userEvent.click(screen.getByRole("button", { name: "Comparar" }));
  expect(await screen.findByText("Coincide")).toBeInTheDocument();
  expect(await posted(calls, "/journey/validate")).toEqual({
    row: { superficie: 130, barrio: "Norte" },
    actual: "98000",
  });
});

it("uso: pide solo los campos que el modelo necesita y predice con el run", async () => {
  const { calls } = mount(
    { ...model, state: "ready_to_use" },
    {
      "POST /api/v1/runs/run_1/predict": () => ({
        predictions: [{ prediction: 101500, confidence: null, probabilities: null }],
      }),
    },
  );
  expect(await screen.findByRole("heading", { name: "Usar el modelo" })).toBeInTheDocument();
  expect(screen.queryByLabelText("Precio")).not.toBeInTheDocument();
  const use = screen.getByRole("button", { name: "Predecir" }).closest("form") as HTMLElement;
  await userEvent.type(use.querySelector("input") as HTMLInputElement, "90");
  await userEvent.click(screen.getByRole("button", { name: "Predecir" }));
  expect(await screen.findByText(/Predicción:/)).toBeInTheDocument();
  expect(await posted(calls, "/runs/run_1/predict")).toEqual({
    rows: [{ superficie: 90, barrio: null }],
  });
});
