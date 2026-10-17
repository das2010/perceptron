import { QueryClient } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { App } from "@/app/App";
import { Providers } from "@/app/providers";
import { createTestRouter } from "@/app/router";
import { resetApiClient } from "@/lib/api/client";
import { mockEngine, project } from "@/test/engine";

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

function mount(
  journey: Record<string, unknown>,
  extra: Record<string, (r: Request) => unknown> = {},
) {
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

beforeEach(() => resetApiClient());
afterEach(() => vi.unstubAllGlobals());

it("entrevista: muestra la conversación y solo el próximo dato que falta", async () => {
  const { calls } = mount({
    ...base,
    facts: { data_kind: "table" },
    pending: ["problem", "prediction"],
    messages: [
      { role: "user", text: "Quiero estimar el precio de las casas" },
      { role: "assistant", text: "¿Querés estimar un número o elegir una categoría?" },
    ],
  });
  expect(await screen.findByText("Quiero estimar el precio de las casas")).toBeInTheDocument();
  expect(screen.getByLabelText("¿Qué querés obtener?")).toBeInTheDocument();
  expect(
    screen.queryByLabelText("¿Qué querés predecir, con tus palabras?"),
  ).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Preparar y entrenar" })).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Sí, es esto" })).not.toBeInTheDocument();
  // Las herramientas técnicas quedan plegadas en «Opciones avanzadas».
  expect(screen.getByText("Opciones avanzadas").closest("details")).not.toHaveAttribute("open");

  await userEvent.selectOptions(screen.getByLabelText("¿Qué querés obtener?"), "value");
  await userEvent.click(screen.getByRole("button", { name: "Responder" }));
  await waitFor(() => expect(calls.some((c) => c.method === "POST")).toBe(true));
  const post = calls.find((c) => c.method === "POST");
  const body = (await post?.clone().json()) as Record<string, unknown>;
  expect(body).toMatchObject({ action: "message", version: 3, facts: { problem: "value" } });
});

it("con todo respondido pide confirmar lo que entendió", async () => {
  mount({
    ...base,
    pending: [],
    facts: { data_kind: "table", problem: "value", success_target: 10 },
    messages: [{ role: "user", text: "Estimar precios" }],
  });
  expect(await screen.findByRole("button", { name: "Sí, es esto" })).toBeInTheDocument();
  expect(screen.getByText("Una tabla (planilla, CSV, Excel)")).toBeInTheDocument();
  expect(screen.getByText("¿Qué error típico aceptás, como máximo (en %)?")).toBeInTheDocument();
});

it("datos: explica qué subir con un ejemplo y acepta solo los formatos del caso", async () => {
  mount({
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
    messages: [{ role: "user", text: "Estimar precios" }],
  });
  expect(await screen.findByText("Una fila por casa.")).toBeInTheDocument();
  expect(screen.getByText("95000")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Descargar plantilla" })).toBeInTheDocument();
  expect(screen.getByLabelText("Subir los datos")).toHaveAttribute("accept", ".csv,.xlsx");
  expect(screen.queryByLabelText(/Contame más/)).not.toBeInTheDocument();
});

it("entrenamiento: muestra etapas y progreso, sin otros controles", async () => {
  mount({
    ...base,
    state: "training",
    stage: "learning",
    confirmed: true,
    pending: [],
    progress: { done: 2, total: 8 },
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
};

it("validación: compara un caso nuevo con su resultado real", async () => {
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
  expect(
    await screen.findByText("Tu modelo terminó de entrenar. Mirá el resultado."),
  ).toBeInTheDocument();
  expect(screen.getByText("El error típico es de 4 %.")).toBeInTheDocument();
  await userEvent.type(screen.getByLabelText("superficie"), "130");
  await userEvent.selectOptions(screen.getByLabelText("barrio"), "Norte");
  await userEvent.type(screen.getByLabelText("Resultado real"), "98000");
  await userEvent.click(screen.getByRole("button", { name: "Comparar" }));
  expect(await screen.findByText("Coincide")).toBeInTheDocument();
  const post = calls.find((c) => c.url.endsWith("/journey/validate"));
  expect(await post?.clone().json()).toEqual({
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
  expect(screen.queryByLabelText("precio")).not.toBeInTheDocument();
  const use = screen.getByRole("button", { name: "Predecir" }).closest("form") as HTMLElement;
  await userEvent.type(use.querySelector("input") as HTMLInputElement, "90");
  await userEvent.click(screen.getByRole("button", { name: "Predecir" }));
  expect(await screen.findByText(/Predicción:/)).toBeInTheDocument();
  const post = calls.find((c) => c.url.endsWith("/runs/run_1/predict"));
  expect(await post?.clone().json()).toEqual({ rows: [{ superficie: 90, barrio: null }] });
});
