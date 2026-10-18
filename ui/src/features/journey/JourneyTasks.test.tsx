import { QueryClient } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { App } from "@/app/App";
import { Providers } from "@/app/providers";
import { createTestRouter } from "@/app/router";
import { resetApiClient } from "@/lib/api/client";
import { postForm } from "@/lib/api/upload";
import { mockEngine, project } from "@/test/engine";

vi.mock("@/lib/api/upload", () => ({ postForm: vi.fn() }));

const base = {
  version: 2,
  schema_version: 2,
  confirmed: true,
  llm_failed: false,
  stop_requested: false,
  pending: [],
  assumptions: [],
  blockers: [],
  warnings: [],
  validations: [],
  stage: "complete",
  run_id: "run_1",
};

function mount(journey: Record<string, unknown>, extra: Record<string, () => unknown> = {}) {
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

beforeEach(() => {
  resetApiClient();
  vi.clearAllMocks();
});
afterEach(() => vi.unstubAllGlobals());

it("con fotos ofrece ubicar objetos, marcar zonas o leer texto", async () => {
  mount({
    ...base,
    state: "interview",
    stage: "interview",
    confirmed: false,
    run_id: null,
    facts: { data_kind: "images" },
    pending: ["problem"],
    messages: [
      { role: "user", text: "Tengo fotos de piezas", step: "case" },
      {
        role: "assistant",
        text: "¿Qué querés que haga con cada foto?",
        step: "case",
        field: "problem",
      },
    ],
  });
  expect(
    await screen.findByRole("button", { name: "Ubicar dónde están los objetos" }),
  ).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Marcar la zona exacta" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Leer un texto" })).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Estimar un número" })).not.toBeInTheDocument();
});

it("ubicar objetos: muestra lo que encontró y la persona dice si está bien", async () => {
  const journey = {
    ...base,
    state: "ready_to_validate",
    facts: { data_kind: "images", problem: "locate", success_target: 50 },
    contract: { outputs: { task: "object_detection" } },
    form: { kind: "image", task: "object_detection", fields: [] },
    result: { meets_target: true, explanation: [], next_steps: [] },
    messages: [{ role: "assistant", text: "Ubica bien el 62 % de los objetos.", step: "validate" }],
  };
  mount(journey, {
    "POST /api/v1/runs/run_1/predict/file": () => ({
      task: "object_detection",
      input_kind: "image",
      predictions: [
        {
          prediction: 2,
          details: {
            objects: [
              { label: "fisura", score: 0.91, box: [0.1, 0.1, 0.3, 0.3] },
              { label: "fisura", score: 0.64, box: [0.5, 0.5, 0.7, 0.8] },
            ],
          },
        },
      ],
    }),
  });
  vi.mocked(postForm).mockResolvedValue({
    prediction: 2,
    actual: "bien",
    correct: true,
    checked: 1,
    journey: { ...journey, validations: [{ key: "k", correct: true }] },
  });
  const photo = new File(["pixels"], "pieza.png", { type: "image/png" });
  await userEvent.upload(await screen.findByLabelText("Archivo"), photo);
  expect(await screen.findByText("Encontré 2 objetos.")).toBeInTheDocument();
  expect(screen.getByText("fisura (91 % de seguridad)")).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Sí, está bien" }));
  await waitFor(() => expect(postForm).toHaveBeenCalledTimes(1));
  const [url, body] = vi.mocked(postForm).mock.calls[0] ?? [];
  expect(url).toContain("/journey/validate/file");
  expect((body as FormData).get("actual")).toBe("sí");
});

it("pronóstico: se sube la historia y muestra los próximos valores", async () => {
  mount({
    ...base,
    state: "ready_to_use",
    facts: { data_kind: "table", problem: "forecast", horizon: 2 },
    contract: { outputs: { task: "forecasting" } },
    form: {
      kind: "series",
      task: "forecasting",
      history: 8,
      horizon: 2,
      fields: [
        { name: "semana", kind: "date", options: [] },
        { name: "sku", kind: "category", options: [] },
        { name: "unidades", kind: "number", options: [] },
      ],
    },
    result: { meets_target: true, explanation: [], next_steps: [] },
    messages: [{ role: "assistant", text: "Ya podés usarlo.", step: "use" }],
  });
  vi.mocked(postForm).mockResolvedValue({
    task: "forecasting",
    predictions: [
      {
        prediction: [140.5, 151.25],
        details: { series: "SKU-000", times: ["2025-01-06T00:00:00", "2025-01-13T00:00:00"] },
      },
    ],
  });
  expect(await screen.findByText(/al menos 8 por serie/)).toBeInTheDocument();
  expect(
    screen.getAllByText("CSV o Excel con las columnas: semana, sku, unidades")[0],
  ).toBeInTheDocument();
  const csv = new File(["semana,sku,unidades\n"], "historia.csv", { type: "text/csv" });
  await userEvent.upload(screen.getByLabelText("Subí la historia reciente"), csv);
  expect(await screen.findByText("Serie SKU-000")).toBeInTheDocument();
  expect(screen.getByText("2025-01-06 00:00:00")).toBeInTheDocument();
  expect(vi.mocked(postForm).mock.calls[0]?.[0]).toContain("/journey/predict/series");
});

it("leer texto: se compara con el texto real", async () => {
  mount({
    ...base,
    state: "ready_to_validate",
    facts: { data_kind: "images", problem: "read_text", success_target: 90 },
    contract: { outputs: { task: "ocr" } },
    form: { kind: "image", task: "ocr", fields: [] },
    result: { meets_target: true, explanation: [], next_steps: [] },
    messages: [
      { role: "assistant", text: "Lee bien el 95 % de los caracteres.", step: "validate" },
    ],
  });
  vi.mocked(postForm).mockResolvedValue({
    prediction: "LOTE 4471-B",
    actual: "LOTE 4471-B",
    correct: true,
    checked: 1,
    journey: { ...base, state: "ready_to_validate" },
  });
  await userEvent.upload(
    await screen.findByLabelText("Archivo"),
    new File(["px"], "etiqueta.png", { type: "image/png" }),
  );
  await userEvent.type(screen.getByLabelText("Texto real"), "LOTE 4471-B");
  // jsdom no da por completo un input de archivo obligatorio: se envía el formulario.
  fireEvent.submit(
    screen.getByRole("button", { name: "Comparar" }).closest("form") as HTMLFormElement,
  );
  await waitFor(() => expect(postForm).toHaveBeenCalledTimes(1));
  expect((vi.mocked(postForm).mock.calls[0]?.[1] as FormData).get("actual")).toBe("LOTE 4471-B");
});
