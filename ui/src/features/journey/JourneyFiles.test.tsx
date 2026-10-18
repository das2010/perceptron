import { QueryClient } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { App } from "@/app/App";
import { Providers } from "@/app/providers";
import { createTestRouter } from "@/app/router";
import { resetApiClient } from "@/lib/api/client";
import { postForm } from "@/lib/api/upload";
import { mockEngine, project } from "@/test/engine";

import { FIRST_FILES, FIRST_MESSAGE } from "./journeyStart";

vi.mock("@/lib/api/upload", () => ({ postForm: vi.fn() }));

const base = {
  version: 4,
  schema_version: 2,
  state: "interview",
  stage: "interview",
  confirmed: false,
  llm_failed: false,
  stop_requested: false,
  facts: {},
  pending: ["data_kind", "problem"],
  messages: [],
  assumptions: [],
  blockers: [],
  warnings: [],
};

const goal = (id: string, title: string, detail: string, problem: string) => ({
  id,
  title,
  detail,
  facts: { data_kind: "table", problem },
  parts: ["churn.csv"],
  keys: [],
});

const proposed = {
  ...base,
  source_id: "src_1",
  goals: [
    goal("g1", "Predecir «churn»", "elegir entre 2 respuestas posibles", "category"),
    goal("g2", "Estimar «cargo_mensual»", "un número, con las otras 6 columnas", "value"),
  ],
  messages: [
    { role: "user", text: "Te paso «churn.csv».", step: "case", action: "files" },
    {
      role: "assistant",
      text: "Leí «churn.csv» (1.000 filas y 8 columnas). Con esto, te propongo estos objetivos.",
      step: "case",
      goals: ["g1", "g2"],
    },
  ],
};

function mount(journey: Record<string, unknown>) {
  const routes = mockEngine({
    "GET /api/v1/projects": () => [project()],
    "GET /api/v1/projects/prj_1": () => project(),
    "GET /api/v1/projects/prj_1/journey": () => journey,
    "POST /api/v1/projects/prj_1/journey": () => journey,
  });
  render(
    <Providers client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <App router={createTestRouter("/projects/prj_1")} />
    </Providers>,
  );
  return routes;
}

async function posted(calls: Request[]) {
  await waitFor(() =>
    expect(calls.some((c) => c.method === "POST" && c.url.endsWith("/journey"))).toBe(true),
  );
  const post = calls.filter((c) => c.method === "POST" && c.url.endsWith("/journey")).at(-1);
  return (await post?.clone().json()) as Record<string, unknown>;
}

beforeEach(() => {
  resetApiClient();
  vi.clearAllMocks();
  sessionStorage.clear();
  FIRST_FILES.clear();
});
afterEach(() => vi.unstubAllGlobals());

it("los objetivos propuestos se eligen con un clic", async () => {
  const { calls } = mount(proposed);
  const cards = await screen.findByRole("list", { name: "Objetivos que te propongo" });
  expect(cards).toHaveTextContent("Estimar «cargo_mensual»");
  expect(cards).toHaveTextContent("un número, con las otras 6 columnas");
  await userEvent.click(screen.getByRole("button", { name: /Predecir «churn»/ }));
  expect(await posted(calls)).toMatchObject({ action: "choose", goal_id: "g1", version: 4 });
});

it("con el objetivo elegido ya no se ofrecen las tarjetas", async () => {
  mount({ ...proposed, goal_id: "g1" });
  expect(await screen.findByText(/Leí «churn.csv»/)).toBeInTheDocument();
  expect(screen.queryByRole("list", { name: "Objetivos que te propongo" })).toBeNull();
});

it("adjuntar archivos en la conversación los sube y le pide al asistente que los lea", async () => {
  vi.mocked(postForm).mockResolvedValue({ id: "src_9", name: "datos" });
  const { calls } = mount({
    ...base,
    messages: [
      { role: "user", text: "Tengo datos de clientes", step: "case" },
      { role: "assistant", text: "¿Qué datos tenés?", step: "case", field: "data_kind" },
    ],
  });
  const files = [
    new File(["a,b\n1,2\n"], "clientes.csv", { type: "text/csv" }),
    new File(["a,c\n1,3\n"], "pagos.csv", { type: "text/csv" }),
  ];
  await userEvent.upload(await screen.findByLabelText("Adjuntar archivos de datos"), files);
  await waitFor(() => expect(postForm).toHaveBeenCalledTimes(1));
  const [url, body] = vi.mocked(postForm).mock.calls[0] ?? [];
  expect(url).toBe("/api/v1/projects/prj_1/uploads");
  expect((body as FormData).getAll("files")).toHaveLength(2);
  expect(await posted(calls)).toMatchObject({ action: "files", source_id: "src_9", message: "" });
});

it("lo adjuntado en Inicio llega con el primer mensaje", async () => {
  vi.mocked(postForm).mockResolvedValue({ id: "src_2", name: "ventas.xlsx" });
  sessionStorage.setItem(FIRST_MESSAGE("prj_1"), "Quiero mejorar las ventas");
  FIRST_FILES.set("prj_1", [new File(["x"], "ventas.xlsx")]);
  const { calls } = mount(base);
  expect(await posted(calls)).toMatchObject({
    action: "files",
    source_id: "src_2",
    message: "Quiero mejorar las ventas",
  });
  expect(FIRST_FILES.has("prj_1")).toBe(false);
});

it("en Inicio se puede empezar solo con archivos", async () => {
  const created = project({ id: "prj_nuevo", name: "ventas.csv" });
  let body: Record<string, unknown> = {};
  mockEngine({
    "GET /api/v1/projects": () => [],
    "POST /api/v1/projects": async (req) => {
      body = (await req.json()) as Record<string, unknown>;
      return new Response(JSON.stringify(created), {
        status: 201,
        headers: { "Content-Type": "application/json" },
      });
    },
    "GET /api/v1/projects/prj_nuevo": () => created,
    "GET /api/v1/projects/prj_nuevo/journey": () => base,
    "POST /api/v1/projects/prj_nuevo/journey": () => base,
  });
  vi.mocked(postForm).mockReturnValue(new Promise(() => {}));
  render(
    <Providers client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <App router={createTestRouter("/")} />
    </Providers>,
  );
  const start = await screen.findByRole("button", { name: "Empezar" });
  expect(start).toBeDisabled();
  const file = new File(["a,b\n"], "ventas.csv", { type: "text/csv" });
  await userEvent.upload(screen.getByLabelText("Adjuntar archivos de datos"), file);
  expect(screen.getByRole("button", { name: "Quitar ventas.csv" })).toBeInTheDocument();
  await userEvent.click(start);
  await waitFor(() => expect(body).toMatchObject({ name: "ventas.csv", goal: "" }));
  // Al abrir el proyecto, el asistente recibe los archivos y muestra que los está leyendo.
  expect(await screen.findByRole("status", { name: "Leyendo tus archivos" })).toBeInTheDocument();
  expect(postForm).toHaveBeenCalledWith("/api/v1/projects/prj_nuevo/uploads", expect.any(FormData));
});
