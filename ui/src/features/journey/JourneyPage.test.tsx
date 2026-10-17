import { QueryClient } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, afterEach, expect, it, vi } from "vitest";
import { App } from "@/app/App";
import { Providers } from "@/app/providers";
import { createTestRouter } from "@/app/router";
import { resetApiClient } from "@/lib/api/client";
import { postForm } from "@/lib/api/upload";
import { mockEngine, project } from "@/test/engine";

vi.mock("@/lib/api/upload", () => ({ postForm: vi.fn() }));
const base = {
  version: 1,
  state: "interview",
  stage: "interview",
  facts: {},
  pending: ["modality"],
  messages: [],
  assumptions: [],
  blockers: [],
  warnings: [],
};
function mount(journey: Record<string, unknown> = base) {
  const routes = mockEngine({
    "GET /api/v1/projects": () => [project()],
    "GET /api/v1/projects/prj_1": () => project(),
    "GET /api/v1/projects/prj_1/journey": () => journey,
  });
  const tree = render(
    <Providers client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <App router={createTestRouter("/projects/prj_1")} />
    </Providers>,
  );
  return { ...tree, ...routes };
}
beforeEach(() => {
  resetApiClient();
  vi.clearAllMocks();
});
afterEach(() => vi.unstubAllGlobals());

it("recovers the conversation and shows only the next essential question", async () => {
  mount({
    ...base,
    facts: { modality: "image" },
    pending: ["task", "classes"],
    messages: [{ role: "user", text: "Necesito detectar fisuras en fotos" }],
  });
  expect(await screen.findByText("Necesito detectar fisuras en fotos")).toBeInTheDocument();
  expect(screen.getByText("¿Qué resultado necesitás para cada imagen?")).toBeInTheDocument();
  expect(
    screen.queryByText("¿Cuáles son las categorías? Separalas con comas."),
  ).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Preparar y entrenar" })).not.toBeInTheDocument();
  expect(screen.getByText("Opciones avanzadas")).toBeInTheDocument();
});

it("restores completion and keeps actual labels out of prediction requests", async () => {
  const j = {
    ...base,
    state: "ready_to_validate",
    stage: "complete",
    notification: "training_finished",
    contract: { inputs: { kind: "image" }, outputs: { classes: ["fisura", "sano"] } },
    facts: { positive_class: "fisura" },
    validations: [],
  };
  mount(j);
  expect(
    await screen.findByText(
      "El entrenamiento terminó. Revisá el resultado y validá con casos nuevos.",
    ),
  ).toBeInTheDocument();
  expect(screen.queryByText("Resultado real (solo para comparar)")).not.toBeInTheDocument();
  vi.mocked(postForm).mockResolvedValue({
    prediction: "fisura",
    actual: null,
    correct: null,
    journey: j,
  });
  const image = new File(["pixels"], "foto.png", { type: "image/png" });
  await userEvent.upload(screen.getByLabelText("Imagen"), image);
  const buttons = screen.getAllByRole("button", { name: "Probar una predicción" });
  expect((screen.getByLabelText("Imagen") as HTMLInputElement).files).toHaveLength(1);
  expect(buttons.at(-1)).toBeEnabled();
  const form = screen.getByLabelText("Imagen").closest("form");
  if (!form) throw new Error("Missing image form");
  fireEvent.submit(form);
  await waitFor(() => expect(postForm).toHaveBeenCalledTimes(1));
  expect(vi.mocked(postForm).mock.calls[0]?.[0]).toContain("/journey/predict");
  expect(vi.mocked(postForm).mock.calls[0]?.[1].has("actual")).toBe(false);
  await userEvent.click(screen.getByRole("button", { name: "Validar con casos conocidos" }));
  await userEvent.selectOptions(
    screen.getByLabelText("Resultado real (solo para comparar)"),
    "sano",
  );
  fireEvent.submit(form);
  await waitFor(() => expect(postForm).toHaveBeenCalledTimes(2));
  expect(vi.mocked(postForm).mock.calls[1]?.[0]).toContain("/journey/validate");
  expect(vi.mocked(postForm).mock.calls[1]?.[1].get("actual")).toBe("sano");
});
