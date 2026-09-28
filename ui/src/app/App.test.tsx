import { QueryClient } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { resetApiClient } from "@/lib/api/client";
import { hardware, health, mockEngine, project } from "@/test/engine";

import { App } from "./App";
import { Providers } from "./providers";
import { createTestRouter } from "./router";

function renderApp(path = "/") {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <Providers client={client}>
      <App router={createTestRouter(path)} />
    </Providers>,
  );
}

describe("App", () => {
  beforeEach(() => resetApiClient());
  afterEach(() => {
    vi.unstubAllGlobals();
    delete document.documentElement.dataset.theme;
    localStorage.clear();
  });

  it("muestra el estado del Engine, el hardware y los proyectos", async () => {
    const { calls } = mockEngine({
      "GET /api/v1/system/health": () => health,
      "GET /api/v1/system/hardware": () => hardware,
      "GET /api/v1/projects": () => [project()],
    });
    renderApp();
    expect(await screen.findByText("Engine operativo")).toBeInTheDocument();
    expect(screen.getByText("Versión 0.1.0")).toBeInTheDocument();
    expect(await screen.findByText("8 núcleos")).toBeInTheDocument();
    expect((await screen.findAllByText("Churn")).length).toBeGreaterThan(0);
    expect(calls.some((r) => new URL(r.url).pathname === "/api/v1/system/health")).toBe(true);
  });

  it("muestra error y permite reintentar", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.reject(new TypeError("network"))),
    );
    renderApp();
    // El estado del Engine y la lista de proyectos muestran el mismo aviso de conexión.
    expect(
      (await screen.findAllByText("No se pudo conectar con el Engine")).length,
    ).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: "Reintentar" })).toBeInTheDocument();
  });

  it("cambia de idioma a inglés", async () => {
    mockEngine({ "GET /api/v1/projects": () => [] });
    renderApp();
    await userEvent.selectOptions(await screen.findByLabelText("Idioma"), "en");
    expect(
      await screen.findByText("Train neural networks locally, guided by AI"),
    ).toBeInTheDocument();
    await userEvent.selectOptions(screen.getByLabelText("Language"), "es");
  });

  it("aplica y recuerda el tema elegido", async () => {
    mockEngine({ "GET /api/v1/projects": () => [] });
    renderApp();
    await userEvent.selectOptions(await screen.findByLabelText("Tema"), "dark");
    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(localStorage.getItem("perceptron.theme")).toBe("dark");
  });

  it("crea un proyecto y abre la pestaña de datos", async () => {
    const created = project({ id: "prj_nuevo", name: "Motores" });
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
      "GET /api/v1/projects/prj_nuevo/datasets": () => [],
    });
    renderApp();
    await userEvent.click(await screen.findByRole("button", { name: "Nuevo proyecto" }));
    await userEvent.type(screen.getByLabelText("Nombre"), "Motores");
    await userEvent.click(screen.getByRole("button", { name: "Crear" }));
    expect(await screen.findByText("Agregar datos")).toBeInTheDocument();
    expect(body).toMatchObject({ name: "Motores", privacy_level: "L1" });
  });
});
