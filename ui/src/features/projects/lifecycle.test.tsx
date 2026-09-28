import { QueryClient } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "@/app/App";
import { Providers } from "@/app/providers";
import { createTestRouter } from "@/app/router";
import { resetApiClient } from "@/lib/api/client";
import { health, mockEngine, project } from "@/test/engine";

function renderAt(path: string) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <Providers client={client}>
      <App router={createTestRouter(path)} />
    </Providers>,
  );
}

function engine(extra: Record<string, (req: Request) => unknown> = {}) {
  const current = project({ id: "prj_1", name: "Churn", version: 3 });
  return mockEngine({
    "GET /api/v1/system/health": () => health,
    "GET /api/v1/projects": () => [
      current,
      project({ id: "prj_viejo", name: "Viejo", status: "archived" }),
    ],
    "GET /api/v1/projects/prj_1": () => current,
    "GET /api/v1/projects/prj_1/datasets": () => [],
    "GET /api/v1/projects/prj_1/runs": () => [],
    "GET /api/v1/projects/prj_1/models": () => [],
    ...extra,
  });
}

describe("ciclo de vida del proyecto (RF-PRJ-01)", () => {
  beforeEach(() => resetApiClient());
  afterEach(() => vi.unstubAllGlobals());

  it("los archivados no aparecen salvo que se pidan", async () => {
    engine();
    renderAt("/");
    expect((await screen.findAllByText("Churn")).length).toBeGreaterThan(0);
    expect(screen.queryByText("Viejo")).not.toBeInTheDocument();
    await userEvent.click(screen.getByLabelText("Mostrar el archivado (1)"));
    expect(await screen.findByText("Viejo")).toBeInTheDocument();
  });

  it("archiva con la versión conocida", async () => {
    let body: Record<string, unknown> = {};
    engine({
      "PATCH /api/v1/projects/prj_1": async (req) => {
        body = (await req.json()) as Record<string, unknown>;
        return project({ status: "archived", version: 4 });
      },
    });
    renderAt("/projects/prj_1");
    await userEvent.click(await screen.findByRole("button", { name: /Archivar/ }));
    await waitFor(() => expect(body).toEqual({ version: 3, status: "archived" }));
  });

  it("exporta el paquete .perceptron con los datos si se piden (RF-PRJ-03)", async () => {
    const pkg = vi.fn((_req: Request) => new Response("zip", { status: 200 }));
    engine({ "GET /api/v1/projects/prj_1/package": pkg });
    const created = vi.fn(() => "blob:paquete");
    vi.stubGlobal(
      "URL",
      Object.assign(URL, { createObjectURL: created, revokeObjectURL: vi.fn() }),
    );
    renderAt("/projects/prj_1");
    await userEvent.click(await screen.findByLabelText("con los datos"));
    await userEvent.click(screen.getByRole("button", { name: /Exportar .perceptron/ }));
    await waitFor(() => expect(created).toHaveBeenCalled(), { timeout: 5000 });
    const req = pkg.mock.calls[0]?.[0];
    if (!req) throw new Error("no se pidió el paquete");
    expect(new URL(req.url).searchParams.get("include_data")).toBe("true");
  });

  it("eliminar exige escribir el nombre", async () => {
    const deleted = vi.fn(() => new Response(null, { status: 204 }));
    engine({ "DELETE /api/v1/projects/prj_1": deleted });
    renderAt("/projects/prj_1");
    await userEvent.click(await screen.findByRole("button", { name: /^Eliminar$/ }));
    const confirm = await screen.findByRole("button", { name: "Eliminar definitivamente" });
    expect(confirm).toBeDisabled();
    await userEvent.type(screen.getByLabelText("Escribí «Churn» para confirmar"), "Churn");
    await userEvent.click(confirm);
    await waitFor(() => expect(deleted).toHaveBeenCalledTimes(1));
  });
});
