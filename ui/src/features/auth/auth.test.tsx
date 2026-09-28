import { QueryClient } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { App } from "@/app/App";
import { Providers } from "@/app/providers";
import { createTestRouter } from "@/app/router";
import { resetApiClient } from "@/lib/api/client";
import { health, mockEngine, project } from "@/test/engine";

import { AuthGate } from "./AuthGate";
import { atLeast, canAdminister, type Me, projectRole } from "./session";

const me = (overrides: Partial<Me> = {}): Me =>
  ({
    user: { id: "usr_1", email: "ana@preteco.test", display_name: "Ana", is_active: true },
    is_server_admin: false,
    memberships: [
      { id: "mbr_1", user_id: "usr_1", workspace_id: "wsp_1", project_id: null, role: "viewer" },
    ],
    workspaces: [{ id: "wsp_1", name: "Equipo" }],
    ...overrides,
  }) as Me;

function renderGate(path = "/") {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <Providers client={client}>
      <AuthGate>
        <App router={createTestRouter(path)} />
      </AuthGate>
    </Providers>,
  );
}

describe("sesión del Team Server (Capa 5a)", () => {
  beforeEach(() => resetApiClient());
  afterEach(() => {
    document.cookie = "pt_csrf=; max-age=0";
  });

  it("roles: el del proyecto manda sobre el del workspace", () => {
    const ana = me({
      memberships: [
        { id: "m1", user_id: "usr_1", workspace_id: "wsp_1", project_id: null, role: "viewer" },
        { id: "m2", user_id: "usr_1", workspace_id: "wsp_1", project_id: "prj_2", role: "editor" },
      ],
    });
    expect(projectRole(ana, { id: "prj_1", workspace_id: "wsp_1" })).toBe("viewer");
    expect(projectRole(ana, { id: "prj_2", workspace_id: "wsp_1" })).toBe("editor");
    expect(projectRole(ana, { id: "prj_3", workspace_id: "wsp_9" })).toBeNull();
    expect(projectRole(null, { id: "prj_1" })).toBe("admin"); // desktop
    expect(atLeast("viewer", "editor")).toBe(false);
    expect(canAdminister(ana)).toBe(false);
    expect(canAdminister(me({ is_server_admin: true }))).toBe(true);
  });

  it("sin Team Server (Engine local) no pide login", async () => {
    mockEngine({
      "GET /api/v1/system/health": () => health,
      "GET /api/v1/projects": () => [],
    });
    renderGate();
    expect(await screen.findByText("Engine operativo")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Ingresar" })).not.toBeInTheDocument();
  });

  it("pide login, manda el CSRF en las escrituras y muestra el rol", async () => {
    let session: Me | null = null;
    const csrfSeen: (string | null)[] = [];
    const { calls } = mockEngine({
      "GET /api/v1/auth/config": () => ({ mode: "server", password_login: true }),
      "GET /api/v1/auth/me": () =>
        session ??
        new Response(JSON.stringify({ code: "unauthorized", message: "no" }), { status: 401 }),
      "POST /api/v1/auth/login": async (req) => {
        const body = (await req.json()) as { password: string };
        if (body.password !== "clave-correcta-123") {
          return new Response(JSON.stringify({ code: "unauthorized", message: "x" }), {
            status: 401,
          });
        }
        document.cookie = "pt_csrf=tok123";
        session = me();
        return session;
      },
      "POST /api/v1/auth/logout": (req) => {
        csrfSeen.push(req.headers.get("X-CSRF-Token"));
        session = null;
        return new Response(null, { status: 204 });
      },
      "GET /api/v1/system/health": () => health,
      "GET /api/v1/projects": () => [project({ workspace_id: "wsp_1" })],
      "GET /api/v1/projects/prj_1": () => project({ workspace_id: "wsp_1" }),
      "GET /api/v1/projects/prj_1/datasets": () => [],
      "GET /api/v1/projects/prj_1/runs": () => [],
      "GET /api/v1/projects/prj_1/models": () => [],
    });
    const user = userEvent.setup();
    renderGate("/projects/prj_1");

    await user.type(await screen.findByLabelText("Email"), "ana@preteco.test");
    await user.type(screen.getByLabelText("Contraseña"), "mala");
    await user.click(screen.getByRole("button", { name: "Ingresar" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Email o contraseña incorrectos");

    await user.clear(screen.getByLabelText("Contraseña"));
    await user.type(screen.getByLabelText("Contraseña"), "clave-correcta-123");
    await user.click(screen.getByRole("button", { name: "Ingresar" }));
    expect(await screen.findByText("Solo lectura")).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Administración" })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Cerrar sesión" }));
    expect(await screen.findByRole("button", { name: "Ingresar" })).toBeInTheDocument();
    await waitFor(() => expect(csrfSeen).toEqual(["tok123"]));
    // El login no lleva CSRF (no hay sesión previa).
    const login = calls.find((c) => c.url.endsWith("/auth/login"));
    expect(login?.headers.get("X-CSRF-Token")).toBeNull();
  });
});
