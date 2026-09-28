import { QueryClient } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { Providers } from "@/app/providers";
import { resetApiClient } from "@/lib/api/client";
import type { Project } from "@/lib/api/hooks";
import { setPlatform, WebPlatformBridge } from "@/lib/platform/bridge";
import { mockEngine, project } from "@/test/engine";

import { TeamSync } from "./TeamSync";

class DesktopBridge extends WebPlatformBridge {
  // @ts-expect-error: el test simula el desktop sobre el bridge web
  override readonly kind = "desktop" as const;
}

describe("sincronización del proyecto de equipo (RF-SRV-03)", () => {
  beforeEach(() => {
    resetApiClient();
    setPlatform(new DesktopBridge());
  });
  afterEach(() => setPlatform(new WebPlatformBridge()));

  it("muestra cambios y conflictos y baja con la resolución elegida", async () => {
    const pulls: unknown[] = [];
    mockEngine({
      "GET /api/v1/remote/servers": () => [{ name: "equipo", url: "https://srv", email: "a@b" }],
      "GET /api/v1/projects/prj_1/remote/status": () => ({
        server: "equipo",
        items: [
          { kind: "Project", id: "prj_1", name: "Churn", state: "synced" },
          { kind: "Pipeline", id: "pip_1", name: "limpieza", state: "conflict" },
          { kind: "ArchSpecRecord", id: "arc_1", name: "mlp", state: "pull" },
        ],
      }),
      "POST /api/v1/projects/prj_1/remote/pull": async (req) => {
        pulls.push(await req.json());
        return { id: "job_1", kind: "sync_pull", status: "running" };
      },
      "GET /api/v1/jobs/job_1": () => ({
        id: "job_1",
        kind: "sync_pull",
        status: "succeeded",
        result: { pulled: 1, pushed: 1, conflicts_left: [], downloaded: 0 },
      }),
    });
    render(
      <Providers client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <TeamSync project={{ ...project(), scope: "team" } as Project} />
      </Providers>,
    );
    expect(await screen.findByText(/1 conflicto/)).toBeInTheDocument();
    expect(screen.getByText("Cambió en el servidor")).toBeInTheDocument();
    // Con conflictos no se sube nada.
    expect(screen.getByRole("button", { name: /Subir cambios/ })).toBeDisabled();
    await userEvent.selectOptions(screen.getByLabelText("Resolver limpieza"), "mine");
    await userEvent.click(screen.getByRole("button", { name: /Bajar cambios/ }));
    await waitFor(() =>
      expect(pulls).toEqual([{ server: "equipo", resolutions: { pip_1: "mine" } }]),
    );
    expect(await screen.findByText("Bajados: 1 · subidos: 1.")).toBeInTheDocument();
  });
});
