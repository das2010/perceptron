import { QueryClient } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";

import { App } from "@/app/App";
import { Providers } from "@/app/providers";
import { createTestRouter } from "@/app/router";
import { type Me, SessionContext } from "@/features/auth/session";
import { resetApiClient } from "@/lib/api/client";
import { mockEngine } from "@/test/engine";

const me = {
  user: { id: "usr_1", email: "ana@preteco.test", display_name: "Ana", is_active: true },
  is_server_admin: false,
  memberships: [],
  workspaces: [],
} as unknown as Me;

describe("cola del Team Server (Capa 5b)", () => {
  beforeEach(() => resetApiClient());

  it("muestra workers con su hardware y los estudios en curso", async () => {
    mockEngine({
      "GET /api/v1/projects": () => [],
      "GET /api/v1/server/queue": () => ({
        mode: "queue",
        quota_per_user: 2,
        quota_per_workspace: 6,
        workers: [
          {
            worker: "gpu-1",
            queues: ["gpu"],
            busy_job: "job_1",
            gpus: [{ name: "NVIDIA L4" }],
            seen_s_ago: 1,
          },
          { worker: "cpu-1", queues: ["cpu"], busy_job: null, gpus: [], seen_s_ago: 2 },
        ],
        jobs: [
          {
            id: "job_1",
            kind: "study",
            status: "running",
            runner: "queue:gpu",
            worker: "gpu-1",
            created_at: "2026-09-28T10:00:00Z",
            refs: { study_id: "std_1", project_id: "prj_1", queue: "gpu" },
          },
        ],
      }),
    });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <Providers client={client}>
        <SessionContext.Provider value={me}>
          <App router={createTestRouter("/queue")} />
        </SessionContext.Provider>
      </Providers>,
    );
    expect(await screen.findByText("NVIDIA L4")).toBeInTheDocument();
    expect(screen.getByText("Solo CPU")).toBeInTheDocument();
    expect(screen.getByText("Ocupado")).toBeInTheDocument();
    expect(screen.getByText("Libre")).toBeInTheDocument();
    expect(screen.getByText("Entrenando")).toBeInTheDocument();
    expect(screen.getByText(/2 por persona y 6 por workspace/)).toBeInTheDocument();
  });
});
