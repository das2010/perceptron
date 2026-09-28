import { QueryClient } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Providers } from "@/app/providers";
import { resetApiClient } from "@/lib/api/client";
import { mockEngine } from "@/test/engine";

import { SystemTab } from "./SystemTab";

describe("consola: sistema (RF-SRV-06)", () => {
  beforeEach(() => resetApiClient());
  afterEach(() => vi.unstubAllGlobals());

  it("muestra almacenamiento por proyecto, cuotas y workers", async () => {
    mockEngine({
      "GET /api/v1/admin/system": () => ({
        workspace_path: "/data/workspace",
        used_bytes: 3 * 1024 ** 3,
        free_bytes: 40 * 1024 ** 3,
        projects: [{ project_id: "prj_1", name: "Churn", bytes: 250 * 1024 ** 2 }],
        quotas: { max_running_studies_per_user: 2, max_running_studies_per_workspace: 6 },
        queue_mode: "queue",
        workers: [
          { worker: "gpu-1", queues: ["gpu"], gpus: [{}], busy_job: "job_1", seen_s_ago: 2 },
        ],
      }),
    });
    render(
      <Providers client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <SystemTab />
      </Providers>,
    );
    expect(await screen.findByText("Churn")).toBeInTheDocument();
    expect(screen.getByText("250.0 MB")).toBeInTheDocument();
    expect(screen.getByText("Estudios en curso por usuario")).toBeInTheDocument();
    expect(screen.getByText("gpu-1")).toBeInTheDocument();
    expect(screen.getByText("ocupado")).toBeInTheDocument();
  });
});
