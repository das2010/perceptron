import { QueryClient } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "@/app/App";
import { Providers } from "@/app/providers";
import { createTestRouter } from "@/app/router";
import { useUiStore } from "@/app/store";
import { resetApiClient } from "@/lib/api/client";
import { mockEngine, project } from "@/test/engine";

function draft(
  step = "goal",
  values: Record<string, unknown> = { goal: "detectar fallas" },
  version = 1,
) {
  return {
    draft: { id: "dft_1", project_id: "prj_1", step, values, history: [], version },
    steps: [
      "goal",
      "data",
      "quality",
      "labeling",
      "task",
      "architecture",
      "hpo",
      "budget",
      "review",
    ],
    values,
  };
}

function renderAt(path: string) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <Providers client={client}>
      <App router={createTestRouter(path)} />
    </Providers>,
  );
}

class FakeSocket {
  static OPEN = 1;
  static last: FakeSocket | null = null;
  readyState = 1;
  onmessage: ((ev: MessageEvent<string>) => void) | null = null;
  sent: string[] = [];
  constructor(public url: string) {
    FakeSocket.last = this;
  }
  send(data: string) {
    this.sent.push(data);
    const emit = (msg: unknown) =>
      this.onmessage?.({ data: JSON.stringify(msg) } as MessageEvent<string>);
    emit({ type: "token", text: "Conviene " });
    emit({ type: "token", text: "optimizar el recall." });
    emit({ type: "done", text: "Conviene optimizar el recall." });
    emit({
      type: "patch",
      llm_call_id: "llc_1",
      patch: {
        changes: [
          {
            field: "target_metric",
            value: "val_recall_macro",
            rationale: "Perder fallas cuesta más.",
          },
        ],
      },
    });
  }
  close() {}
}

describe("wizard y copiloto", () => {
  beforeEach(() => resetApiClient());
  afterEach(() => {
    vi.unstubAllGlobals();
    useUiStore.setState({ copilotOpen: false });
  });

  it("guarda el objetivo y avanza de paso", async () => {
    const patches: Record<string, unknown>[] = [];
    let current = draft();
    mockEngine({
      "GET /api/v1/projects": () => [project()],
      "GET /api/v1/projects/prj_1": () => project(),
      "GET /api/v1/projects/prj_1/draft": () => current,
      "PATCH /api/v1/projects/prj_1/draft": async (req) => {
        const body = (await req.json()) as {
          step?: string;
          values?: Record<string, unknown>;
          version: number;
        };
        patches.push(body);
        current = draft(
          body.step ?? current.draft.step,
          { ...current.values, ...body.values },
          body.version + 1,
        );
        return current;
      },
      "GET /api/v1/projects/prj_1/datasets": () => [],
    });
    renderAt("/projects/prj_1/wizard");
    const goal = await screen.findByLabelText("¿Qué querés lograr?");
    await userEvent.clear(goal);
    await userEvent.type(goal, "anticipar fallas de motor");
    await userEvent.click(screen.getByRole("button", { name: /Siguiente/ }));
    await waitFor(() => expect(patches.some((p) => p.step === "data")).toBe(true));
    expect(patches[0]).toMatchObject({ values: { goal: "anticipar fallas de motor" }, version: 1 });
    expect(await screen.findByText("Agregar datos")).toBeInTheDocument();
  });

  it("el copiloto responde en streaming y su sugerencia se acepta", async () => {
    vi.stubGlobal("WebSocket", FakeSocket);
    const patches: Record<string, unknown>[] = [];
    mockEngine({
      "GET /api/v1/projects": () => [project()],
      "GET /api/v1/projects/prj_1": () => project(),
      "GET /api/v1/projects/prj_1/draft": () => draft("task"),
      "PATCH /api/v1/projects/prj_1/draft": async (req) => {
        patches.push((await req.json()) as Record<string, unknown>);
        return draft("task", { target_metric: "val_recall_macro" }, 2);
      },
    });
    useUiStore.setState({ copilotOpen: true });
    renderAt("/projects/prj_1/wizard");
    const box = await screen.findByLabelText("Preguntale al copiloto…");
    await waitFor(() => expect(FakeSocket.last?.url).toContain("/api/v1/projects/prj_1/copilot"));
    await userEvent.type(box, "¿Qué métrica uso?");
    await userEvent.click(screen.getByRole("button", { name: "Enviar" }));
    expect(await screen.findByText("Conviene optimizar el recall.")).toBeInTheDocument();
    expect(screen.getByText("Cambios sugeridos al borrador")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Aceptar" }));
    await waitFor(() =>
      expect(patches[0]).toMatchObject({
        values: { target_metric: "val_recall_macro" },
        origin: "copilot",
      }),
    );
  });

  it("agente: pide aprobación y la registra", async () => {
    let approved = false;
    const run = (state: string) => ({
      id: "agr_1",
      project_id: "prj_1",
      dataset_version_id: "dsv_1",
      pipeline_id: "pip_1",
      state,
      iterations: 0,
      steps: 1,
      trials: 0,
      cost_usd: 0.01,
      fallback: false,
      log: [{ ts: 1, kind: "approval", message: "Aprobación requerida (antes de cada iteración)" }],
      test_metrics: {},
      version: 1,
    });
    vi.stubGlobal(
      "WebSocket",
      class {
        close() {}
      },
    );
    mockEngine({
      "GET /api/v1/projects": () => [project()],
      "GET /api/v1/projects/prj_1": () => project(),
      "GET /api/v1/agent/runs/agr_1": () => run(approved ? "running" : "awaiting_approval"),
      "POST /api/v1/agent/runs/agr_1/approve": () => {
        approved = true;
        return { id: "job_1", kind: "agent", status: "queued" };
      },
    });
    renderAt("/projects/prj_1/agent?agent=agr_1");
    expect(await screen.findByText("El agente pide tu aprobación")).toBeInTheDocument();
    expect(screen.getAllByText(/antes de cada iteración/).length).toBeGreaterThan(0);
    await userEvent.click(screen.getByRole("button", { name: "Aceptar" }));
    await waitFor(() => expect(approved).toBe(true));
    expect(await screen.findByText("Trabajando")).toBeInTheDocument();
  });
});
