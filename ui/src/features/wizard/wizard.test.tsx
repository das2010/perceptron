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

// Todos los pasos (como `steps` del API); «formula» y «threshold» son condicionales.
const STEPS = [
  "goal",
  "data",
  "quality",
  "formula",
  "labeling",
  "task",
  "threshold",
  "architecture",
  "hpo",
  "budget",
  "review",
];
const BASE = STEPS.filter((s) => s !== "formula" && s !== "threshold");

function draft(
  step = "goal",
  values: Record<string, unknown> = { goal: "detectar fallas" },
  version = 1,
  plan: Record<string, unknown> = {},
) {
  return {
    draft: { id: "dft_1", project_id: "prj_1", step, values, history: [], version },
    steps: STEPS,
    values,
    plan: {
      steps: BASE.map((id) => ({ id, reason: null })),
      skipped: [],
      defaults: [],
      checks: [],
      adapted: false,
      ...plan,
    },
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
  onopen: (() => void) | null = null;
  constructor(public url: string) {
    FakeSocket.last = this;
    queueMicrotask(() => this.onopen?.());
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

  it("¿Por qué? abre el copiloto y pregunta por el paso actual (RF-WIZ-02)", async () => {
    vi.stubGlobal("WebSocket", FakeSocket);
    mockEngine({
      "GET /api/v1/projects": () => [project()],
      "GET /api/v1/projects/prj_1": () => project(),
      "GET /api/v1/projects/prj_1/draft": () => draft("task"),
    });
    renderAt("/projects/prj_1/wizard");
    await userEvent.click(await screen.findByRole("button", { name: /¿Por qué?/ }));
    await waitFor(() => expect(FakeSocket.last?.sent.length).toBe(1));
    const sent = JSON.parse(FakeSocket.last?.sent[0] ?? "{}") as { text: string };
    expect(sent.text).toContain("«Tarea y métrica»");
    expect(await screen.findByText("Conviene optimizar el recall.")).toBeInTheDocument();
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

describe("wizard adaptativo (ADR-0040)", () => {
  beforeEach(() => resetApiClient());
  afterEach(() => vi.unstubAllGlobals());

  it("muestra el plan: salteados, por qué, avisos y defaults que se pueden usar", async () => {
    const plan = {
      steps: BASE.filter((s) => s !== "labeling").map((id) => ({
        id,
        reason: id === "task" ? "En la ficha: perder una falla es lo peor." : null,
      })),
      skipped: [{ id: "labeling", reason: "Ya hay etiquetas: el objetivo es «falla»." }],
      defaults: [
        {
          key: "target_metric",
          value: "val_recall_macro",
          step: "task",
          reason: "El recall mide cuántas fallas reales se detectan.",
        },
      ],
      checks: [{ code: "few_rows", severity: "info", step: "task", message: "Hay 80 filas." }],
      adapted: true,
    };
    const { calls } = mockEngine({
      "GET /api/v1/projects": () => [project()],
      "GET /api/v1/projects/prj_1": () => project(),
      "GET /api/v1/projects/prj_1/draft": () => draft("task", { task: "classification" }, 3, plan),
      "PATCH /api/v1/projects/prj_1/draft": () =>
        draft("task", { task: "classification", target_metric: "val_recall_macro" }, 4, plan),
    });
    renderAt("/projects/prj_1/wizard");
    expect(await screen.findByText(/Se saltea «Etiquetado»/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Etiquetado/ })).not.toBeInTheDocument();
    expect(screen.getByText("En la ficha: perder una falla es lo peor.")).toBeInTheDocument();
    expect(screen.getByText("Hay 80 filas.")).toBeInTheDocument();
    expect(
      screen.getByText("El recall mide cuántas fallas reales se detectan."),
    ).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Usar" }));
    await waitFor(() => expect(calls.some((r) => r.method === "PATCH")).toBe(true));
    const patch = calls.find((r) => r.method === "PATCH");
    expect(await patch?.clone().json()).toMatchObject({
      values: { target_metric: "val_recall_macro" },
    });
  });

  it("la entrevista propone cambios a la ficha y se aplican solo al aceptarlos", async () => {
    const { calls } = mockEngine({
      "GET /api/v1/projects": () => [project()],
      "GET /api/v1/projects/prj_1": () => project(),
      "GET /api/v1/projects/prj_1/draft": () => draft("goal", { goal: "rodamientos" }, 1),
      "PATCH /api/v1/projects/prj_1/draft": () => draft("goal", { goal: "rodamientos" }, 2),
      "POST /api/v1/projects/prj_1/draft/intake": () => ({
        llm_call_id: "llc_9",
        patch: {
          changes: [
            { field: "problem", value: "category", rationale: "Quiere detectar fallas." },
            { field: "extrapolate", value: false, rationale: "Mismos equipos." },
          ],
          assumptions: [],
          next_question: "¿Los datos vienen de varias máquinas?",
        },
      }),
    });
    const user = userEvent.setup();
    renderAt("/projects/prj_1/wizard");
    await user.type(await screen.findByLabelText("Tu mensaje"), "Quiero anticipar fallas");
    await user.click(screen.getByRole("button", { name: "Enviar" }));
    expect(await screen.findByLabelText("Pregunta del asistente")).toHaveTextContent(
      "¿Los datos vienen de varias máquinas?",
    );
    const proposed = screen.getByText("El asistente propone:").closest("div");
    expect(proposed).toHaveTextContent("Tipo de problema: Predecir una categoría");
    expect(calls.some((r) => r.method === "PATCH")).toBe(false); // nada sin aceptar
    await user.click(screen.getByRole("button", { name: "Aceptar cambios" }));
    await waitFor(() => expect(calls.some((r) => r.method === "PATCH")).toBe(true));
    const body = await calls
      .find((r) => r.method === "PATCH")
      ?.clone()
      .json();
    expect(body.origin).toBe("copilot");
    expect(body.values.brief).toMatchObject({ problem: "category", extrapolate: false });
  });
});

describe("wizard adaptativo, fase 2", () => {
  beforeEach(() => resetApiClient());
  afterEach(() => vi.unstubAllGlobals());

  const dataset = {
    id: "dsv_1",
    project_id: "prj_1",
    content_hash: "abcdef0123456789",
    num_samples: 108,
    size_bytes: 1,
    modality: "tabular",
    target: "Salida",
    version: 1,
    created_at: "2026-10-04T10:00:00Z",
  };
  const planWith = (extra: string[]) => ({
    steps: [
      "goal",
      "data",
      "quality",
      ...extra,
      "task",
      "architecture",
      "hpo",
      "budget",
      "review",
    ].map((id) => ({
      id,
      reason: id === "formula" ? "En la ficha: el objetivo es descubrir una regla." : null,
    })),
    adapted: true,
  });

  it("muestra los cambios del plan y el paso Fórmula con la búsqueda", async () => {
    const values = {
      dataset_version_id: "dsv_1",
      brief: { problem: "rule" },
      plan_diff: {
        added_steps: ["formula"],
        removed_steps: [],
        changed_defaults: ["architecture_hint"],
        new_checks: [],
        resolved_checks: [],
      },
    };
    mockEngine({
      "GET /api/v1/projects": () => [project()],
      "GET /api/v1/projects/prj_1": () => project(),
      "GET /api/v1/projects/prj_1/draft": () => draft("formula", values, 2, planWith(["formula"])),
      "GET /api/v1/projects/prj_1/datasets": () => [dataset],
      "GET /api/v1/projects/prj_1/symbolic": () => [],
    });
    const user = userEvent.setup();
    renderAt("/projects/prj_1/wizard");
    const banner = await screen.findByRole("status", { name: "El plan cambió" });
    expect(banner).toHaveTextContent("Nuevo paso: Fórmula sugerida");
    expect(banner).toHaveTextContent("Cambió la sugerencia de arquitectura");
    expect(
      screen.getByText("En la ficha: el objetivo es descubrir una regla."),
    ).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: "Buscar fórmula" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Entendido" }));
    expect(screen.queryByRole("status", { name: "El plan cambió" })).not.toBeInTheDocument();
  });

  it("la reconciliación propone cambios desde Calidad y se aceptan", async () => {
    const values = {
      dataset_version_id: "dsv_1",
      brief: { problem: "rule", independent_inputs: true },
    };
    const { calls } = mockEngine({
      "GET /api/v1/projects": () => [project()],
      "GET /api/v1/projects/prj_1": () => project(),
      "GET /api/v1/projects/prj_1/draft": () => draft("quality", values, 3, planWith([])),
      "PATCH /api/v1/projects/prj_1/draft": () => draft("quality", values, 4, planWith([])),
      "GET /api/v1/datasets/dsv_1/profile": () => new Response("{}", { status: 404 }),
      "POST /api/v1/projects/prj_1/draft/reconcile": () => ({
        llm_call_id: "llc_2",
        patch: {
          changes: [
            {
              field: "independent_inputs",
              value: false,
              rationale: "S2 es la raíz de S1: varían juntas.",
            },
          ],
          assumptions: [],
          next_question: null,
        },
      }),
    });
    const user = userEvent.setup();
    renderAt("/projects/prj_1/wizard");
    await user.click(await screen.findByRole("button", { name: "Revisar la ficha" }));
    expect(await screen.findByText("— S2 es la raíz de S1: varían juntas.")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Aceptar cambios" }));
    await waitFor(() => expect(calls.some((r) => r.method === "PATCH")).toBe(true));
    const body = await calls
      .find((r) => r.method === "PATCH")
      ?.clone()
      .json();
    expect(body.origin).toBe("copilot");
    expect(body.values.brief).toMatchObject({ independent_inputs: false });
  });

  it("el paso Umbral pide cuántas veces peor es el error", async () => {
    const values = { brief: { problem: "category", error_costs: "false_negative_worse" } };
    const { calls } = mockEngine({
      "GET /api/v1/projects": () => [project()],
      "GET /api/v1/projects/prj_1": () => project(),
      "GET /api/v1/projects/prj_1/draft": () =>
        draft("threshold", values, 5, planWith(["threshold"])),
      "PATCH /api/v1/projects/prj_1/draft": () =>
        draft("threshold", values, 6, planWith(["threshold"])),
    });
    const user = userEvent.setup();
    renderAt("/projects/prj_1/wizard");
    expect(await screen.findByText(/no se puede calcular el umbral por costo/)).toBeInTheDocument();
    const ratio = screen.getByLabelText("¿Cuántas veces peor?");
    await user.type(ratio, "10");
    await user.tab();
    await waitFor(() => expect(calls.some((r) => r.method === "PATCH")).toBe(true));
    const body = await calls
      .find((r) => r.method === "PATCH")
      ?.clone()
      .json();
    expect(body.values.brief).toMatchObject({
      error_costs: "false_negative_worse",
      error_cost_ratio: 10,
    });
  });

  it("diseño guiado: corre el job, muestra la ganadora con evidencia y se acepta", async () => {
    const design = {
      pipeline_id: "pip_1",
      origin: "llm",
      requirements: {
        items: [
          {
            code: "pretrained_backbone",
            level: "must",
            scope: "any",
            message: "Con 300 imágenes, una red preentrenada generaliza mejor.",
          },
        ],
      },
      candidates: [
        {
          archspec_id: "arc_m",
          title: "MobileNetV3 preentrenada",
          rationale: "r",
          origin: "llm",
          score: 97,
          recommended: true,
          eligible: true,
          checks: [{ code: "pretrained_backbone", level: "must", met: true }],
          tournament_metric: 0.91,
          tournament_status: "complete",
        },
        {
          archspec_id: "arc_e",
          title: "EfficientNet-B0 preentrenada",
          rationale: "r",
          origin: "llm",
          score: 95,
          recommended: false,
          eligible: true,
          checks: [{ code: "pretrained_backbone", level: "must", met: true }],
          tournament_metric: 0.84,
          tournament_status: "complete",
        },
      ],
      tournament: { metric: "val_f1_macro", direction: "maximize", subset: 1, winner: "arc_m" },
      pick: "arc_m",
      pick_reason: "Ganó el entrenamiento corto de comparación con val_f1_macro = 0.91.",
      max_epochs_per_trial: 20,
      max_trials: 12,
      strategy: { strategy: "tpe", pruner: "median" },
    };
    let values: Record<string, unknown> = { dataset_version_id: "dsv_1" };
    let jobPolls = 0;
    const patches: Record<string, unknown>[] = [];
    mockEngine({
      "GET /api/v1/projects": () => [project()],
      "GET /api/v1/projects/prj_1": () => project(),
      "GET /api/v1/projects/prj_1/draft": () => draft("architecture", values, 3),
      "POST /api/v1/projects/prj_1/draft/design": () => ({
        id: "job_1",
        kind: "design",
        status: "queued",
      }),
      "GET /api/v1/jobs/job_1": () => {
        jobPolls += 1;
        if (jobPolls < 2)
          return {
            id: "job_1",
            kind: "design",
            status: "running",
            progress: { stage: "tournament" },
          };
        values = { ...values, design };
        return { id: "job_1", kind: "design", status: "succeeded" };
      },
      "PATCH /api/v1/projects/prj_1/draft": async (req) => {
        const body = (await req.json()) as { values?: Record<string, unknown> };
        patches.push(body.values ?? {});
        values = { ...values, ...body.values };
        return draft("architecture", values, 4);
      },
    });
    const user = userEvent.setup();
    renderAt("/projects/prj_1/wizard");
    await user.click(await screen.findByRole("button", { name: "Diseñar automáticamente" }));
    expect(
      await screen.findByText("Comparando las mejores con un entrenamiento corto…"),
    ).toBeInTheDocument();
    expect(
      await screen.findByText("Ganó la comparación", {}, { timeout: 5000 }),
    ).toBeInTheDocument();
    expect(screen.getByText(/val_f1_macro = 0.91/)).toBeInTheDocument();
    expect(screen.getByText(/tpe, 12 intentos de hasta 20 épocas/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Aceptar el diseño" }));
    await waitFor(() => expect(patches).toHaveLength(1));
    expect(patches[0]).toMatchObject({
      archspec_id: "arc_m",
      strategy: { strategy: "tpe" },
      max_epochs_per_trial: 20,
      max_trials: 12,
    });
    expect(await screen.findByText("Diseño aceptado")).toBeInTheDocument();
  });

  it("la revisión explica por qué esta arquitectura (memo del diseño)", async () => {
    const values = {
      dataset_version_id: "dsv_1",
      pipeline_id: "pip_1",
      archspec_id: "arc_e",
      design: {
        pipeline_id: "pip_1",
        origin: "llm",
        candidates: [
          {
            archspec_id: "arc_m",
            title: "MobileNetV3 preentrenada",
            rationale: "r",
            origin: "llm",
            checks: [{ code: "pretrained_backbone", level: "must", met: true }],
          },
          {
            archspec_id: "arc_e",
            title: "CNN desde cero",
            rationale: "r",
            origin: "llm",
            checks: [{ code: "pretrained_backbone", level: "must", met: false }],
          },
        ],
        pick: "arc_m",
        pick_reason: "Ganó la comparación.",
      },
    };
    mockEngine({
      "GET /api/v1/projects": () => [project()],
      "GET /api/v1/projects/prj_1": () => project(),
      "GET /api/v1/projects/prj_1/draft": () => draft("review", values, 7),
    });
    renderAt("/projects/prj_1/wizard");
    const memo = await screen.findByRole("region", { name: "Por qué esta arquitectura" });
    expect(memo).toHaveTextContent("CNN desde cero");
    expect(memo).toHaveTextContent("el diseño guiado proponía otra");
    expect(memo).toHaveTextContent("No cumple: Red preentrenada.");
  });

  it("«Aceptar y revisar» fija el diseño y pasa a la revisión en un clic", async () => {
    const design = {
      pipeline_id: "pip_1",
      origin: "rules",
      candidates: [
        {
          archspec_id: "arc_l",
          title: "Regresión lineal",
          rationale: "r",
          origin: "rules",
          score: 97,
          recommended: true,
          eligible: true,
          checks: [{ code: "linear_option", level: "must", met: true }],
        },
      ],
      pick: "arc_l",
      pick_reason: "Es la que mejor cumple los requisitos de diseño del escenario.",
      max_epochs_per_trial: 80,
      strategy: { strategy: "tpe" },
    };
    let current = draft("architecture", { dataset_version_id: "dsv_1", design }, 2);
    const patches: Record<string, unknown>[] = [];
    mockEngine({
      "GET /api/v1/projects": () => [project()],
      "GET /api/v1/projects/prj_1": () => project(),
      "GET /api/v1/projects/prj_1/draft": () => current,
      "PATCH /api/v1/projects/prj_1/draft": async (req) => {
        const body = (await req.json()) as { step?: string; values?: Record<string, unknown> };
        patches.push(body);
        current = draft(body.step ?? "architecture", { ...current.values, ...body.values }, 3);
        return current;
      },
    });
    const user = userEvent.setup();
    renderAt("/projects/prj_1/wizard");
    await user.click(await screen.findByRole("button", { name: "Aceptar y revisar" }));
    await waitFor(() => expect(patches).toHaveLength(1));
    expect(patches[0]).toMatchObject({
      step: "review",
      values: { archspec_id: "arc_l", max_epochs_per_trial: 80, strategy: { strategy: "tpe" } },
    });
    const memo = await screen.findByRole("region", { name: "Por qué esta arquitectura" });
    expect(memo).toHaveTextContent("Regresión lineal");
    expect(memo).toHaveTextContent("Cumple: Opción lineal.");
  });
});
