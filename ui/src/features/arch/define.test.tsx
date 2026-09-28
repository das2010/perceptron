import { QueryClient } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";

import { Providers } from "@/app/providers";
import { resetApiClient } from "@/lib/api/client";
import { mockEngine } from "@/test/engine";

import { DefineWizard } from "./DefineWizard";

const opt = (id: string, extra: Record<string, unknown> = {}) => ({
  id,
  title: id.toUpperCase(),
  description: `desc ${id}`,
  recommended: false,
  available: true,
  reason: null,
  ...extra,
});

function planFor(choices: Record<string, string>) {
  const family = choices.family ?? null;
  const backbones =
    family === "strong"
      ? [opt("resnet18", { recommended: true })]
      : [opt("small_cnn:32", { recommended: true })];
  const steps = [
    {
      step: "family",
      title: "Familia",
      choice: family,
      options: [
        opt("scratch", { recommended: true }),
        opt("strong", { available: false, reason: "Sin conexión" }),
      ],
    },
    { step: "backbone", title: "Backbone", choice: choices.backbone ?? null, options: backbones },
    {
      step: "head",
      title: "Cabeza",
      choice: choices.head ?? null,
      options: [opt("ce", { recommended: true }), opt("focal")],
    },
    {
      step: "regularization",
      title: "Regularización",
      choice: choices.regularization ?? null,
      options: [opt("medium", { recommended: true }), opt("high")],
    },
  ];
  return {
    modality: "image",
    task: "classification",
    steps,
    complete: steps.every((s) => s.choice),
  };
}

describe("sub-wizard de definición (§7.6)", () => {
  beforeEach(() => resetApiClient());

  it("elige paso a paso, respeta las opciones no disponibles y crea la ArchSpec", async () => {
    const built: Record<string, string>[] = [];
    let done: string | null = null;
    mockEngine({
      "POST /api/v1/projects/prj_1/arch/define": async (req) =>
        planFor(((await req.json()) as { choices: Record<string, string> }).choices),
      "POST /api/v1/projects/prj_1/arch/define/build": async (req) => {
        built.push(((await req.json()) as { choices: Record<string, string> }).choices);
        return new Response(
          JSON.stringify({
            id: "arc_9",
            project_id: "prj_1",
            name: "x",
            origin: "manual",
            content_hash: "h",
            version: 1,
          }),
          { status: 201, headers: { "Content-Type": "application/json" } },
        );
      },
    });
    render(
      <Providers client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <DefineWizard
          projectId="prj_1"
          datasetVersionId="dsv_1"
          pipelineId="pip_1"
          onBuilt={(r) => (done = r.id)}
        />
      </Providers>,
    );
    expect(await screen.findByRole("radio", { name: /STRONG/ })).toBeDisabled();
    await userEvent.click(screen.getByRole("radio", { name: /SCRATCH/ }));
    expect(await screen.findByRole("radio", { name: /SMALL_CNN:32/ })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("radio", { name: /SMALL_CNN:32/ }));
    await userEvent.click(await screen.findByRole("radio", { name: /FOCAL/ }));
    expect(screen.getByRole("button", { name: "Crear esta arquitectura" })).toBeDisabled();
    await userEvent.click(screen.getByRole("button", { name: "Completar con lo recomendado" }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Crear esta arquitectura" })).toBeEnabled(),
    );
    await userEvent.click(screen.getByRole("button", { name: "Crear esta arquitectura" }));
    await waitFor(() => expect(done).toBe("arc_9"));
    expect(built[0]).toEqual({
      family: "scratch",
      backbone: "small_cnn:32",
      head: "focal",
      regularization: "medium",
    });
  });
});
