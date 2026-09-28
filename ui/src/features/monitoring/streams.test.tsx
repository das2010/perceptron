import { QueryClient } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";

import { App } from "@/app/App";
import { Providers } from "@/app/providers";
import { createTestRouter } from "@/app/router";
import { resetApiClient } from "@/lib/api/client";
import { mockEngine, project } from "@/test/engine";

describe("fuentes streaming (RF-ING-05)", () => {
  beforeEach(() => resetApiClient());

  it("crea una fuente Kafka con SASL y lee el buffer a mano", async () => {
    const created: unknown[] = [];
    const pulls: string[] = [];
    const sources = [
      {
        source: {
          id: "src_1",
          project_id: "prj_1",
          name: "sensores",
          type: "stream",
          config: { stream: { kind: "mqtt" }, last_poll: "2026-09-28T10:00:00Z" },
        },
        buffer: { added: 0, batches: 3, rows: 120, state: {} },
      },
    ];
    mockEngine({
      "GET /api/v1/projects": () => [project()],
      "GET /api/v1/projects/prj_1": () => project(),
      "GET /api/v1/projects/prj_1/deployments": () => [],
      "GET /api/v1/projects/prj_1/alerts": () => [],
      "GET /api/v1/projects/prj_1/sources/stream": () => sources,
      "POST /api/v1/projects/prj_1/sources/stream": async (req) => {
        created.push(await req.json());
        return { id: "src_2", project_id: "prj_1", name: "pedidos", type: "stream", config: {} };
      },
      "POST /api/v1/sources/src_1/pull": () => {
        pulls.push("src_1");
        return { added: 10, batches: 4, rows: 130, state: {} };
      },
    });
    render(
      <Providers client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <App router={createTestRouter("/projects/prj_1/monitoring")} />
      </Providers>,
    );
    expect(await screen.findByText("120 filas en 3 lotes")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /Leer ahora/ }));
    await waitFor(() => expect(pulls).toEqual(["src_1"]));

    await userEvent.click(screen.getByRole("button", { name: "Agregar fuente" }));
    await userEvent.selectOptions(screen.getByLabelText("Tipo"), "kafka");
    await userEvent.type(screen.getByLabelText("Nombre"), "pedidos");
    await userEvent.type(screen.getByLabelText(/Brokers/), "kafka-1:9092, kafka-2:9092");
    await userEvent.type(screen.getByLabelText(/Tópico/), "pedidos");
    await userEvent.selectOptions(screen.getByLabelText("Autenticación"), "sasl_scram_512");
    await userEvent.type(screen.getByLabelText("Usuario"), "lector");
    await userEvent.type(screen.getByLabelText(/Token o contraseña/), "secreto");
    await userEvent.click(screen.getByLabelText("Conexión cifrada (TLS)"));
    await userEvent.click(screen.getByRole("button", { name: "Crear fuente" }));
    await waitFor(() => expect(created).toHaveLength(1));
    expect(created[0]).toEqual({
      name: "pedidos",
      kind: "kafka",
      config: {
        bootstrap_servers: ["kafka-1:9092", "kafka-2:9092"],
        topic: "pedidos",
        group_id: "perceptron",
        auth: "sasl_scram_512",
        username: "lector",
        ssl: true,
      },
      token: "secreto",
      poll_interval_s: null,
    });
  });
});
