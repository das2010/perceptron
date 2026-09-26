import { QueryClient } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { resetApiClient } from "@/lib/api/client";

import { App } from "./App";
import { Providers } from "./providers";

function renderApp() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <Providers client={client}>
      <App />
    </Providers>,
  );
}

function mockFetch(response: Response | Error) {
  const fn = vi.fn((_input: RequestInfo | URL) =>
    response instanceof Error ? Promise.reject(response) : Promise.resolve(response),
  );
  vi.stubGlobal("fetch", fn);
  return fn;
}

describe("App", () => {
  beforeEach(() => resetApiClient());
  afterEach(() => {
    vi.unstubAllGlobals();
    delete document.documentElement.dataset.theme;
    localStorage.clear();
  });

  it("muestra el estado del Engine cuando responde", async () => {
    const fetchMock = mockFetch(
      new Response(JSON.stringify({ status: "ok", version: "0.1.0" }), {
        headers: { "Content-Type": "application/json" },
      }),
    );
    renderApp();
    expect(screen.getByText("Conectando con el Engine…")).toBeInTheDocument();
    expect(await screen.findByText("Engine operativo")).toBeInTheDocument();
    expect(screen.getByText("Versión 0.1.0")).toBeInTheDocument();
    const request = fetchMock.mock.calls[0]?.[0] as Request;
    expect(new URL(request.url).pathname).toBe("/api/v1/system/health");
  });

  it("muestra error y permite reintentar", async () => {
    mockFetch(new TypeError("network"));
    renderApp();
    expect(await screen.findByText("No se pudo conectar con el Engine")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Reintentar" })).toBeInTheDocument();
  });

  it("cambia de idioma a inglés", async () => {
    mockFetch(new TypeError("network"));
    renderApp();
    await userEvent.selectOptions(screen.getByLabelText("Idioma"), "en");
    expect(
      await screen.findByText("Train neural networks locally, guided by AI"),
    ).toBeInTheDocument();
  });

  it("aplica y recuerda el tema elegido", async () => {
    mockFetch(new TypeError("network"));
    renderApp();
    await userEvent.selectOptions(screen.getByLabelText("Tema"), "dark");
    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(localStorage.getItem("perceptron.theme")).toBe("dark");
  });
});
