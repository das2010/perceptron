import { vi } from "vitest";

type Handler = (req: Request) => unknown | Promise<unknown>;

/** Engine falso por ruta: `{"GET /api/v1/projects": () => [...]}`; lo demás responde 404. */
export function mockEngine(routes: Record<string, Handler>) {
  const calls: Request[] = [];
  const fn = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const req = input instanceof Request ? input : new Request(input, init);
    calls.push(req);
    const { pathname } = new URL(req.url);
    const handler = routes[`${req.method} ${pathname}`];
    if (!handler) {
      return new Response(JSON.stringify({ code: "not_found", message: pathname }), {
        status: 404,
        headers: { "Content-Type": "application/json" },
      });
    }
    const body = await handler(req);
    if (body instanceof Response) return body;
    if (body instanceof Error) throw body;
    return new Response(JSON.stringify(body), { headers: { "Content-Type": "application/json" } });
  });
  vi.stubGlobal("fetch", fn);
  return { fn, calls };
}

export const health = { status: "ok", version: "0.1.0" };
export const hardware = {
  os: "Linux",
  python: "3.12",
  cpu: { model: "cpu", physical_cores: 4, logical_cores: 8, arch: "x86_64", flags: [] },
  ram_total_gb: 16,
  ram_available_gb: 8,
  disk_free_gb: 100,
  gpus: [],
  torch: { version: "2.14", cuda: null, rocm: null, xpu: false },
  recommended_device: "cpu",
  recommended_torch_variant: "cpu",
  notes: [],
};

export function project(overrides: Record<string, unknown> = {}) {
  return {
    id: "prj_1",
    name: "Churn",
    description: "",
    goal: "Anticipar bajas",
    modalities: ["tabular"],
    privacy_level: "L1",
    scope: "local",
    status: "active",
    version: 1,
    created_at: "2026-09-27T10:00:00Z",
    updated_at: "2026-09-27T10:00:00Z",
    ...overrides,
  };
}
