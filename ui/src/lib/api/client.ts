import createClient, { type Client, type Middleware } from "openapi-fetch";

import { getPlatform } from "@/lib/platform/bridge";

import type { paths } from "./schema";

export const TOKEN_HEADER = "X-Perceptron-Token";
export const CSRF_HEADER = "X-CSRF-Token";
/** Evento global: el Team Server respondió 401 (sesión vencida o cerrada en otro lado). */
export const UNAUTHORIZED_EVENT = "perceptron:unauthorized";

export type ApiClient = Client<paths>;

const SAFE_METHODS = new Set(["GET", "HEAD", "OPTIONS"]);

/** Cookie `pt_csrf` del Team Server (double submit, RF-SRV-01); no existe en el desktop. */
export function csrfToken(): string | null {
  if (typeof document === "undefined") return null;
  const match = /(?:^|;\s*)pt_csrf=([^;]+)/.exec(document.cookie);
  return match?.[1] ? decodeURIComponent(match[1]) : null;
}

const sessionMiddleware: Middleware = {
  onRequest({ request }) {
    const csrf = csrfToken();
    if (csrf && !SAFE_METHODS.has(request.method)) request.headers.set(CSRF_HEADER, csrf);
    return request;
  },
  onResponse({ request, response }) {
    if (response.status === 401 && !new URL(request.url).pathname.includes("/auth/")) {
      window.dispatchEvent(new Event(UNAUTHORIZED_EVENT));
    }
    return response;
  },
};

let clientPromise: Promise<ApiClient> | null = null;

/** Cliente tipado del Engine; los tipos se generan desde OpenAPI (`pnpm gen:api`). */
export function getApiClient(): Promise<ApiClient> {
  clientPromise ??= getPlatform()
    .engine()
    .then(({ baseUrl, token }) => {
      const client = createClient<paths>({
        baseUrl,
        headers: token ? { [TOKEN_HEADER]: token } : {},
      });
      client.use(sessionMiddleware);
      return client;
    });
  return clientPromise;
}

export function resetApiClient(): void {
  clientPromise = null;
}
