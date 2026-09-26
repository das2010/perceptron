import createClient, { type Client } from "openapi-fetch";

import { getPlatform } from "@/lib/platform/bridge";

import type { paths } from "./schema";

export const TOKEN_HEADER = "X-Perceptron-Token";

export type ApiClient = Client<paths>;

let clientPromise: Promise<ApiClient> | null = null;

/** Cliente tipado del Engine; los tipos se generan desde OpenAPI (`pnpm gen:api`). */
export function getApiClient(): Promise<ApiClient> {
  clientPromise ??= getPlatform()
    .engine()
    .then(({ baseUrl, token }) =>
      createClient<paths>({
        baseUrl,
        headers: token ? { [TOKEN_HEADER]: token } : {},
      }),
    );
  return clientPromise;
}

export function resetApiClient(): void {
  clientPromise = null;
}
