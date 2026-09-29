/** Subidas con progreso: `fetch` no informa el avance del cuerpo, `XMLHttpRequest` sí. */
import { getPlatform } from "@/lib/platform/bridge";

import { CSRF_HEADER, TOKEN_HEADER, UNAUTHORIZED_EVENT, csrfToken } from "./client";
import { ApiError } from "./hooks";

/** POST multipart al Engine; `onProgress` recibe la fracción enviada (0–1). */
export async function postForm<T>(
  path: string,
  form: FormData,
  onProgress?: (fraction: number) => void,
): Promise<T> {
  const { baseUrl, token } = await getPlatform().engine();
  return new Promise<T>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", new URL(path, baseUrl).toString());
    xhr.withCredentials = true; // cookie de sesión del Team Server
    if (token) xhr.setRequestHeader(TOKEN_HEADER, token);
    const csrf = csrfToken();
    if (csrf) xhr.setRequestHeader(CSRF_HEADER, csrf);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && e.total > 0) onProgress?.(e.loaded / e.total);
    };
    xhr.onload = () => {
      let body: unknown;
      try {
        body = xhr.responseText ? JSON.parse(xhr.responseText) : null;
      } catch {
        body = null;
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(body as T);
        return;
      }
      if (xhr.status === 401) window.dispatchEvent(new Event(UNAUTHORIZED_EVENT));
      const err = (body ?? {}) as {
        code?: string;
        message?: string;
        details?: Record<string, unknown>;
      };
      reject(
        new ApiError(
          err.message ?? `HTTP ${xhr.status}`,
          err.code ?? "http_error",
          xhr.status,
          err.details ?? {},
        ),
      );
    };
    xhr.onerror = () =>
      reject(new ApiError("no se pudo conectar con el Engine", "network_error", 0, {}));
    xhr.send(form);
  });
}
