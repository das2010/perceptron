import { useEffect, useRef, useState } from "react";

import { getPlatform } from "@/lib/platform/bridge";

/** URL WebSocket del Engine con el token por query (el navegador no permite headers en WS). */
export async function wsUrl(path: string): Promise<string> {
  const { baseUrl, token } = await getPlatform().engine();
  const url = new URL(path, baseUrl);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  if (token) url.searchParams.set("token", token);
  return url.toString();
}

/** Espera antes de reconectar: 1 s, 2 s, 4 s… hasta 15 s. */
export const reconnectDelay = (attempt: number) => Math.min(15_000, 1_000 * 2 ** attempt);

/** Mensajes JSON de un WebSocket del Engine (progreso de jobs y runs, RF-TRN-06).
 *
 * Si la conexión se corta (red, reinicio del servidor), se reconecta sola con espera
 * creciente; el servidor reenvía el historial reciente del job, así que no se pierden eventos.
 * Un cierre normal (1000: el stream terminó) o desmontar el componente no reconectan. */
export function useEngineSocket<T>(path: string | null, onMessage: (msg: T) => void): boolean {
  const [open, setOpen] = useState(false);
  const handler = useRef(onMessage);
  useEffect(() => {
    handler.current = onMessage;
  }, [onMessage]);

  useEffect(() => {
    if (!path) return;
    let socket: WebSocket | null = null;
    let closed = false;
    let attempt = 0;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      void wsUrl(path).then((url) => {
        if (closed) return;
        socket = new WebSocket(url);
        socket.onopen = () => {
          attempt = 0;
          setOpen(true);
        };
        socket.onclose = (ev: CloseEvent) => {
          setOpen(false);
          if (closed || ev.code === 1000) return;
          timer = setTimeout(connect, reconnectDelay(attempt));
          attempt += 1;
        };
        socket.onmessage = (ev: MessageEvent<string>) => {
          try {
            handler.current(JSON.parse(ev.data) as T);
          } catch {
            /* mensaje no JSON: se ignora */
          }
        };
      });
    };
    connect();
    return () => {
      closed = true;
      if (timer) clearTimeout(timer);
      socket?.close();
    };
  }, [path]);

  return open;
}
