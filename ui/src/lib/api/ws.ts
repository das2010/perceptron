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

/** Mensajes JSON de un WebSocket del Engine (progreso de jobs y runs, RF-TRN-06). */
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
    void wsUrl(path).then((url) => {
      if (closed) return;
      socket = new WebSocket(url);
      socket.onopen = () => setOpen(true);
      socket.onclose = () => setOpen(false);
      socket.onmessage = (ev: MessageEvent<string>) => {
        try {
          handler.current(JSON.parse(ev.data) as T);
        } catch {
          /* mensaje no JSON: se ignora */
        }
      };
    });
    return () => {
      closed = true;
      socket?.close();
    };
  }, [path]);

  return open;
}
