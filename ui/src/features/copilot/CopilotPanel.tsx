/**
 * Copiloto del wizard (SPEC §7.6/§11.2): panel violeta, respuesta en streaming y cambios
 * sugeridos al borrador que el usuario acepta o rechaza (nunca se aplican solos).
 */
import { useParams } from "@tanstack/react-router";
import { Send, Sparkles } from "lucide-react";
import { useEffect, useRef, useState, type FormEvent } from "react";
import { useTranslation } from "react-i18next";

import { AiSuggestion, Button, Textarea } from "@/components/ui";
import { useDraft, useUpdateDraft } from "@/lib/api/hooks";
import { wsUrl } from "@/lib/api/ws";

interface Change {
  field: string;
  value: unknown;
  rationale: string;
}
interface ChatMessage {
  role: "user" | "assistant";
  text: string;
}
type ServerEvent =
  | { type: "token"; text: string }
  | { type: "done"; text: string }
  | { type: "patch"; patch: { changes: Change[] }; llm_call_id: string }
  | { type: "error"; code: string; message: string };

function useCopilotSocket(projectId: string | undefined) {
  const socket = useRef<WebSocket | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [patches, setPatches] = useState<Change[][]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!projectId) return;
    let closed = false;
    void wsUrl(`/api/v1/projects/${projectId}/copilot`).then((url) => {
      if (closed) return;
      const ws = new WebSocket(url);
      socket.current = ws;
      ws.onmessage = (ev: MessageEvent<string>) => {
        const msg = JSON.parse(ev.data) as ServerEvent;
        if (msg.type === "token") {
          setMessages((prev) => {
            const last = prev.at(-1);
            if (last?.role === "assistant")
              return [...prev.slice(0, -1), { ...last, text: last.text + msg.text }];
            return [...prev, { role: "assistant", text: msg.text }];
          });
        } else if (msg.type === "done") {
          setBusy(false);
        } else if (msg.type === "patch") {
          setPatches((prev) => [...prev, msg.patch.changes]);
        } else {
          setBusy(false);
          setError(msg.code === "llm_unavailable" ? "llm_unavailable" : msg.message);
        }
      };
    });
    return () => {
      closed = true;
      socket.current?.close();
      socket.current = null;
    };
  }, [projectId]);

  const send = (text: string) => {
    if (!socket.current || socket.current.readyState !== WebSocket.OPEN) return false;
    setError(null);
    setBusy(true);
    setMessages((prev) => [...prev, { role: "user", text }, { role: "assistant", text: "" }]);
    socket.current.send(JSON.stringify({ type: "message", text }));
    return true;
  };
  const dismiss = (i: number) => setPatches((prev) => prev.filter((_, j) => j !== i));
  return { messages, patches, busy, error, send, dismiss };
}

function PatchCard({ changes, onDone }: { changes: Change[]; onDone: () => void }) {
  const { t } = useTranslation();
  const projectId = (useParams({ strict: false }) as { projectId?: string }).projectId ?? "";
  const update = useUpdateDraft(projectId);
  return (
    <AiSuggestion
      title={t("copilot.patchTitle")}
      onAccept={() =>
        update.mutate(
          { values: Object.fromEntries(changes.map((c) => [c.field, c.value])), origin: "copilot" },
          { onSuccess: onDone },
        )
      }
      onReject={onDone}
    >
      <ul className="list-disc pl-5">
        {changes.map((c) => (
          <li key={c.field}>
            <strong>{t(`copilot.field.${c.field}`, { defaultValue: c.field })}</strong>:{" "}
            {String(c.value)} — {c.rationale}
          </li>
        ))}
      </ul>
    </AiSuggestion>
  );
}

export function CopilotPanel() {
  const { t } = useTranslation();
  const projectId = (useParams({ strict: false }) as { projectId?: string }).projectId;
  useDraft(projectId); // precarga la versión del borrador para aplicar sugerencias
  const { messages, patches, busy, error, send, dismiss } = useCopilotSocket(projectId);
  const [text, setText] = useState("");

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const q = text.trim();
    if (q && send(q)) setText("");
  };

  return (
    <aside
      aria-label={t("copilot.title")}
      className="flex w-96 shrink-0 flex-col border-l border-line bg-card"
    >
      <h2 className="flex items-center gap-2 border-b border-line p-4 font-semibold">
        <Sparkles className="h-4 w-4 text-copilot" aria-hidden="true" />
        {t("copilot.title")}
      </h2>
      {!projectId ? (
        <p className="p-4 text-sm text-muted">{t("copilot.openProject")}</p>
      ) : (
        <>
          <div className="flex-1 space-y-3 overflow-y-auto p-4 text-sm" aria-live="polite">
            {messages.length === 0 && <p className="text-muted">{t("copilot.intro")}</p>}
            {messages.map((m, i) => (
              <p
                key={i}
                className={
                  m.role === "user"
                    ? "ml-8 rounded-pt bg-canvas p-2"
                    : "mr-8 rounded-pt bg-copilot-bg/50 p-2"
                }
              >
                {m.text || (busy && i === messages.length - 1 ? "…" : "")}
              </p>
            ))}
            {patches.map((changes, i) => (
              <PatchCard key={i} changes={changes} onDone={() => dismiss(i)} />
            ))}
            {error && (
              <p role="alert" className="rounded-pt border border-warn p-2 text-xs">
                {error === "llm_unavailable" ? t("copilot.unavailable") : error}
              </p>
            )}
          </div>
          <form onSubmit={submit} className="flex gap-2 border-t border-line p-3">
            <Textarea
              aria-label={t("copilot.ask")}
              placeholder={t("copilot.ask")}
              rows={2}
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) submit(e);
              }}
              className="flex-1"
            />
            <Button
              type="submit"
              size="icon"
              variant="ai"
              disabled={busy || !text.trim()}
              aria-label={t("copilot.send")}
            >
              <Send className="h-4 w-4" />
            </Button>
          </form>
        </>
      )}
    </aside>
  );
}
