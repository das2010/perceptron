/**
 * El hilo de la conversación con el asistente: burbujas, respuestas rápidas, el «escribiendo…»
 * y la caja de texto, siempre disponible en todas las etapas.
 */
import { ArrowRight, Paperclip, SendHorizontal, Sparkles, Upload } from "lucide-react";
import { type KeyboardEvent, type ReactNode, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui";
import type { Schemas } from "@/lib/api/hooks";

import type { QuickReply } from "./journeyText";

type Turn = Schemas["Turn"];

export function Bubble({
  turn,
  children,
}: {
  turn: Pick<Turn, "role" | "text">;
  children?: ReactNode;
}) {
  const { t } = useTranslation();
  const mine = turn.role === "user";
  return (
    <div className={`flex ${mine ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-2 ${
          mine
            ? "rounded-br-sm bg-primary text-on-primary"
            : "rounded-bl-sm border border-line bg-card"
        }`}
      >
        <p className="sr-only">{t(mine ? "journey.user" : "journey.assistant")}:</p>
        <p className="whitespace-pre-wrap">{turn.text}</p>
        {children}
      </div>
    </div>
  );
}

export function Typing({ label }: { label?: string }) {
  const { t } = useTranslation();
  return (
    <div className="flex justify-start" role="status" aria-label={label ?? t("journey.typing")}>
      <div className="flex items-center gap-1 rounded-2xl rounded-bl-sm border border-line bg-card px-4 py-3">
        {label && <span className="mr-2 text-sm text-muted">{label}</span>}
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            aria-hidden="true"
            className="h-2 w-2 animate-bounce rounded-full bg-muted"
            style={{ animationDelay: `${i * 150}ms` }}
          />
        ))}
      </div>
    </div>
  );
}

export function Chips({
  replies,
  disabled,
  onPick,
}: {
  replies: QuickReply[];
  disabled: boolean;
  onPick: (r: QuickReply) => void;
}) {
  if (!replies.length) return null;
  return (
    <div className="flex flex-wrap justify-end gap-2">
      {replies.map((r) => (
        <button
          key={r.label}
          type="button"
          disabled={disabled}
          onClick={() => onPick(r)}
          className="rounded-full border border-brand px-3 py-1 text-sm text-brand transition hover:bg-brand hover:text-white disabled:opacity-50"
        >
          {r.label}
        </button>
      ))}
    </div>
  );
}

type Goal = Schemas["GoalProposal"];

/** Los objetivos que el asistente propone después de leer los archivos: se elige con un clic. */
export function GoalCards({
  goals,
  disabled,
  onPick,
}: {
  goals: Goal[];
  disabled: boolean;
  onPick: (id: string) => void;
}) {
  const { t } = useTranslation();
  if (!goals.length) return null;
  return (
    <ul aria-label={t("journey.files.goals")} className="grid gap-2 sm:grid-cols-2">
      {goals.map((g) => (
        <li key={g.id}>
          <button
            type="button"
            disabled={disabled}
            onClick={() => onPick(g.id)}
            className="group flex h-full w-full items-start gap-2 rounded-2xl border border-line bg-card p-3 text-left transition hover:border-brand disabled:opacity-50"
          >
            <span className="flex-1">
              <span className="block font-semibold text-ink group-hover:text-brand">{g.title}</span>
              <span className="mt-0.5 block text-sm text-muted">{g.detail}</span>
            </span>
            <ArrowRight className="mt-1 h-4 w-4 shrink-0 text-brand" aria-hidden="true" />
          </button>
        </li>
      ))}
    </ul>
  );
}

/** Los archivos que acepta el asistente para proponer objetivos (o un zip con carpetas). */
export const DATA_FILES =
  ".csv,.tsv,.txt,.xlsx,.xls,.parquet,.json,.jsonl,.zip,.png,.jpg,.jpeg,.wav,.mp3,.flac";

/** El clip para adjuntar archivos de datos. */
export function AttachButton({
  disabled,
  onFiles,
}: {
  disabled: boolean;
  onFiles: (files: File[]) => void;
}) {
  const { t } = useTranslation();
  return (
    <label
      title={t("journey.files.attach")}
      className={`mb-1 inline-flex h-8 w-8 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted hover:bg-canvas hover:text-brand ${
        disabled ? "pointer-events-none opacity-50" : ""
      }`}
    >
      <Paperclip className="h-4 w-4" aria-hidden="true" />
      <input
        type="file"
        multiple
        className="sr-only"
        accept={DATA_FILES}
        disabled={disabled}
        aria-label={t("journey.files.attach")}
        onChange={(e) => {
          const files = Array.from(e.target.files ?? []);
          if (files.length) onFiles(files);
          e.target.value = "";
        }}
      />
    </label>
  );
}

/** La caja de texto: Enter envía, Shift+Enter agrega una línea. Con `onFiles`, también se
 * adjuntan archivos (con el clip o arrastrándolos). */
export function Composer({
  placeholder,
  disabled,
  onSend,
  onFiles,
  autoFocus,
}: {
  placeholder: string;
  disabled: boolean;
  onSend: (text: string) => void;
  onFiles?: (files: File[]) => void;
  autoFocus?: boolean;
}) {
  const { t } = useTranslation();
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    if (autoFocus) ref.current?.focus();
  }, [autoFocus]);
  const send = () => {
    const value = text.trim();
    if (!value || disabled) return;
    onSend(value);
    setText("");
  };
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      send();
    }
  };
  return (
    <form
      className="flex items-end gap-2 rounded-2xl border border-line bg-card p-2 shadow-sm focus-within:border-brand"
      onSubmit={(e) => {
        e.preventDefault();
        send();
      }}
      onDragOver={onFiles ? (e) => e.preventDefault() : undefined}
      onDrop={
        onFiles
          ? (e) => {
              e.preventDefault();
              const files = Array.from(e.dataTransfer.files);
              if (files.length && !disabled) onFiles(files);
            }
          : undefined
      }
    >
      {onFiles ? (
        <AttachButton disabled={disabled} onFiles={onFiles} />
      ) : (
        <Sparkles className="mb-2 ml-1 h-4 w-4 shrink-0 text-brand" aria-hidden="true" />
      )}
      <textarea
        ref={ref}
        rows={1}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKey}
        placeholder={placeholder}
        aria-label={t("journey.message")}
        className="max-h-40 min-h-[2.5rem] flex-1 resize-none bg-transparent px-1 py-2 outline-none focus-visible:outline-none"
      />
      <Button
        type="submit"
        size="sm"
        disabled={disabled || !text.trim()}
        aria-label={t("journey.send")}
      >
        <SendHorizontal className="h-4 w-4" aria-hidden="true" />
      </Button>
    </form>
  );
}

/** Arrastrar y soltar el archivo, o elegirlo: más claro que el selector del navegador. */
export function DropZone({
  label,
  hint,
  accept,
  multiple,
  disabled,
  onFiles,
}: {
  label: string;
  hint: string;
  accept: string;
  multiple: boolean;
  disabled: boolean;
  onFiles: (files: File[]) => void;
}) {
  const [over, setOver] = useState(false);
  return (
    <label
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        const files = Array.from(e.dataTransfer.files);
        if (files.length && !disabled) onFiles(multiple ? files : files.slice(0, 1));
      }}
      className={`flex cursor-pointer flex-col items-center gap-1 rounded-2xl border-2 border-dashed p-6 text-center transition ${
        over ? "border-brand bg-canvas" : "border-line hover:border-brand"
      } ${disabled ? "pointer-events-none opacity-60" : ""}`}
    >
      <Upload className="h-6 w-6 text-brand" aria-hidden="true" />
      <span className="font-semibold">{label}</span>
      <span className="text-xs text-muted">{hint}</span>
      <input
        type="file"
        className="sr-only"
        accept={accept}
        multiple={multiple}
        disabled={disabled}
        aria-label={label}
        onChange={(e) => {
          const files = Array.from(e.target.files ?? []);
          if (files.length) onFiles(files);
          e.target.value = "";
        }}
      />
    </label>
  );
}
