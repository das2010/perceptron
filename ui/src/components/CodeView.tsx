/**
 * Editor de código Monaco (SPEC §5.2), cargado bajo demanda y desde el paquete local:
 * la app de escritorio funciona sin conexión, así que nada se baja de un CDN.
 */
import Editor, { loader, type OnMount } from "@monaco-editor/react";
// Solo el núcleo del editor y Python: el paquete completo trae ~80 lenguajes y pesa varios MB.
import * as monaco from "monaco-editor/editor/editor.api";
import "monaco-editor/languages/definitions/python/register";
import EditorWorker from "monaco-editor/editor/editor.worker?worker";
import { useEffect, useRef } from "react";

import type { CodeMarker } from "./code-marker";

self.MonacoEnvironment = { getWorker: () => new EditorWorker() };
loader.config({ monaco: monaco as unknown as Parameters<typeof loader.config>[0]["monaco"] });

type EditorInstance = Parameters<OnMount>[0];

export default function CodeView({
  code,
  language = "python",
  height = 480,
  dark,
  onChange,
  markers,
  label,
}: {
  code: string;
  language?: string;
  height?: number;
  dark?: boolean;
  /** Si se pasa, el editor es editable. */
  onChange?: (value: string) => void;
  /** Problemas a marcar en el código (p. ej. la validación estática del modo experto). */
  markers?: CodeMarker[];
  label?: string;
}) {
  const editorRef = useRef<EditorInstance | null>(null);

  useEffect(() => {
    const model = editorRef.current?.getModel();
    if (!model) return;
    monaco.editor.setModelMarkers(
      model as unknown as monaco.editor.ITextModel,
      "perceptron",
      (markers ?? []).map((m) => {
        const line = Math.max(1, Math.min(m.line, model.getLineCount()));
        return {
          startLineNumber: line,
          startColumn: m.col + 1,
          endLineNumber: line,
          endColumn: model.getLineMaxColumn(line),
          message: m.message,
          severity: monaco.MarkerSeverity.Error,
        };
      }),
    );
  }, [markers]);

  return (
    <Editor
      height={height}
      language={language}
      value={code}
      theme={dark ? "vs-dark" : "vs"}
      onMount={(editor) => {
        editorRef.current = editor;
      }}
      onChange={(value) => onChange?.(value ?? "")}
      options={{
        readOnly: !onChange,
        minimap: { enabled: false },
        fontSize: 13,
        scrollBeyondLastLine: false,
        ariaLabel: label ?? "code",
      }}
    />
  );
}
