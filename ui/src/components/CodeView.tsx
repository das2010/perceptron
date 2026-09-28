/**
 * Editor de código Monaco (SPEC §5.2), cargado bajo demanda y desde el paquete local:
 * la app de escritorio funciona sin conexión, así que nada se baja de un CDN.
 */
import Editor, { loader } from "@monaco-editor/react";
// Solo el núcleo del editor y Python: el paquete completo trae ~80 lenguajes y pesa varios MB.
import * as monaco from "monaco-editor/editor/editor.api";
import "monaco-editor/languages/definitions/python/register";
import EditorWorker from "monaco-editor/editor/editor.worker?worker";

self.MonacoEnvironment = { getWorker: () => new EditorWorker() };
loader.config({ monaco: monaco as unknown as Parameters<typeof loader.config>[0]["monaco"] });

export default function CodeView({
  code,
  language = "python",
  height = 480,
  dark,
}: {
  code: string;
  language?: string;
  height?: number;
  dark?: boolean;
}) {
  return (
    <Editor
      height={height}
      language={language}
      value={code}
      theme={dark ? "vs-dark" : "vs"}
      options={{
        readOnly: true,
        minimap: { enabled: false },
        fontSize: 13,
        scrollBeyondLastLine: false,
      }}
    />
  );
}
