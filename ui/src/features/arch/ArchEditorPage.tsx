/**
 * Editor visual de ArchSpec (RF-ARC-05): bloques del catálogo como nodos, parámetros en el
 * panel lateral, validación en vivo con el validador del Engine (§9.3), resumen de parámetros
 * y memoria, y "ver como código" (RF-ARC-07). Guardar crea una ArchSpec nueva con origen
 * manual: la original (reglas/LLM/agente) queda intacta como procedencia.
 */
import { useNavigate, useParams } from "@tanstack/react-router";
import {
  Background,
  Controls,
  ReactFlow,
  type Connection,
  type Edge,
  type Node,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { ArrowLeft, Code2, Plus, Save, Trash2 } from "lucide-react";
import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import {
  Badge,
  Button,
  Card,
  CardTitle,
  ErrorNote,
  Field,
  Input,
  Select,
  Spinner,
} from "@/components/ui";
import {
  type ArchSpecRecord,
  type ValidationReport,
  useArchCode,
  useArchSpec,
  useCatalogBlocks,
  useSaveArchSpec,
  validateArchSpec,
} from "@/lib/api/hooks";
import { formatNumber } from "@/lib/format";

import { addBlock, INPUT_NODE, isHp, hpName, removeBlock, type Spec, type SpecNode } from "./spec";

const CodeView = lazy(() => import("@/components/CodeView"));

interface ParamInfo {
  type: string;
  default?: unknown;
  low?: number | null;
  high?: number | null;
  choices?: unknown[] | null;
  tunable?: boolean;
  description?: string;
}
interface BlockInfo {
  key: string;
  description: string;
  params: Record<string, ParamInfo>;
}

function layout(spec: Spec, selected: string | null): { nodes: Node[]; edges: Edge[] } {
  const order = [INPUT_NODE, ...spec.nodes.map((n) => n.id)];
  const nodes: Node[] = order.map((id, i) => {
    const block =
      id === INPUT_NODE ? spec.modality : (spec.nodes.find((n) => n.id === id)?.block ?? "");
    return {
      id,
      position: { x: 0, y: i * 100 },
      data: { label: `${id}\n${block}` },
      deletable: id !== INPUT_NODE,
      style: {
        whiteSpace: "pre-line",
        fontSize: 12,
        borderRadius: 10,
        border: id === selected ? "2px solid var(--color-accent)" : "1px solid var(--color-border)",
        background: "var(--color-surface)",
        color: "var(--color-text)",
      },
    };
  });
  const edges: Edge[] = spec.edges.map(([a, b]) => ({ id: `${a}->${b}`, source: a, target: b }));
  return { nodes, edges };
}

// ---------------------------------------------------------------- panel de parámetros

function ParamEditor({
  node,
  block,
  onChange,
}: {
  node: SpecNode;
  block: BlockInfo | undefined;
  onChange: (params: Record<string, unknown>) => void;
}) {
  const { t } = useTranslation();
  const entries = Object.entries(block?.params ?? {});
  if (!entries.length) return <p className="text-sm text-muted">{t("arch.noParams")}</p>;
  return (
    <div className="space-y-3">
      {entries.map(([name, info]) => {
        const current = node.params[name] ?? info.default;
        const tuned = isHp(current);
        const value = tuned ? current.default : current;
        const set = (v: unknown, asHp = tuned) => {
          const next = { ...node.params };
          next[name] = asHp ? { hp: tuned ? current.hp : hpName(node.id, name), default: v } : v;
          onChange(next);
        };
        const parse = (raw: string): unknown =>
          info.type === "int"
            ? Number.parseInt(raw, 10)
            : info.type === "float"
              ? Number(raw)
              : info.type === "int_list"
                ? raw.split(",").map((x) => Number.parseInt(x.trim(), 10))
                : raw;
        return (
          <div key={name} className="rounded-pt border border-line p-2">
            <Field label={name} {...(info.description ? { hint: info.description } : {})}>
              {info.choices?.length ? (
                <Select
                  value={String(value)}
                  onChange={(e) =>
                    set(info.choices?.find((c) => String(c) === e.target.value) ?? e.target.value)
                  }
                >
                  {info.choices.map((c) => (
                    <option key={String(c)} value={String(c)}>
                      {String(c)}
                    </option>
                  ))}
                </Select>
              ) : info.type === "bool" ? (
                <Select value={String(value)} onChange={(e) => set(e.target.value === "true")}>
                  <option value="true">true</option>
                  <option value="false">false</option>
                </Select>
              ) : (
                <Input
                  defaultValue={Array.isArray(value) ? value.join(", ") : String(value ?? "")}
                  onBlur={(e) => set(parse(e.target.value))}
                />
              )}
            </Field>
            {info.tunable && (
              <label className="mt-2 flex items-center gap-2 text-xs">
                <input
                  type="checkbox"
                  checked={tuned}
                  onChange={(e) => set(value, e.target.checked)}
                />
                {t("arch.tunable")}
                {(info.low != null || info.high != null) && (
                  <span className="text-muted">
                    [{String(info.low ?? "…")}, {String(info.high ?? "…")}]
                  </span>
                )}
              </label>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------- editor

function useLiveValidation(spec: Spec) {
  const [report, setReport] = useState<ValidationReport | null>(null);
  const [error, setError] = useState<unknown>(null);
  useEffect(() => {
    let alive = true;
    const handle = setTimeout(() => {
      validateArchSpec(spec as unknown as Record<string, unknown>).then(
        (r) => {
          if (!alive) return;
          setReport(r);
          setError(null);
        },
        (e: unknown) => {
          if (alive) setError(e);
        },
      );
    }, 400);
    return () => {
      alive = false;
      clearTimeout(handle);
    };
  }, [spec]);
  return { report, error };
}

function prefersDark(): boolean {
  const theme = document.documentElement.dataset.theme;
  if (theme) return theme === "dark";
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
}

function ArchEditor({ record, projectId }: { record: ArchSpecRecord; projectId: string }) {
  const { t, i18n } = useTranslation();
  const navigate = useNavigate();
  const [spec, setSpec] = useState<Spec>(() => record.spec as unknown as Spec);
  const [selected, setSelected] = useState<string | null>(null);
  const [newBlock, setNewBlock] = useState("");
  const [showCode, setShowCode] = useState(false);
  const blocks = useCatalogBlocks(spec.modality);
  const catalog = (blocks.data ?? []) as unknown as BlockInfo[];
  const { report, error } = useLiveValidation(spec);
  const code = useArchCode(
    spec as unknown as Record<string, unknown>,
    showCode && Boolean(report?.valid),
  );
  const save = useSaveArchSpec(projectId);
  const graph = useMemo(() => layout(spec, selected), [spec, selected]);
  const node = spec.nodes.find((n) => n.id === selected);
  const errors = (report?.issues ?? []).filter((i) => i.severity === "error");

  const onConnect = (c: Connection) => {
    if (!c.source || !c.target || c.source === c.target) return;
    if (spec.edges.some(([a, b]) => a === c.source && b === c.target)) return;
    setSpec({ ...spec, edges: [...spec.edges, [c.source, c.target]] });
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <Button
          variant="ghost"
          size="icon"
          aria-label={t("arch.back")}
          onClick={() =>
            void navigate({ to: "/projects/$projectId/design", params: { projectId } })
          }
        >
          <ArrowLeft className="h-4 w-4" />
        </Button>
        <Input
          value={spec.name}
          onChange={(e) => setSpec({ ...spec, name: e.target.value })}
          aria-label={t("arch.name")}
          className="w-64"
        />
        {report && (
          <Badge tone={report.valid ? "ok" : "bad"}>
            {report.valid ? t("arch.valid") : t("arch.invalid", { count: errors.length })}
          </Badge>
        )}
        {report?.num_params != null && (
          <span className="text-sm text-muted">
            {t("arch.summary", {
              params: formatNumber(report.num_params, i18n.language, 0),
              memory: formatNumber(report.estimated_memory_mb ?? null, i18n.language, 0),
            })}
          </span>
        )}
        <div className="ml-auto flex gap-2">
          <Button
            variant="secondary"
            onClick={() => setShowCode((v) => !v)}
            aria-pressed={showCode}
          >
            <Code2 className="h-4 w-4" aria-hidden="true" />
            {t("arch.code")}
          </Button>
          <Button
            disabled={!report?.valid}
            loading={save.isPending}
            onClick={() =>
              save.mutate(spec as unknown as Record<string, unknown>, {
                onSuccess: (rec) =>
                  void navigate({
                    to: "/projects/$projectId/archspecs/$archspecId",
                    params: { projectId, archspecId: rec.id },
                  }),
              })
            }
          >
            <Save className="h-4 w-4" aria-hidden="true" />
            {t("arch.saveAsNew")}
          </Button>
        </div>
      </div>
      <ErrorNote error={save.error ?? error} />
      <div className="grid gap-4 lg:grid-cols-[1fr_22rem]">
        <Card className="h-[32rem] overflow-hidden p-0">
          <ReactFlow
            nodes={graph.nodes}
            edges={graph.edges}
            fitView
            nodesDraggable={false}
            onNodeClick={(_, n) => setSelected(n.id === INPUT_NODE ? null : n.id)}
            onConnect={onConnect}
            onNodesDelete={(ns) => {
              setSpec(ns.reduce((s, n) => removeBlock(s, n.id), spec));
              setSelected(null);
            }}
            onEdgesDelete={(es) =>
              setSpec({
                ...spec,
                edges: spec.edges.filter(
                  ([a, b]) => !es.some((e) => e.source === a && e.target === b),
                ),
              })
            }
          >
            <Background />
            <Controls showInteractive={false} />
          </ReactFlow>
        </Card>
        <div className="space-y-4">
          <Card>
            <CardTitle>{t("arch.addBlock")}</CardTitle>
            <div className="flex gap-2">
              <Select
                value={newBlock}
                onChange={(e) => setNewBlock(e.target.value)}
                className="flex-1"
                aria-label={t("arch.addBlock")}
              >
                <option value="">{t("arch.pickBlock")}</option>
                {catalog.map((b) => (
                  <option key={b.key} value={b.key} title={b.description}>
                    {b.key}
                  </option>
                ))}
              </Select>
              <Button
                size="icon"
                disabled={!newBlock}
                aria-label={t("arch.add")}
                onClick={() => {
                  const next = addBlock(spec, newBlock);
                  setSpec(next.spec);
                  setSelected(next.id);
                }}
              >
                <Plus className="h-4 w-4" />
              </Button>
            </div>
          </Card>
          {node ? (
            <Card>
              <CardTitle className="flex items-center justify-between gap-2">
                <span>
                  {node.id} <span className="text-xs font-normal text-muted">{node.block}</span>
                </span>
                <Button
                  size="icon"
                  variant="ghost"
                  aria-label={t("arch.remove")}
                  onClick={() => {
                    setSpec(removeBlock(spec, node.id));
                    setSelected(null);
                  }}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </CardTitle>
              <ParamEditor
                key={node.id}
                node={node}
                block={catalog.find((b) => b.key === node.block)}
                onChange={(params) =>
                  setSpec({
                    ...spec,
                    nodes: spec.nodes.map((n) => (n.id === node.id ? { ...n, params } : n)),
                  })
                }
              />
            </Card>
          ) : (
            <p className="text-sm text-muted">{t("arch.selectHint")}</p>
          )}
          {errors.length > 0 && (
            <Card>
              <CardTitle>{t("arch.issues")}</CardTitle>
              <ul className="space-y-1 text-xs">
                {errors.map((i, k) => (
                  <li key={k}>
                    <Badge tone="bad" className="mr-1">
                      {i.stage}
                    </Badge>
                    <span className="font-mono">{i.path}</span>: {i.message}
                  </li>
                ))}
              </ul>
            </Card>
          )}
        </div>
      </div>
      {showCode && (
        <Card className="p-0">
          {!report?.valid ? (
            <p className="p-4 text-sm text-muted">{t("arch.codeNeedsValid")}</p>
          ) : code.data ? (
            <Suspense fallback={<Spinner />}>
              <CodeView code={code.data} dark={prefersDark()} />
            </Suspense>
          ) : (
            <div className="p-4">
              <Spinner />
            </div>
          )}
        </Card>
      )}
    </div>
  );
}

export function ArchEditorPage() {
  const { projectId, archspecId } = useParams({
    from: "/projects/$projectId/archspecs/$archspecId",
  });
  const record = useArchSpec(archspecId);
  if (record.error) return <ErrorNote error={record.error} />;
  if (!record.data?.spec) return <Spinner />;
  return <ArchEditor key={archspecId} record={record.data} projectId={projectId} />;
}
