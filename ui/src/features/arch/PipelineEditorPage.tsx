/**
 * Editor visual de pipeline (RF-PIP-02): los pasos como cadena en React Flow; agregar, quitar,
 * reordenar y parametrizar en el panel lateral; vista previa del resultado sobre las primeras
 * filas de train. Guarda con bloqueo optimista (PUT con versión) y el Engine valida el grafo.
 */
import { useNavigate, useParams } from "@tanstack/react-router";
import { Background, Controls, ReactFlow, type Edge, type Node } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { ArrowDown, ArrowLeft, ArrowUp, Eye, Plus, Save, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import {
  Badge,
  Button,
  Card,
  CardTitle,
  ErrorNote,
  Field,
  Select,
  Spinner,
  Table,
  Td,
  Textarea,
  Th,
} from "@/components/ui";
import {
  type Pipeline,
  type PipelineSpec,
  useDatasets,
  usePipeline,
  usePreviewPipeline,
  useUpdatePipeline,
} from "@/lib/api/hooks";

import { move, STEP_KINDS } from "./spec";

type Step = NonNullable<PipelineSpec["steps"]>[number];

function layout(steps: Step[], selected: number | null): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = steps.map((s, i) => ({
    id: s.id,
    position: { x: 0, y: i * 90 },
    data: { label: `${i + 1}. ${s.kind}\n${s.columns.length} col.` },
    style: {
      whiteSpace: "pre-line",
      fontSize: 12,
      borderRadius: 10,
      border: i === selected ? "2px solid var(--color-accent)" : "1px solid var(--color-border)",
      background: "var(--color-surface)",
      color: "var(--color-text)",
    },
  }));
  const edges: Edge[] = steps.slice(1).map((s, i) => ({
    id: `${steps[i]?.id ?? ""}->${s.id}`,
    source: steps[i]?.id ?? "",
    target: s.id,
  }));
  return { nodes, edges };
}

function StepPanel({ step, onChange }: { step: Step; onChange: (s: Step) => void }) {
  const { t } = useTranslation();
  const [paramsText, setParamsText] = useState(() => JSON.stringify(step.params ?? {}, null, 2));
  const [paramsError, setParamsError] = useState(false);
  return (
    <div className="space-y-3">
      <Field label={t("pipeline.kind")}>
        <Select value={step.kind} onChange={(e) => onChange({ ...step, kind: e.target.value })}>
          {[...new Set([step.kind, ...STEP_KINDS])].map((k) => (
            <option key={k} value={k}>
              {k}
            </option>
          ))}
        </Select>
      </Field>
      <Field label={t("pipeline.columns")} hint={t("pipeline.columnsHint")}>
        <Textarea
          rows={3}
          defaultValue={step.columns.join(", ")}
          onBlur={(e) =>
            onChange({
              ...step,
              columns: e.target.value
                .split(",")
                .map((c) => c.trim())
                .filter(Boolean),
            })
          }
        />
      </Field>
      <Field label={t("pipeline.params")} hint={t("pipeline.paramsHint")}>
        <Textarea
          rows={4}
          className="font-mono text-xs"
          value={paramsText}
          aria-invalid={paramsError}
          onChange={(e) => setParamsText(e.target.value)}
          onBlur={() => {
            try {
              const parsed: unknown = JSON.parse(paramsText || "{}");
              if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed))
                throw new Error();
              setParamsError(false);
              onChange({ ...step, params: parsed as Record<string, unknown> });
            } catch {
              setParamsError(true);
            }
          }}
        />
      </Field>
      {paramsError && <p className="text-xs text-bad">{t("pipeline.paramsInvalid")}</p>}
    </div>
  );
}

function Preview({ pipeline, dirty }: { pipeline: Pipeline; dirty: boolean }) {
  const { t } = useTranslation();
  const datasets = useDatasets(pipeline.project_id);
  const [dv, setDv] = useState("");
  const preview = usePreviewPipeline(pipeline.id ?? "");
  const chosen = dv || datasets.data?.at(-1)?.id || "";
  const data = preview.data;
  const headers = data ? [...data.numeric_features, ...data.categorical_features] : [];
  return (
    <Card>
      <CardTitle>{t("pipeline.preview")}</CardTitle>
      <div className="flex flex-wrap items-end gap-2">
        <Select
          value={chosen}
          onChange={(e) => setDv(e.target.value)}
          aria-label={t("wizard.data.pick")}
        >
          {(datasets.data ?? []).map((d) => (
            <option key={d.id} value={d.id}>
              {d.content_hash.slice(0, 10)} · {d.num_samples} · {d.target ?? "—"}
            </option>
          ))}
        </Select>
        <Button
          variant="secondary"
          disabled={!chosen || dirty}
          loading={preview.isPending}
          onClick={() => preview.mutate(chosen)}
        >
          <Eye className="h-4 w-4" aria-hidden="true" />
          {t("pipeline.runPreview")}
        </Button>
        {dirty && <span className="text-xs text-muted">{t("pipeline.saveFirst")}</span>}
      </div>
      <ErrorNote error={preview.error} />
      {data && headers.length > 0 && (
        <div className="mt-3 max-h-80 overflow-auto">
          <Table>
            <thead>
              <tr>
                {headers.map((h) => (
                  <Th key={h}>{h}</Th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.x_num.map((row, i) => (
                <tr key={i}>
                  {[...row, ...(data.x_cat[i] ?? [])].map((v, j) => (
                    <Td key={j} className="font-mono text-xs">
                      {Number.isInteger(v) ? v : v.toFixed(3)}
                    </Td>
                  ))}
                </tr>
              ))}
            </tbody>
          </Table>
        </div>
      )}
      {data && headers.length === 0 && (
        <p className="mt-2 text-sm text-muted">{t("pipeline.noTabularPreview")}</p>
      )}
    </Card>
  );
}

function PipelineEditor({ pipeline }: { pipeline: Pipeline }) {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const original = pipeline.graph as unknown as PipelineSpec;
  const [graph, setGraph] = useState<PipelineSpec>(original);
  const [selected, setSelected] = useState<number | null>(null);
  const save = useUpdatePipeline(pipeline.id ?? "");
  const steps = useMemo(() => graph.steps ?? [], [graph.steps]);
  const flow = useMemo(() => layout(steps, selected), [steps, selected]);
  const dirty = JSON.stringify(graph) !== JSON.stringify(original);
  const step = selected !== null ? steps[selected] : undefined;
  const setSteps = (next: Step[]) => setGraph({ ...graph, steps: next });

  const addStep = () => {
    let n = steps.length + 1;
    while (steps.some((s) => s.id === `paso_${n}`)) n++;
    setSteps([...steps, { id: `paso_${n}`, kind: "scale", columns: [], params: {} }]);
    setSelected(steps.length);
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <Button
          variant="ghost"
          size="icon"
          aria-label={t("arch.back")}
          onClick={() =>
            void navigate({
              to: "/projects/$projectId/design",
              params: { projectId: pipeline.project_id },
            })
          }
        >
          <ArrowLeft className="h-4 w-4" />
        </Button>
        <h2 className="font-semibold">{pipeline.name}</h2>
        <Badge>v{pipeline.version}</Badge>
        <Badge>{graph.modality}</Badge>
        {dirty && <Badge tone="warn">{t("pipeline.unsaved")}</Badge>}
        <div className="ml-auto flex gap-2">
          <Button variant="secondary" disabled={!dirty} onClick={() => setGraph(original)}>
            {t("pipeline.discard")}
          </Button>
          <Button disabled={!dirty} loading={save.isPending} onClick={() => save.mutate(graph)}>
            <Save className="h-4 w-4" aria-hidden="true" />
            {t("pipeline.save")}
          </Button>
        </div>
      </div>
      <ErrorNote error={save.error} />
      <div className="grid gap-4 lg:grid-cols-[1fr_22rem]">
        <Card className="h-[28rem] overflow-hidden p-0">
          <ReactFlow
            nodes={flow.nodes}
            edges={flow.edges}
            fitView
            nodesDraggable={false}
            nodesConnectable={false}
            onNodeClick={(_, n) => setSelected(steps.findIndex((s) => s.id === n.id))}
            onNodesDelete={(ns) => {
              setSteps(steps.filter((s) => !ns.some((n) => n.id === s.id)));
              setSelected(null);
            }}
          >
            <Background />
            <Controls showInteractive={false} />
          </ReactFlow>
        </Card>
        <div className="space-y-4">
          <ol className="space-y-1" aria-label={t("pipeline.steps")}>
            {steps.map((s, i) => (
              <li key={s.id}>
                <button
                  type="button"
                  aria-current={i === selected}
                  onClick={() => setSelected(i)}
                  className={`w-full rounded-pt border px-2 py-1 text-left text-sm ${i === selected ? "border-brand font-semibold" : "border-line"}`}
                >
                  {i + 1}. {s.kind} <span className="text-xs text-muted">({s.columns.length})</span>
                </button>
              </li>
            ))}
          </ol>
          <Button variant="secondary" onClick={addStep}>
            <Plus className="h-4 w-4" aria-hidden="true" />
            {t("pipeline.addStep")}
          </Button>
          {step && selected !== null ? (
            <Card>
              <CardTitle className="flex items-center justify-between gap-2">
                <span className="font-mono text-sm">{step.id}</span>
                <span className="flex gap-1">
                  <Button
                    size="icon"
                    variant="ghost"
                    aria-label={t("pipeline.up")}
                    disabled={selected === 0}
                    onClick={() => {
                      setSteps(move(steps, selected, -1));
                      setSelected(selected - 1);
                    }}
                  >
                    <ArrowUp className="h-4 w-4" />
                  </Button>
                  <Button
                    size="icon"
                    variant="ghost"
                    aria-label={t("pipeline.down")}
                    disabled={selected === steps.length - 1}
                    onClick={() => {
                      setSteps(move(steps, selected, 1));
                      setSelected(selected + 1);
                    }}
                  >
                    <ArrowDown className="h-4 w-4" />
                  </Button>
                  <Button
                    size="icon"
                    variant="ghost"
                    aria-label={t("arch.remove")}
                    onClick={() => {
                      setSteps(steps.filter((_, i) => i !== selected));
                      setSelected(null);
                    }}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </span>
              </CardTitle>
              <StepPanel
                key={step.id}
                step={step}
                onChange={(s) => setSteps(steps.map((x, i) => (i === selected ? s : x)))}
              />
            </Card>
          ) : (
            <p className="text-sm text-muted">{t("pipeline.selectHint")}</p>
          )}
          {(graph.rationale ?? []).length > 0 && (
            <Card>
              <CardTitle>{t("pipeline.rationale")}</CardTitle>
              <ul className="list-disc space-y-1 pl-4 text-xs">
                {(graph.rationale ?? []).map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            </Card>
          )}
        </div>
      </div>
      <Preview pipeline={pipeline} dirty={dirty} />
    </div>
  );
}

export function PipelineEditorPage() {
  const { pipelineId } = useParams({ from: "/projects/$projectId/pipelines/$pipelineId" });
  const pipeline = usePipeline(pipelineId);
  if (pipeline.error) return <ErrorNote error={pipeline.error} />;
  if (!pipeline.data) return <Spinner />;
  // key con la versión: tras guardar, el editor arranca del grafo que devolvió el Engine.
  return <PipelineEditor key={`${pipelineId}:${pipeline.data.version}`} pipeline={pipeline.data} />;
}
