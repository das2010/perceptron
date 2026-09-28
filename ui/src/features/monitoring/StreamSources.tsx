/** Fuentes streaming y API (RF-ING-05): alta (REST, WebSocket, Kafka, MQTT, archivo), estado del
 * buffer y lectura manual. El reentrenamiento las usa como datos nuevos. */
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Radio, RefreshCw } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import {
  Badge,
  Button,
  Card,
  CardTitle,
  EmptyState,
  ErrorNote,
  Field,
  Input,
  Select,
  Table,
  Td,
  Th,
} from "@/components/ui";
import { getApiClient } from "@/lib/api/client";
import { type Schemas, unwrap } from "@/lib/api/hooks";
import { formatDate } from "@/lib/format";

import { streamKey, useStreamSources } from "./hooks";

type Create = Schemas["StreamSourceCreate"];
const KINDS = ["rest", "websocket", "kafka", "mqtt", "file"] as const;
type Kind = (typeof KINDS)[number];

type Fields = Record<string, string>;
const DEFAULTS: Record<Kind, Fields> = {
  rest: { url: "", method: "GET", records_path: "", pagination: "none", auth: "none" },
  websocket: { url: "", records_path: "", auth: "none", max_messages: "1000" },
  kafka: { bootstrap_servers: "", topic: "", group_id: "perceptron", auth: "none", username: "" },
  mqtt: { host: "", port: "1883", topic: "", qos: "1", client_id: "perceptron", username: "" },
  file: { path: "" },
};
const OPTIONS: Record<string, readonly string[]> = {
  method: ["GET", "POST"],
  pagination: ["none", "offset", "page", "cursor", "link"],
  qos: ["0", "1", "2"],
};
const AUTH: Record<Kind, readonly string[]> = {
  rest: ["none", "bearer", "header"],
  websocket: ["none", "bearer", "header"],
  kafka: ["none", "sasl_plain", "sasl_scram_256", "sasl_scram_512"],
  mqtt: [],
  file: [],
};
const NUMERIC = new Set(["port", "qos", "max_messages"]);

/** Campos del formulario → config del backend (solo lo cargado; tipos del schema). */
function toConfig(kind: Kind, f: Fields, secure: boolean): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(f)) {
    if (v === "") continue;
    if (k === "bootstrap_servers")
      out[k] = v
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
    else out[k] = NUMERIC.has(k) ? Number(v) : v;
  }
  if (kind === "kafka") out["ssl"] = secure;
  if (kind === "mqtt") out["tls"] = secure;
  return out;
}

function CreateForm({ projectId, onDone }: { projectId: string; onDone: () => void }) {
  const { t } = useTranslation();
  const qc = useQueryClient();
  const [kind, setKind] = useState<Kind>("rest");
  const [name, setName] = useState("");
  const [fields, setFields] = useState<Fields>(DEFAULTS.rest);
  const [secret, setSecret] = useState("");
  const [secure, setSecure] = useState(false);
  const [poll, setPoll] = useState("");
  const create = useMutation({
    mutationFn: async (body: Create) =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/projects/{project_id}/sources/stream", {
          params: { path: { project_id: projectId } },
          body,
        }),
      ),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: streamKey(projectId) });
      onDone();
    },
  });
  const needsSecret =
    kind === "mqtt"
      ? Boolean(fields["username"])
      : Boolean(fields["auth"] && fields["auth"] !== "none");
  const set = (k: string) => (e: { target: { value: string } }) =>
    setFields((x) => ({ ...x, [k]: e.target.value }));
  return (
    <form
      className="grid gap-3 sm:grid-cols-3"
      onSubmit={(e) => {
        e.preventDefault();
        create.mutate({
          name: name || `${kind} ${fields["topic"] ?? fields["url"] ?? fields["path"] ?? ""}`,
          kind,
          config: toConfig(kind, fields, secure),
          token: needsSecret && secret ? secret : null,
          poll_interval_s: poll ? Number(poll) : null,
        });
      }}
    >
      <Field label={t("streams.kind")}>
        <Select
          value={kind}
          onChange={(e) => {
            const k = e.target.value as Kind;
            setKind(k);
            setFields(DEFAULTS[k]);
          }}
        >
          {KINDS.map((k) => (
            <option key={k} value={k}>
              {t(`streams.kinds.${k}`)}
            </option>
          ))}
        </Select>
      </Field>
      <Field label={t("streams.name")}>
        <Input value={name} onChange={(e) => setName(e.target.value)} />
      </Field>
      {Object.keys(DEFAULTS[kind]).map((k) =>
        k === "auth" ? (
          <Field key={k} label={t("streams.field.auth")}>
            <Select value={fields[k] ?? "none"} onChange={set(k)}>
              {AUTH[kind].map((a) => (
                <option key={a} value={a}>
                  {t(`streams.auth.${a}`, { defaultValue: a })}
                </option>
              ))}
            </Select>
          </Field>
        ) : OPTIONS[k] ? (
          <Field key={k} label={t(`streams.field.${k}`)}>
            <Select value={fields[k] ?? ""} onChange={set(k)}>
              {(OPTIONS[k] ?? []).map((o) => (
                <option key={o} value={o}>
                  {o}
                </option>
              ))}
            </Select>
          </Field>
        ) : (
          <Field key={k} label={t(`streams.field.${k}`)} hint={t(`streams.hint.${k}`, "")}>
            <Input
              value={fields[k] ?? ""}
              onChange={set(k)}
              inputMode={NUMERIC.has(k) ? "numeric" : undefined}
              required={["url", "topic", "host", "bootstrap_servers", "path"].includes(k)}
            />
          </Field>
        ),
      )}
      {needsSecret && (
        <Field label={t("streams.secret")} hint={t("sources.passwordHint")}>
          <Input
            type="password"
            value={secret}
            onChange={(e) => setSecret(e.target.value)}
            autoComplete="off"
          />
        </Field>
      )}
      {(kind === "kafka" || kind === "mqtt") && (
        <label className="flex items-center gap-2 self-end pb-2 text-sm">
          <input type="checkbox" checked={secure} onChange={(e) => setSecure(e.target.checked)} />
          {t("streams.tls")}
        </label>
      )}
      <Field label={t("streams.poll")} hint={t("streams.pollHint")}>
        <Input inputMode="numeric" value={poll} onChange={(e) => setPoll(e.target.value)} />
      </Field>
      <div className="sm:col-span-3">
        <Button type="submit" loading={create.isPending}>
          {t("streams.create")}
        </Button>
        <ErrorNote error={create.error} />
      </div>
    </form>
  );
}

export function StreamSources({ projectId }: { projectId: string }) {
  const { t, i18n } = useTranslation();
  const qc = useQueryClient();
  const sources = useStreamSources(projectId);
  const [open, setOpen] = useState(false);
  const pull = useMutation({
    mutationFn: async (id: string) =>
      unwrap(
        await (
          await getApiClient()
        ).POST("/api/v1/sources/{source_id}/pull", { params: { path: { source_id: id } } }),
      ),
    onSuccess: () => void qc.invalidateQueries({ queryKey: streamKey(projectId) }),
  });
  return (
    <Card>
      <CardTitle className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-2">
          <Radio className="h-4 w-4" aria-hidden="true" />
          {t("streams.title")}
        </span>
        <Button variant="ghost" size="sm" aria-expanded={open} onClick={() => setOpen(!open)}>
          {t("streams.add")}
        </Button>
      </CardTitle>
      {open && <CreateForm projectId={projectId} onDone={() => setOpen(false)} />}
      <ErrorNote error={sources.error ?? pull.error} />
      {sources.data?.length === 0 && !open && <EmptyState>{t("streams.none")}</EmptyState>}
      {sources.data && sources.data.length > 0 && (
        <Table>
          <thead>
            <tr>
              <Th>{t("streams.name")}</Th>
              <Th>{t("streams.kind")}</Th>
              <Th>{t("streams.rows")}</Th>
              <Th>{t("streams.lastPoll")}</Th>
              <Th />
            </tr>
          </thead>
          <tbody>
            {sources.data.map(({ source, buffer }) => {
              const cfg = (source.config ?? {}) as {
                stream?: { kind?: string };
                last_poll?: string;
              };
              return (
                <tr key={source.id}>
                  <Td>{source.name}</Td>
                  <Td>
                    <Badge>{t(`streams.kinds.${cfg.stream?.kind ?? "rest"}`)}</Badge>
                  </Td>
                  <Td>{t("streams.buffer", { rows: buffer.rows, batches: buffer.batches })}</Td>
                  <Td className="text-xs">
                    {cfg.last_poll ? formatDate(cfg.last_poll, i18n.language) : "—"}
                  </Td>
                  <Td>
                    <Button
                      size="sm"
                      variant="secondary"
                      loading={pull.isPending && pull.variables === source.id}
                      onClick={() => source.id && pull.mutate(source.id)}
                    >
                      <RefreshCw className="h-4 w-4" aria-hidden="true" />
                      {t("streams.pull")}
                    </Button>
                  </Td>
                </tr>
              );
            })}
          </tbody>
        </Table>
      )}
    </Card>
  );
}
