/** Fuentes remotas (RF-ING-03/04): consulta SQL o dataset público de Hugging Face / Kaggle. */
import { Database, Globe } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { Button, ErrorNote, Field, Input, Select, Textarea } from "@/components/ui";
import {
  type DataSource,
  type SourcePreview,
  useCreateDbSource,
  useCreateHubSource,
} from "@/lib/api/hooks";

type Ready = (res: { source: DataSource; preview: SourcePreview }) => void;
const DIALECTS = ["postgresql", "mysql", "mssql", "sqlite"] as const;
type Dialect = (typeof DIALECTS)[number];

interface DbFields {
  name: string;
  dialect: Dialect;
  host: string;
  port: string;
  database: string;
  user: string;
  password: string;
  query: string;
}

function DbForm({ projectId, onReady }: { projectId: string; onReady: Ready }) {
  const { t } = useTranslation();
  const create = useCreateDbSource(projectId);
  const [f, setF] = useState<DbFields>({
    name: "",
    dialect: "postgresql",
    host: "",
    port: "",
    database: "",
    user: "",
    password: "",
    query: "SELECT * FROM ",
  });
  const set = (k: keyof DbFields) => (e: { target: { value: string } }) =>
    setF((x) => ({ ...x, [k]: e.target.value }));
  const sqlite = f.dialect === "sqlite";
  return (
    <form
      className="grid gap-3 sm:grid-cols-3"
      onSubmit={(e) => {
        e.preventDefault();
        create.mutate(
          {
            name: f.name || f.database,
            password: f.password || null,
            config: {
              dialect: f.dialect,
              database: f.database,
              host: sqlite ? null : f.host || null,
              port: sqlite || !f.port ? null : Number(f.port),
              user: sqlite ? null : f.user || null,
              query: f.query,
              max_rows: 5_000_000,
              odbc_driver: "ODBC Driver 18 for SQL Server",
            },
          },
          { onSuccess: onReady },
        );
      }}
    >
      <Field label={t("sources.dialect")}>
        <Select value={f.dialect} onChange={set("dialect")}>
          {DIALECTS.map((d) => (
            <option key={d} value={d}>
              {t(`sources.dialects.${d}`)}
            </option>
          ))}
        </Select>
      </Field>
      <Field label={sqlite ? t("sources.file") : t("sources.database")}>
        <Input value={f.database} onChange={set("database")} required />
      </Field>
      <Field label={t("sources.name")}>
        <Input value={f.name} onChange={set("name")} />
      </Field>
      {!sqlite && (
        <>
          <Field label={t("sources.host")}>
            <Input value={f.host} onChange={set("host")} required />
          </Field>
          <Field label={t("sources.port")}>
            <Input value={f.port} onChange={set("port")} inputMode="numeric" />
          </Field>
          <Field label={t("sources.user")}>
            <Input value={f.user} onChange={set("user")} autoComplete="off" />
          </Field>
          <Field label={t("sources.password")} hint={t("sources.passwordHint")}>
            <Input
              type="password"
              value={f.password}
              onChange={set("password")}
              autoComplete="off"
            />
          </Field>
        </>
      )}
      <div className="sm:col-span-3">
        <Field label={t("sources.query")} hint={t("sources.queryHint")}>
          <Textarea
            rows={3}
            className="font-mono text-xs"
            value={f.query}
            onChange={set("query")}
            required
          />
        </Field>
      </div>
      <div className="sm:col-span-3">
        <Button type="submit" loading={create.isPending}>
          <Database className="h-4 w-4" aria-hidden="true" />
          {t("sources.runQuery")}
        </Button>
        <ErrorNote error={create.error} />
      </div>
    </form>
  );
}

function HubForm({ projectId, onReady }: { projectId: string; onReady: Ready }) {
  const { t } = useTranslation();
  const create = useCreateHubSource(projectId);
  const [provider, setProvider] = useState<"huggingface" | "kaggle">("huggingface");
  const [dataset, setDataset] = useState("");
  const [split, setSplit] = useState("train");
  const [token, setToken] = useState("");
  return (
    <form
      className="grid gap-3 sm:grid-cols-3"
      onSubmit={(e) => {
        e.preventDefault();
        create.mutate(
          {
            provider,
            dataset,
            split: provider === "huggingface" ? split || null : null,
            token: token || null,
            name: null,
          },
          { onSuccess: onReady },
        );
      }}
    >
      <Field label={t("sources.provider")}>
        <Select
          value={provider}
          onChange={(e) => setProvider(e.target.value as "huggingface" | "kaggle")}
        >
          <option value="huggingface">Hugging Face</option>
          <option value="kaggle">Kaggle</option>
        </Select>
      </Field>
      <Field
        label={t("sources.dataset")}
        hint={provider === "kaggle" ? "owner/slug" : "org/dataset"}
      >
        <Input value={dataset} onChange={(e) => setDataset(e.target.value)} required />
      </Field>
      {provider === "huggingface" && (
        <Field label={t("sources.split")}>
          <Input value={split} onChange={(e) => setSplit(e.target.value)} />
        </Field>
      )}
      <Field
        label={provider === "kaggle" ? t("sources.kaggleKey") : t("sources.hfToken")}
        hint={t("sources.passwordHint")}
      >
        <Input
          type="password"
          value={token}
          onChange={(e) => setToken(e.target.value)}
          autoComplete="off"
        />
      </Field>
      <div className="sm:col-span-3">
        <Button type="submit" loading={create.isPending}>
          <Globe className="h-4 w-4" aria-hidden="true" />
          {t("sources.download")}
        </Button>
        <ErrorNote error={create.error} />
      </div>
    </form>
  );
}

export function RemoteSources({ projectId, onReady }: { projectId: string; onReady: Ready }) {
  const { t } = useTranslation();
  const [open, setOpen] = useState<"db" | "hub" | null>(null);
  return (
    <div className="mt-3 space-y-3">
      <div className="flex flex-wrap gap-2">
        <Button
          variant="ghost"
          size="sm"
          aria-expanded={open === "db"}
          onClick={() => setOpen(open === "db" ? null : "db")}
        >
          <Database className="h-4 w-4" aria-hidden="true" />
          {t("sources.fromDb")}
        </Button>
        <Button
          variant="ghost"
          size="sm"
          aria-expanded={open === "hub"}
          onClick={() => setOpen(open === "hub" ? null : "hub")}
        >
          <Globe className="h-4 w-4" aria-hidden="true" />
          {t("sources.fromHub")}
        </Button>
      </div>
      {open === "db" && <DbForm projectId={projectId} onReady={onReady} />}
      {open === "hub" && <HubForm projectId={projectId} onReady={onReady} />}
    </div>
  );
}
