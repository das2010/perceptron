/** «Fuentes del servidor» (RF-SRV-05): en la UI web reemplazan al selector de carpetas local. */
import { ChevronRight, Folder, FileText, HardDrive } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { Button, ErrorNote, Spinner } from "@/components/ui";
import { useSession } from "@/features/auth/session";
import {
  type DataSource,
  type SourcePreview,
  useBrowseServerSource,
  useCreatePathSource,
  useServerSources,
} from "@/lib/api/hooks";

type Ready = (res: { source: DataSource; preview: SourcePreview }) => void;

function Browser({
  projectId,
  index,
  onReady,
}: {
  projectId: string;
  index: number;
  onReady: Ready;
}) {
  const { t } = useTranslation();
  const [path, setPath] = useState("");
  const listing = useBrowseServerSource(index, path);
  const create = useCreatePathSource(projectId);
  const parts = path ? path.split("/") : [];
  return (
    <div className="space-y-2 rounded-pt border border-line p-3">
      <nav
        aria-label={t("serverSources.path")}
        className="flex flex-wrap items-center gap-1 text-sm"
      >
        <button type="button" className="underline" onClick={() => setPath("")}>
          {listing.data?.root.name ?? "/"}
        </button>
        {parts.map((p, i) => (
          <span key={`${p}-${i}`} className="flex items-center gap-1">
            <ChevronRight className="h-3 w-3" aria-hidden="true" />
            <button
              type="button"
              className="underline"
              onClick={() => setPath(parts.slice(0, i + 1).join("/"))}
            >
              {p}
            </button>
          </span>
        ))}
      </nav>
      {listing.isPending && <Spinner />}
      <ErrorNote error={listing.error ?? create.error} />
      <ul className="max-h-64 space-y-1 overflow-y-auto text-sm">
        {listing.data?.entries.map((e) => (
          <li key={e.path} className="flex items-center justify-between gap-2">
            {e.kind === "dir" ? (
              <button
                type="button"
                className="flex items-center gap-2 hover:underline"
                onClick={() => setPath(path ? `${path}/${e.name}` : e.name)}
              >
                <Folder className="h-4 w-4" aria-hidden="true" />
                {e.name}
              </button>
            ) : (
              <span className="flex items-center gap-2">
                <FileText className="h-4 w-4" aria-hidden="true" />
                {e.name}
              </span>
            )}
            <Button
              size="sm"
              variant="ghost"
              loading={create.isPending && create.variables.path === e.path}
              onClick={() => create.mutate({ path: e.path, kind: e.kind }, { onSuccess: onReady })}
            >
              {t("serverSources.use")}
            </Button>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function ServerSources({ projectId, onReady }: { projectId: string; onReady: Ready }) {
  const { t } = useTranslation();
  const me = useSession();
  const roots = useServerSources(Boolean(me));
  const [open, setOpen] = useState<number | null>(null);
  // Solo en el Team Server y si el Admin habilitó alguna carpeta (y el rol lo permite).
  if (!me || !roots.data?.length) return null;
  return (
    <div className="mt-3 space-y-2">
      <p className="text-sm font-semibold">{t("serverSources.title")}</p>
      <div className="flex flex-wrap gap-2">
        {roots.data.map((r) => (
          <Button
            key={r.index}
            variant="ghost"
            size="sm"
            aria-expanded={open === r.index}
            onClick={() => setOpen(open === r.index ? null : r.index)}
          >
            <HardDrive className="h-4 w-4" aria-hidden="true" />
            {r.name}
          </Button>
        ))}
      </div>
      {open !== null && <Browser projectId={projectId} index={open} onReady={onReady} />}
    </div>
  );
}
