/** Servidores de equipo (RF-SRV-03): conectar el desktop a un Team Server. */
import { Server, Trash2 } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import {
  Button,
  Card,
  CardTitle,
  EmptyState,
  ErrorNote,
  Field,
  Input,
  Table,
  Td,
  Th,
} from "@/components/ui";
import { useSession } from "@/features/auth/session";
import { useConnectRemote, useRemoteServers, useRemoveRemote } from "@/lib/api/hooks";

export function RemoteServers() {
  const { t } = useTranslation();
  const me = useSession();
  const servers = useRemoteServers(!me);
  const connect = useConnectRemote();
  const remove = useRemoveRemote();
  const empty = { name: "equipo", url: "https://", email: "", password: "" };
  const [f, setF] = useState(empty);
  // En la UI web del propio Team Server no tiene sentido conectarse a otro.
  if (me) return null;
  return (
    <Card>
      <CardTitle>{t("remote.title")}</CardTitle>
      <p className="mb-4 text-sm text-muted">{t("remote.hint")}</p>
      {servers.data?.length === 0 && <EmptyState>{t("remote.empty")}</EmptyState>}
      {servers.data && servers.data.length > 0 && (
        <Table className="mb-4">
          <thead>
            <tr>
              <Th>{t("remote.name")}</Th>
              <Th>URL</Th>
              <Th>{t("remote.user")}</Th>
              <Th />
            </tr>
          </thead>
          <tbody>
            {servers.data.map((s) => (
              <tr key={s.name}>
                <Td className="font-semibold">{s.name}</Td>
                <Td className="text-xs">{s.url}</Td>
                <Td className="text-xs">{s.email}</Td>
                <Td className="text-right">
                  <Button
                    size="sm"
                    variant="ghost"
                    aria-label={t("remote.remove", { name: s.name })}
                    onClick={() => remove.mutate(s.name)}
                  >
                    <Trash2 className="h-4 w-4" aria-hidden="true" />
                  </Button>
                </Td>
              </tr>
            ))}
          </tbody>
        </Table>
      )}
      <form
        className="grid gap-3 sm:grid-cols-4"
        onSubmit={(e) => {
          e.preventDefault();
          connect.mutate(f, { onSuccess: () => setF(empty) });
        }}
      >
        <Field label={t("remote.name")}>
          <Input value={f.name} onChange={(e) => setF({ ...f, name: e.target.value })} required />
        </Field>
        <Field label="URL">
          <Input
            type="url"
            value={f.url}
            onChange={(e) => setF({ ...f, url: e.target.value })}
            required
          />
        </Field>
        <Field label={t("auth.email")}>
          <Input
            type="email"
            value={f.email}
            onChange={(e) => setF({ ...f, email: e.target.value })}
            required
          />
        </Field>
        <Field label={t("auth.password")} hint={t("remote.passwordHint")}>
          <Input
            type="password"
            autoComplete="off"
            value={f.password}
            onChange={(e) => setF({ ...f, password: e.target.value })}
            required
          />
        </Field>
        <div className="sm:col-span-4">
          <Button type="submit" loading={connect.isPending}>
            <Server className="h-4 w-4" aria-hidden="true" />
            {t("remote.connect")}
          </Button>
          <ErrorNote error={connect.error ?? remove.error ?? servers.error} />
        </div>
      </form>
    </Card>
  );
}
