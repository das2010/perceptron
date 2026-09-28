/**
 * Consola de administración del Team Server (RF-SRV-06, parcial en la Capa 5a): usuarios,
 * roles por workspace y proyecto, y auditoría (RF-SRV-07).
 */
import { ShieldCheck, Trash2, UserPlus } from "lucide-react";
import { useMemo, useState } from "react";
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
  PageHeader,
  Select,
  Spinner,
  Table,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  Td,
  Th,
} from "@/components/ui";
import { canAdminister, type Role, useSession } from "@/features/auth/session";
import { useProjects } from "@/lib/api/hooks";
import { formatDate } from "@/lib/format";

import {
  type AuditFilters,
  type UserAccount,
  useAuditEvents,
  useCreateUser,
  useGrant,
  useMemberships,
  useRevoke,
  useUpdateUser,
  useUsers,
  useWorkspaces,
} from "./hooks";

const ROLES: Role[] = ["viewer", "editor", "admin"];

function CreateUserForm({ workspaceId }: { workspaceId: string | undefined }) {
  const { t } = useTranslation();
  const create = useCreateUser();
  const empty = { email: "", name: "", password: "", role: "viewer" as Role, admin: false };
  const [f, setF] = useState(empty);
  return (
    <form
      className="grid gap-3 sm:grid-cols-5"
      onSubmit={(e) => {
        e.preventDefault();
        create.mutate(
          {
            email: f.email,
            display_name: f.name,
            password: f.password,
            is_server_admin: f.admin,
            workspace_id: workspaceId ?? null,
            role: f.role,
          },
          { onSuccess: () => setF(empty) },
        );
      }}
    >
      <Field label={t("admin.email")}>
        <Input
          type="email"
          value={f.email}
          onChange={(e) => setF({ ...f, email: e.target.value })}
          required
        />
      </Field>
      <Field label={t("admin.name")}>
        <Input value={f.name} onChange={(e) => setF({ ...f, name: e.target.value })} />
      </Field>
      <Field label={t("admin.initialPassword")} hint={t("admin.passwordHint")}>
        <Input
          type="password"
          autoComplete="new-password"
          value={f.password}
          onChange={(e) => setF({ ...f, password: e.target.value })}
          required
        />
      </Field>
      <Field label={t("admin.workspaceRole")}>
        <Select value={f.role} onChange={(e) => setF({ ...f, role: e.target.value as Role })}>
          {ROLES.map((r) => (
            <option key={r} value={r}>
              {t(`roles.${r}`)}
            </option>
          ))}
        </Select>
      </Field>
      <div className="flex flex-col justify-end gap-2">
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={f.admin}
            onChange={(e) => setF({ ...f, admin: e.target.checked })}
          />
          {t("admin.serverAdmin")}
        </label>
        <Button type="submit" loading={create.isPending} disabled={!workspaceId}>
          <UserPlus className="h-4 w-4" aria-hidden="true" />
          {t("admin.createUser")}
        </Button>
      </div>
      <div className="sm:col-span-5">
        <ErrorNote error={create.error} />
      </div>
    </form>
  );
}

function UsersTab({ workspaceId }: { workspaceId: string | undefined }) {
  const { t, i18n } = useTranslation();
  const users = useUsers();
  const update = useUpdateUser();
  const resetPassword = (u: UserAccount) => {
    const password = window.prompt(t("admin.newPasswordFor", { email: u.user.email }));
    if (password) update.mutate({ userId: u.user.id, patch: { password } });
  };
  return (
    <div className="space-y-4">
      <Card>
        <CardTitle>{t("admin.newUser")}</CardTitle>
        <CreateUserForm workspaceId={workspaceId} />
      </Card>
      <Card>
        <CardTitle>{t("admin.users")}</CardTitle>
        {users.isPending && <Spinner />}
        <ErrorNote error={users.error ?? update.error} />
        {users.data && (
          <Table>
            <thead>
              <tr>
                <Th>{t("admin.email")}</Th>
                <Th>{t("admin.name")}</Th>
                <Th>{t("admin.status")}</Th>
                <Th>{t("admin.lastLogin")}</Th>
                <Th />
              </tr>
            </thead>
            <tbody>
              {users.data.map((u) => (
                <tr key={u.user.id}>
                  <Td>{u.user.email}</Td>
                  <Td>{u.user.display_name}</Td>
                  <Td className="space-x-1">
                    {u.is_server_admin && <Badge tone="brand">{t("admin.serverAdmin")}</Badge>}
                    {u.user.is_active === false ? (
                      <Badge tone="bad">{t("admin.inactive")}</Badge>
                    ) : (
                      <Badge tone="ok">{t("admin.active")}</Badge>
                    )}
                    {u.locked && <Badge tone="warn">{t("admin.locked")}</Badge>}
                  </Td>
                  <Td className="text-xs">
                    {u.last_login_at ? formatDate(u.last_login_at, i18n.language) : "—"}
                  </Td>
                  <Td className="space-x-2 whitespace-nowrap text-right">
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() =>
                        update.mutate({
                          userId: u.user.id,
                          patch: { is_active: u.user.is_active === false },
                        })
                      }
                    >
                      {u.user.is_active === false ? t("admin.activate") : t("admin.deactivate")}
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() =>
                        update.mutate({
                          userId: u.user.id,
                          patch: { is_server_admin: !u.is_server_admin },
                        })
                      }
                    >
                      {u.is_server_admin ? t("admin.removeAdmin") : t("admin.makeAdmin")}
                    </Button>
                    <Button size="sm" variant="ghost" onClick={() => resetPassword(u)}>
                      {t("admin.resetPassword")}
                    </Button>
                  </Td>
                </tr>
              ))}
            </tbody>
          </Table>
        )}
      </Card>
    </div>
  );
}

function RolesTab({ workspaces }: { workspaces: { id: string; name: string }[] }) {
  const { t } = useTranslation();
  const [workspaceId, setWorkspaceId] = useState(workspaces[0]?.id ?? "");
  const memberships = useMemberships(workspaceId || undefined);
  const users = useUsers();
  const projects = useProjects();
  const grant = useGrant();
  const revoke = useRevoke();
  const [userId, setUserId] = useState("");
  const [projectId, setProjectId] = useState("");
  const [role, setRole] = useState<Role>("viewer");

  const emailOf = useMemo(
    () => new Map((users.data ?? []).map((u) => [u.user.id, u.user.email])),
    [users.data],
  );
  const wsProjects = (Array.isArray(projects.data) ? projects.data : []).filter(
    (p) => p.workspace_id === workspaceId,
  );
  const projectName = new Map(wsProjects.map((p) => [p.id, p.name]));

  return (
    <Card>
      <CardTitle>{t("admin.roles")}</CardTitle>
      <p className="mb-4 text-sm text-muted">{t("admin.rolesHint")}</p>
      <div className="mb-4 grid gap-3 sm:grid-cols-5">
        <Field label={t("admin.workspace")}>
          <Select value={workspaceId} onChange={(e) => setWorkspaceId(e.target.value)}>
            {workspaces.map((w) => (
              <option key={w.id} value={w.id}>
                {w.name}
              </option>
            ))}
          </Select>
        </Field>
        <Field label={t("admin.user")}>
          <Select value={userId} onChange={(e) => setUserId(e.target.value)}>
            <option value="">—</option>
            {(users.data ?? []).map((u) => (
              <option key={u.user.id} value={u.user.id}>
                {u.user.email}
              </option>
            ))}
          </Select>
        </Field>
        <Field label={t("admin.scope")}>
          <Select value={projectId} onChange={(e) => setProjectId(e.target.value)}>
            <option value="">{t("admin.wholeWorkspace")}</option>
            {wsProjects.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </Select>
        </Field>
        <Field label={t("admin.role")}>
          <Select value={role} onChange={(e) => setRole(e.target.value as Role)}>
            {ROLES.map((r) => (
              <option key={r} value={r}>
                {t(`roles.${r}`)}
              </option>
            ))}
          </Select>
        </Field>
        <div className="flex items-end">
          <Button
            disabled={!userId || !workspaceId}
            loading={grant.isPending}
            onClick={() =>
              grant.mutate({
                user_id: userId,
                workspace_id: workspaceId,
                project_id: projectId || null,
                role,
              })
            }
          >
            <ShieldCheck className="h-4 w-4" aria-hidden="true" />
            {t("admin.grant")}
          </Button>
        </div>
      </div>
      <ErrorNote error={memberships.error ?? grant.error ?? revoke.error} />
      {memberships.isPending && <Spinner />}
      {memberships.data?.length === 0 && <EmptyState>{t("admin.noMembers")}</EmptyState>}
      {memberships.data && memberships.data.length > 0 && (
        <Table>
          <thead>
            <tr>
              <Th>{t("admin.user")}</Th>
              <Th>{t("admin.scope")}</Th>
              <Th>{t("admin.role")}</Th>
              <Th />
            </tr>
          </thead>
          <tbody>
            {memberships.data.map((m) => (
              <tr key={m.id}>
                <Td>{emailOf.get(m.user_id) ?? m.user_id}</Td>
                <Td>
                  {m.project_id
                    ? (projectName.get(m.project_id) ?? m.project_id)
                    : t("admin.wholeWorkspace")}
                </Td>
                <Td>
                  <Badge>{t(`roles.${m.role}`)}</Badge>
                </Td>
                <Td className="text-right">
                  <Button
                    size="sm"
                    variant="ghost"
                    aria-label={t("admin.revoke")}
                    onClick={() => revoke.mutate(m.id)}
                  >
                    <Trash2 className="h-4 w-4" aria-hidden="true" />
                  </Button>
                </Td>
              </tr>
            ))}
          </tbody>
        </Table>
      )}
    </Card>
  );
}

function AuditTab() {
  const { t, i18n } = useTranslation();
  const [filters, setFilters] = useState<AuditFilters>({});
  const events = useAuditEvents(filters);
  const users = useUsers();
  const emailOf = new Map((users.data ?? []).map((u) => [u.user.id, u.user.email]));
  return (
    <Card>
      <CardTitle>{t("admin.audit")}</CardTitle>
      <div className="mb-4 grid gap-3 sm:grid-cols-3">
        <Field label={t("admin.action")} hint={t("admin.actionHint")}>
          <Input
            value={filters.action ?? ""}
            onChange={(e) => setFilters({ ...filters, action: e.target.value })}
          />
        </Field>
        <Field label={t("admin.user")}>
          <Select
            value={filters.user_id ?? ""}
            onChange={(e) => setFilters({ ...filters, user_id: e.target.value })}
          >
            <option value="">{t("admin.anyone")}</option>
            {(users.data ?? []).map((u) => (
              <option key={u.user.id} value={u.user.id}>
                {u.user.email}
              </option>
            ))}
          </Select>
        </Field>
      </div>
      <ErrorNote error={events.error} />
      {events.isPending && <Spinner />}
      {events.data?.length === 0 && <EmptyState>{t("admin.noEvents")}</EmptyState>}
      {events.data && events.data.length > 0 && (
        <Table>
          <thead>
            <tr>
              <Th>{t("admin.when")}</Th>
              <Th>{t("admin.user")}</Th>
              <Th>{t("admin.action")}</Th>
              <Th>{t("admin.resource")}</Th>
              <Th>{t("admin.result")}</Th>
              <Th>IP</Th>
            </tr>
          </thead>
          <tbody>
            {events.data.map((e) => (
              <tr key={e.id}>
                <Td className="whitespace-nowrap text-xs">{formatDate(e.at, i18n.language)}</Td>
                <Td className="text-xs">
                  {e.user_id ? (emailOf.get(e.user_id) ?? e.user_id) : "—"}
                </Td>
                <Td className="font-mono text-xs">{e.action}</Td>
                <Td className="max-w-xs truncate text-xs">{e.resource ?? e.project_id ?? "—"}</Td>
                <Td>
                  {e.status !== null && e.status !== undefined && (
                    <Badge tone={e.status >= 400 ? "bad" : "ok"}>{e.status}</Badge>
                  )}
                </Td>
                <Td className="text-xs">{e.ip ?? "—"}</Td>
              </tr>
            ))}
          </tbody>
        </Table>
      )}
    </Card>
  );
}

export function AdminPage() {
  const { t } = useTranslation();
  const me = useSession();
  const serverAdmin = Boolean(me?.is_server_admin);
  const workspaces = useWorkspaces(serverAdmin);
  if (!me || !canAdminister(me)) return <EmptyState>{t("admin.forbidden")}</EmptyState>;
  // El Admin de un workspace gestiona los roles de los suyos; el del servidor, todo.
  const adminOf = serverAdmin
    ? (workspaces.data ?? [])
    : me.workspaces.filter((w) =>
        me.memberships.some((m) => m.workspace_id === w.id && !m.project_id && m.role === "admin"),
      );
  const list = adminOf.map((w) => ({ id: w.id ?? "", name: w.name }));
  return (
    <div>
      <PageHeader title={t("admin.title")} description={t("admin.description")} />
      <Tabs defaultValue={serverAdmin ? "users" : "roles"}>
        <TabsList>
          {serverAdmin && <TabsTrigger value="users">{t("admin.users")}</TabsTrigger>}
          <TabsTrigger value="roles">{t("admin.roles")}</TabsTrigger>
          {serverAdmin && <TabsTrigger value="audit">{t("admin.audit")}</TabsTrigger>}
        </TabsList>
        {serverAdmin && (
          <TabsContent value="users">
            <UsersTab workspaceId={list[0]?.id} />
          </TabsContent>
        )}
        <TabsContent value="roles">
          {serverAdmin && workspaces.isPending ? <Spinner /> : <RolesTab workspaces={list} />}
        </TabsContent>
        {serverAdmin && (
          <TabsContent value="audit">
            <AuditTab />
          </TabsContent>
        )}
      </Tabs>
    </div>
  );
}
