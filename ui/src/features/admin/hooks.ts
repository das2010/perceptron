/** Consola de administración del Team Server (RF-SRV-06): usuarios, roles y auditoría. */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { getApiClient } from "@/lib/api/client";
import { type Schemas, unwrap } from "@/lib/api/hooks";

export type UserAccount = Schemas["UserAccount"] & { user: { id: string } };
export type Workspace = Schemas["Workspace"] & { id: string };
export type Membership = Schemas["Membership"] & { id: string };
export type AuditEvent = Schemas["AuditEvent"];
export type AuditFilters = { action?: string; user_id?: string; project_id?: string };

const adminKeys = {
  users: ["admin", "users"] as const,
  workspaces: ["admin", "workspaces"] as const,
  memberships: (ws: string) => ["admin", "memberships", ws] as const,
  audit: (f: AuditFilters) => ["admin", "audit", f] as const,
};

export function useUsers(enabled = true) {
  return useQuery({
    queryKey: adminKeys.users,
    enabled,
    queryFn: async () =>
      unwrap(await (await getApiClient()).GET("/api/v1/admin/users")) as UserAccount[],
  });
}

export function useCreateUser() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: Schemas["UserCreate"]) =>
      unwrap(await (await getApiClient()).POST("/api/v1/admin/users", { body })),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: adminKeys.users });
      void qc.invalidateQueries({ queryKey: ["admin", "memberships"] });
    },
  });
}

export function useUpdateUser() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ userId, patch }: { userId: string; patch: Schemas["UserPatch"] }) =>
      unwrap(
        await (
          await getApiClient()
        ).PATCH("/api/v1/admin/users/{user_id}", {
          params: { path: { user_id: userId } },
          body: patch,
        }),
      ),
    onSuccess: () => void qc.invalidateQueries({ queryKey: adminKeys.users }),
  });
}

export function useWorkspaces(enabled = true) {
  return useQuery({
    queryKey: adminKeys.workspaces,
    enabled,
    queryFn: async () =>
      unwrap(await (await getApiClient()).GET("/api/v1/admin/workspaces")) as Workspace[],
  });
}

export function useMemberships(workspaceId: string | undefined) {
  return useQuery({
    queryKey: adminKeys.memberships(workspaceId ?? ""),
    enabled: Boolean(workspaceId),
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/admin/memberships", {
          params: { query: { workspace_id: workspaceId ?? null } },
        }),
      ) as Membership[],
  });
}

export function useGrant() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: Schemas["MembershipCreate"]) =>
      unwrap(await (await getApiClient()).POST("/api/v1/admin/memberships", { body })),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ["admin", "memberships"] }),
  });
}

export function useRevoke() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (membershipId: string) => {
      const res = await (
        await getApiClient()
      ).DELETE("/api/v1/admin/memberships/{membership_id}", {
        params: { path: { membership_id: membershipId } },
      });
      if (!res.response.ok) unwrap(res);
    },
    onSuccess: () => void qc.invalidateQueries({ queryKey: ["admin", "memberships"] }),
  });
}

export function useAuditEvents(filters: AuditFilters, enabled = true) {
  return useQuery({
    queryKey: adminKeys.audit(filters),
    enabled,
    queryFn: async () =>
      unwrap(
        await (
          await getApiClient()
        ).GET("/api/v1/admin/audit", {
          params: {
            query: {
              action: filters.action || null,
              user_id: filters.user_id || null,
              project_id: filters.project_id || null,
              limit: 200,
            },
          },
        }),
      ),
  });
}

export function useUpdateWorkspacePolicy() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({
      workspaceId,
      policy,
    }: {
      workspaceId: string;
      policy: Schemas["WorkspacePolicy"];
    }) =>
      unwrap(
        await (
          await getApiClient()
        ).PATCH("/api/v1/admin/workspaces/{workspace_id}", {
          params: { path: { workspace_id: workspaceId } },
          body: policy,
        }),
      ),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: adminKeys.workspaces });
      void qc.invalidateQueries({ queryKey: ["auth", "me"] });
    },
  });
}
