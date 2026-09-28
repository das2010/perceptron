/** Sesión del Team Server (RF-SRV-01): configuración de login, usuario actual y logout. */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { createContext, useContext } from "react";

import { getApiClient } from "@/lib/api/client";
import { ApiError, type Schemas, unwrap } from "@/lib/api/hooks";

export type Me = Schemas["Me"];
export type AuthConfig = Schemas["AuthConfig"];
export type Membership = Schemas["Membership"] & { id: string };
export type Role = Membership["role"];

export const authKeys = {
  config: ["auth", "config"] as const,
  me: ["auth", "me"] as const,
};

/** `null` = el Engine no es un Team Server (desktop o `perceptron serve` en desarrollo). */
export function useAuthConfig(enabled: boolean) {
  return useQuery({
    queryKey: authKeys.config,
    enabled,
    staleTime: Infinity,
    retry: false,
    queryFn: async (): Promise<AuthConfig | null> => {
      const res = await (await getApiClient()).GET("/api/v1/auth/config");
      if (res.response.status === 404) return null;
      return unwrap(res);
    },
  });
}

/** Usuario actual; `null` si no hay sesión (401). */
export function useMe(enabled: boolean) {
  return useQuery({
    queryKey: authKeys.me,
    enabled,
    retry: false,
    queryFn: async (): Promise<Me | null> => {
      const res = await (await getApiClient()).GET("/api/v1/auth/me");
      if (res.response.status === 401) return null;
      return unwrap(res);
    },
  });
}

export function useLogin() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: Schemas["Credentials"]) =>
      unwrap(await (await getApiClient()).POST("/api/v1/auth/login", { body })),
    onSuccess: (me) => {
      qc.clear();
      qc.setQueryData(authKeys.me, me);
    },
  });
}

export function useLogout() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async () => {
      const res = await (await getApiClient()).POST("/api/v1/auth/logout");
      if (!res.response.ok && res.response.status !== 401)
        throw new ApiError("logout", "http_error", res.response.status);
    },
    onSettled: () => {
      qc.clear();
      qc.setQueryData(authKeys.me, null);
    },
  });
}

export function useChangePassword() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: Schemas["PasswordChange"]) => {
      const res = await (await getApiClient()).POST("/api/v1/auth/password", { body });
      if (!res.response.ok) unwrap(res);
    },
    // Cambiar la contraseña cierra todas las sesiones: vuelve al login.
    onSuccess: () => qc.setQueryData(authKeys.me, null),
  });
}

/** Sesión activa en el Team Server; `null` en el desktop. */
export const SessionContext = createContext<Me | null>(null);

export function useSession(): Me | null {
  return useContext(SessionContext);
}

const RANK: Record<Role, number> = { viewer: 1, editor: 2, admin: 3 };

/** Admin del servidor o de algún workspace: ve la consola de administración. */
export function canAdminister(me: Me | null): boolean {
  if (!me) return false;
  return me.is_server_admin || me.memberships.some((m) => !m.project_id && m.role === "admin");
}

/** Rol efectivo en un proyecto (la membresía del proyecto manda sobre la del workspace). */
export function projectRole(
  me: Me | null,
  project: { id: string; workspace_id?: string | null },
): Role | null {
  if (!me) return "admin"; // desktop: un solo usuario local
  if (me.is_server_admin) return "admin";
  const own = me.memberships.find((m) => m.project_id === project.id);
  if (own) return own.role;
  const ws = me.memberships.find((m) => !m.project_id && m.workspace_id === project.workspace_id);
  return ws?.role ?? null;
}

export function atLeast(role: Role | null, needed: Role): boolean {
  return role !== null && RANK[role] >= RANK[needed];
}
