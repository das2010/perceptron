import { useQuery } from "@tanstack/react-query";

import { getApiClient } from "@/lib/api/client";

export function useEngineHealth() {
  return useQuery({
    queryKey: ["system", "health"],
    queryFn: async () => {
      const api = await getApiClient();
      const { data, error } = await api.GET("/api/v1/system/health");
      if (error || !data) throw new Error("engine unavailable");
      return data;
    },
  });
}
