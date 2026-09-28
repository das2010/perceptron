import { RouterProvider } from "@tanstack/react-router";
import { useState } from "react";

import { createAppRouter, type AppRouter } from "./router";

export function App({ router }: { router?: AppRouter }) {
  const [instance] = useState(() => router ?? createAppRouter());
  return <RouterProvider router={instance} />;
}
