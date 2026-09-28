import { create } from "zustand";

/** Estado de UI local (Zustand, SPEC §5.2); los datos del Engine viven en TanStack Query. */
interface UiState {
  copilotOpen: boolean;
  toggleCopilot: () => void;
}

export const useUiStore = create<UiState>((set) => ({
  copilotOpen: false,
  toggleCopilot: () => set((s) => ({ copilotOpen: !s.copilotOpen })),
}));
