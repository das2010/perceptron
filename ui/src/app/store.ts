import { create } from "zustand";

/** Estado de UI local (Zustand, SPEC §5.2); los datos del Engine viven en TanStack Query. */
interface UiState {
  copilotOpen: boolean;
  toggleCopilot: () => void;
  /** Pregunta pendiente para el copiloto (p. ej. el "¿Por qué?" de un paso del wizard). */
  copilotQuestion: string | null;
  askCopilot: (question: string) => void;
  takeCopilotQuestion: () => string | null;
}

export const useUiStore = create<UiState>((set, get) => ({
  copilotOpen: false,
  toggleCopilot: () => set((s) => ({ copilotOpen: !s.copilotOpen })),
  copilotQuestion: null,
  askCopilot: (question) => set({ copilotOpen: true, copilotQuestion: question }),
  takeCopilotQuestion: () => {
    const q = get().copilotQuestion;
    if (q !== null) set({ copilotQuestion: null });
    return q;
  },
}));
