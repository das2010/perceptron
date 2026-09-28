/** Problema a marcar en el editor de código (línea 1-based, columna 0-based). */
export interface CodeMarker {
  line: number;
  col: number;
  message: string;
}
