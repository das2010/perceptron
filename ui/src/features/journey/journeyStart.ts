/** Lo que la persona escribe o adjunta en Inicio y el asistente recibe al abrir el proyecto. */

/** El primer mensaje: se envía al abrir el proyecto, con «escribiendo…». */
export const FIRST_MESSAGE = (projectId: string) => `perceptron.journey.first.${projectId}`;

/** Los archivos adjuntados (en memoria: la navegación no recarga la página). */
export const FIRST_FILES = new Map<string, File[]>();
