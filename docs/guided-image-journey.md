# Recorrido guiado de clasificación de imágenes

Implementación de RF-WIZ-01/03/04, RF-ING-01/07/08 y RF-EXP-02 sobre el motor existente.

## Uso

1. En Inicio, describir el problema: «Quiero distinguir fotos con fisuras de fotos sin fisuras».
2. Responder la conversación o los controles de la siguiente pregunta. El asistente distingue
   una categoría por imagen de localizar con recuadros y segmentar píxeles. Revisar los hechos
   en «Lo que ya sabemos»; corregirlos y confirmar el resumen.
3. Subir un ZIP con `categoria/imagen.jpg`. Si varias fotos pertenecen al mismo objeto o sesión,
   usar `categoria/grupo/imagen.jpg`, reutilizando el nombre del grupo entre categorías.
   Resolver los bloqueos y volver a cargar una nueva versión. No se escriben los originales.
4. Elegir «Preparar y entrenar». Se puede salir, volver y detener el trabajo. No se muestran
   épocas, arquitecturas, trials ni porcentajes de progreso estimados.
5. Al finalizar aparece un aviso persistente. Revisar aciertos, falsas alarmas y casos no
   detectados para la categoría de interés. Entrenamiento terminado no significa objetivo cumplido.
6. «Probar una predicción» requiere solo una imagen. «Validar con casos conocidos» solicita además
   la categoría real, exclusivamente para comparar después de la inferencia. Se requieren al menos
   diez imágenes distintas, cobertura de todas las categorías y el porcentaje acordado tanto en
   prueba reservada como en validación nueva para pasar a `ready_to_use`. Es una comprobación
   exploratoria, no una certificación estadística ni garantía de funcionamiento.

Las herramientas anteriores continúan bajo «Opciones avanzadas». La antigua vista general vive
ahora en `/projects/:id/overview`; las URLs de datos, wizard, agente, entrenamiento y modelos se
conservan. Los proyectos existentes no requieren conversión y sus modelos siguen disponibles allí.

## Arquitectura y persistencia

- `services/journey.py`: `Coordinator`, `Facts`, `Journey` y reglas verificables de suficiencia.
  `JourneyRecord` es un documento en la tabla genérica `entities`, con `project_id` y `schema_version=1`.
  No requiere DDL ni migración destructiva. Está incluido en `ALL_ENTITIES` para paquetes,
  sincronización y ciclo de vida del proyecto. No modifica ni reutiliza el borrador avanzado.
- Cada acción exige versión y estado permitido. Las actualizaciones usan un `UPDATE` condicionado
  por versión (compare-and-swap), también en SQLite. La reserva durable precede al lanzamiento:
  doble clic/reintento obsoleto recibe 409 y no crea otro entrenamiento.
- La entrevista persiste el mensaje antes de llamar al gateway. El gateway aplica configuración,
  privacidad, esquema y auditoría habituales. La extracción solo registra campos respaldados por
  citas literales de la respuesta. Supuestos y preguntas pendientes están separados; el usuario
  puede corregir los hechos y debe confirmar el resumen antes de cualquier preparación.
- El LLM no controla transiciones. Sin proveedor o en L0, los controles esenciales siguen operativos.
  No se envían imágenes de validación al LLM. Las explicaciones de resultados son deterministas,
  basadas en la matriz calculada y comparaciones reales; no se inventan causas de errores.
- Preparación: subida existente, `Workflow.ingest`, `Workflow.profile`, manifiestos y dataset
  inmutable. El agrupamiento añade metadatos fuera del esquema predictor. Perfiles y selección
  usan train/val; la evaluación final reutiliza `Workflow.evaluate` y el test sellado.
- Entrenamiento: propuesta de pipeline y arquitectura por reglas, presupuesto de dos pruebas,
  hasta doce épocas por prueba y diez minutos para el estudio. `new_study` y `ctx.launch_study`
  preservan cola, cuotas y workers del servidor. Luego se evalúa, registra y exporta ONNX con
  verificación numérica. La UI consulta estados reales; no inventa duración ni avance porcentual.
- Se guardan etapa, referencia del estudio, jobs, PID y momento de creación del proceso. Al detectar
  una interrupción se conserva el contexto y se usa `study_control.mark_interrupted`; para workers
  remotos restaurados se solicita cancelación y se espera su final antes de habilitar reintento.
  El reintento crea un nuevo estudio; no promete reanudar una época interrumpida.
- `notification` se escribe al terminar, fallar o interrumpirse y permanece hasta que se reconoce.
- El contrato proviene de `ModelVersion.signature`; se contrasta con proyecto/run y con las claves
  del contrato del archivo exportado. `predictJourney` admite lectores; `validateJourney` y las
  mutaciones requieren editor. La categoría real nunca entra al pipeline o al modelo.
- Validación conserva hash, predicción y resultado real, no el archivo subido. Repetir una imagen
  reemplaza su comparación y no aumenta la muestra. Se rechazan archivos idénticos a los datos
  originales como casos nuevos.

## Extensión

Para otra modalidad, agregar su discriminante y requisitos a `Facts`, sus reglas de suficiencia,
preparación/revisión y el contrato de entrada generado por el motor. Mantener comunes la
persistencia, reserva, control de jobs, permisos, notificaciones y separación entre inferencia y
etiqueta real. Agregar controles que correspondan al contrato validado, nunca campos propuestos
por el LLM. Hasta contar con todo ese circuito, devolver `unsupported` con enlace al wizard
avanzado. No hay reglas que dependan de tubos, fisuras ni nombres particulares de categorías.

## Verificación y límites

`engine/tests/api/test_journey.py` cubre persistencia, fallback, extracción con evidencia,
transiciones, compare-and-swap concurrente, reconexión, interrupción, cancelación, permisos entre
proyectos, duplicados, grupos y contrato. Incluye un recorrido con 80 imágenes generadas, motor
real, una época, exportación ONNX e inferencia real (solo reduce el presupuesto para la prueba).
`server/tests/test_server_rbac.py` cubre permisos del recorrido. Las pruebas de React comprueban
restauración de conversación/aviso y separación de los formularios de predicción y validación.

Limitaciones de esta primera versión:

- La entrada guiada de datos es ZIP de carpetas de imágenes; otros formatos y modalidades siguen
  en el modo avanzado. La validación nueva es individual, no por lotes.
- Los hashes detectan duplicados exactos. Los perfiles advierten similitudes, pero no pueden
  certificar que dos fotos distintas sean de objetos independientes: depende del agrupamiento
  declarado y de la selección humana de casos nuevos.
- El objetivo cuantificado es porcentaje de aciertos. La preocupación por falsas alarmas o casos
  omitidos se registra y se informa; no se convierte automáticamente en un costo monetario ni en
  un umbral calibrado. Esas herramientas siguen disponibles en avanzado.
- El límite de diez minutos corresponde al estudio de entrenamiento; ingesta, evaluación y
  exportación tienen sus límites existentes, no un plazo único de extremo a extremo.
- El coordinador local se recupera mediante el proceso y los jobs de la misma instancia/host.
  No es un coordinador distribuido con leases entre varios hosts de API. La ejecución remota
  conserva el launcher del servidor, pero esta entrega no prueba un despliegue real con Celery.
- No se probó un proveedor de IA externo ni una sesión gráfica Tauri. El gateway se prueba con
  FakeLLMProvider y la inferencia/entrenamiento sí se ejecutan realmente.
- En Windows sin rutas largas habilitadas, pytest debe usar un directorio temporal corto.

Comprobaciones locales de esta entrega:

- 15 pruebas del recorrido (incluidas ZIP real, imágenes ilegibles, worker remoto pendiente,
  entrenamiento de una época, exportación e inferencia), ejecutadas en pasadas funcionales.
- RBAC del recorrido: prueba con editor, lector y usuario ajeno aprobada.
- Regresión de ingesta, replay de jobs y paquetes: pasada de 37 pruebas aprobadas
  (incluye las pruebas del recorrido disponibles en ese incremento, excepto entrenamiento largo).
- React: 10 pruebas de entrada, idioma, conversación persistente y separación de formularios.
- TypeScript, Ruff y mypy de los módulos modificados: sin errores. Build de producción aprobado.
- Advertencias no bloqueantes: chunks grandes en Vite, deprecación SQLAlchemy/MLflow,
  `scrollTo` no implementado por JSDOM y advertencia preexistente de Fast Refresh en ProjectLayout.

En esta máquina se usó `uv run python -m ...` y `node .../bin/...` porque los lanzadores
`perceptron-server` y `pnpm` locales no funcionaban correctamente. OpenAPI se regeneró desde
`perceptron_server.cli.openapi` y sus tipos con `openapi-typescript`.
