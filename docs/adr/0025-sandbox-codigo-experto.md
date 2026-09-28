# ADR-0025: Sandbox del código experto (RF-ARC-06)
- Estado: aceptado
- Fecha: 2026-09-28
- Contexto: RF-ARC-06 permite que el usuario (o el LLM) escriba el modelo como código Python con una interfaz fija. SPEC §13.2 pide ejecutarlo en un sandbox con estas capas: proceso separado; sin red (bloqueo en el proceso + monkeypatch de sockets); lista blanca de imports con validación AST previa; límites de CPU, RAM y tiempo (Job Objects en Windows, rlimits en Linux); filesystem restringido al directorio del run; y confirmación explícita con advertencia visible. En el servidor se suma un contenedor efímero (Capa 5). El Engine nunca debe ejecutar ese código en su propio proceso.
- Decisión:
  - **Representación.** Una ArchSpec de código tiene un solo nodo del bloque `code.module`, con `params.code_sha256`, que hace que el `content_hash` cambie con el código. El fuente se guarda en `projects/<id>/archspecs/<record>.py` y se referencia con `ArchSpecRecord.code_path`. La interfaz es `build_model(config) -> nn.Module`, más `input_spec` y `output_spec` opcionales (dicts) que se contrastan con los datos. `config` trae `input` (el `InputSpec`), `num_outputs`, `task` y los hiperparámetros resueltos. El módulo recibe los tensores de la modalidad y devuelve la salida con la forma de la cabeza.
  - **Validación en dos fases.**
    1. Estática, en el Engine y sin ejecutar nada: parseo AST, tamaño máximo e imports solo de una lista blanca (`torch`, `torch.nn`, `torch.nn.functional`, `math`, `typing`, `dataclasses`, `collections`, `functools`, `itertools`, `numbers`). Se prohíben los nombres peligrosos (`eval`, `exec`, `compile`, `open`, `__import__`, `globals`, `getattr`/`setattr`/`delattr`, `vars`, `input`, `breakpoint`) y todo atributo *dunder* salvo `__init__` y `__name__`.
    2. Dinámica, en el sandbox: construye el modelo y hace un forward sobre un batch sintético para obtener la cantidad de parámetros y la forma de la salida.
  - **Proceso sandbox.** Se lanza `python -I` (modo aislado: sin `PYTHON*` ni site del usuario), con un entorno mínimo (sin proxies, tokens ni claves) y el directorio del run como `cwd` y `TMP`. Antes de cargar el código del usuario, el proceso:
    - importa torch, Lightning y el pipeline;
    - instala un *audit hook* (`sys.addaudithook`, no removible) que corta sockets, `subprocess`, `os.system`, `exec`/`spawn`/`fork` y `ctypes`;
    - solo permite escribir dentro del run y leer dentro del run, del dataset y del runtime de Python;
    - reemplaza `socket.socket` por una versión que falla.

    Los DataLoaders usan `num_workers=0`, sin procesos hijos.
  - **Límites del SO.**
    - Linux: `RLIMIT_AS` (RAM), `RLIMIT_CPU` y `RLIMIT_FSIZE` en `preexec_fn`.
    - Windows: un Job Object (vía `ctypes` en el proceso padre) con límite de memoria por proceso, un único proceso activo (sin hijos), tiempo de CPU y *kill on job close*.
    - Tiempo real: el supervisor mata el proceso al vencer el plazo.
  - **Confirmación.** Guardar o entrenar código exige `acknowledge_risk: true` en la API. El run queda marcado como `declarative: false` y la UI muestra la advertencia. El código que propone el LLM pasa por el mismo camino y además requiere que el usuario lo acepte (§7.7).
- Consecuencias:
  - Los *audit hooks* de CPython no son una frontera de seguridad frente a un atacante decidido: la documentación de Python lo advierte. En el desktop, estas capas evitan que un código accidental o inyectado (por ejemplo, vía el LLM) use la red, lance procesos o toque archivos fuera del run. La frontera fuerte del servidor es el contenedor efímero (Capa 5).
  - El modo experto no tiene validación de *shapes* en el Engine: la da la fase dinámica.
  - `to_code` devuelve el fuente tal cual.
- Alternativas consideradas:
  - Ejecutar en el proceso del Engine: descartado por §13.2.
  - RestrictedPython: su licencia ZPL es compatible, pero reescribe el código y rompe PyTorch idiomático.
  - WebAssembly/Pyodide: no corre PyTorch nativo.
  - Contenedores en el desktop: exige Docker, que el usuario puede no tener.
