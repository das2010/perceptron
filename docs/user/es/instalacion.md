# Instalación

## Desktop

### Requisitos

- **Windows 10/11 x64** o **Linux x64** (Ubuntu 22.04+ o Debian 12+).
- No necesitás tener Python instalado: la app trae su propio runtime.
- **Conexión a internet en el primer arranque** (se descargan Python, las dependencias y PyTorch).
- Espacio en disco: reservá varios GB; la variante de PyTorch para GPU (CUDA/ROCm) es la más pesada.
- GPU opcional. Sin GPU se entrena en CPU. Para NVIDIA necesitás el driver instalado; ROCm (AMD) solo
  está disponible en Linux.

### Instalar

| Sistema | Instalador |
|---|---|
| Windows | `.exe` (NSIS) o `.msi` |
| Linux | `.deb` (Debian/Ubuntu) o `AppImage` |

1. Ejecutá el instalador (en Linux: `sudo apt install ./Perceptron_*.deb`, o marcá el AppImage como
   ejecutable y abrilo).
2. Abrí **Perceptron** desde el menú de inicio o de aplicaciones.

!!! warning "Instaladores sin firma (por ahora)"
    La firma de código de los instaladores y la actualización automática firmada llegan
    **próximamente**. Mientras tanto, Windows SmartScreen puede advertir que el editor es
    desconocido: confirmá que el archivo vino de Preteco antes de continuar.

### Primer arranque: «Preparando Perceptron»

La primera vez la app prepara su entorno. Vas a ver los pasos **Python → Entorno → Dependencias →
Perceptron Engine → Detección del hardware → PyTorch**. Perceptron detecta tu hardware (GPU NVIDIA,
AMD, Intel o solo CPU) e instala la variante de PyTorch que corresponde. Puede tardar varios minutos.

- Si algo falla (por ejemplo, sin conexión o con un proxy que bloquea PyPI), tocá **Reintentar**.
- Los arranques siguientes son rápidos y no necesitan internet.

Cuando termina, la pantalla de **Inicio** muestra **Engine operativo** y la tarjeta **Hardware** con el
**Dispositivo recomendado**, CPU, memoria y GPU. Si no hay GPU vas a ver «Sin GPU (se entrena en CPU)».

### Dónde guarda las cosas

| Qué | Dónde (por defecto) | Cómo cambiarlo |
|---|---|---|
| Proyectos, datos y modelos | Windows: `%LOCALAPPDATA%\Perceptron` · Linux: `~/.local/share/perceptron` | Variable de entorno `PERCEPTRON_WORKSPACE_DIR` |
| Runtime de Python y PyTorch | Directorio de datos de la app (`com.preteco.perceptron/runtime`) | Variable de entorno `PERCEPTRON_RUNTIME_DIR` |
| Claves (LLM, bases de datos, tokens) | Llavero del sistema (Credential Manager / Secret Service) | — |

`PERCEPTRON_RUNTIME_DIR` sirve en equipos corporativos que bloquean ejecutables dentro de
`%LOCALAPPDATA%`. Las rutas con espacios y acentos (por ejemplo, carpetas de OneDrive) funcionan.

### Cambiar la variante de PyTorch (GPU)

Si agregaste una GPU, cambiaste el driver o la detección no eligió lo que querías:

1. Andá a **Configuración → Motor local (PyTorch)**. Ahí ves la **Variante instalada**, el **Índice de
   PyTorch** y el **Driver NVIDIA** detectado.
2. En **Cambiar variante** elegí **CPU**, **NVIDIA CUDA**, **AMD ROCm (Linux)** o **Intel XPU**.
3. Tocá **Aplicar**. Se reinstala solo PyTorch y se reinicia el motor; la app no se reinstala.

Si una variante de GPU no arranca, volvé a **CPU** con el mismo procedimiento.

### Próximos pasos

- Configurá un proveedor de LLM (opcional): [LLM y privacidad](llm-y-privacidad.md). Sin LLM la app
  funciona igual, con recomendaciones por reglas.
- Si tu equipo tiene un Team Server, conectalo desde **Configuración → Servidores de equipo**: ver
  [Team Server](team-server.md).
- Entrená tu primer modelo: [Tu primer modelo](primer-modelo.md).

### Actualizar y desinstalar

- Instalar una versión nueva encima de la anterior conserva tus proyectos.
- Tus proyectos viven en la carpeta de proyectos (ver la tabla de arriba), separada de la app. Antes de
  desinstalar, hacé una copia de esa carpeta si querés conservarlos.

## Team Server

El Team Server se despliega con **Docker Compose** (un servidor) o **Helm** (Kubernetes). La imagen
trae la API, la UI web y los workers de la cola; la variante de PyTorch (CPU, CUDA, ROCm o XPU) se
elige al construirla. Esta tarea es del Admin de plataforma:

- [Despliegue del Team Server](https://github.com/das2010/perceptron/blob/main/server/deploy/README.md)
- [SSO con Entra ID / Google Workspace](https://github.com/das2010/perceptron/blob/main/docs/ops/sso.md)
- [Backups y restauración](https://github.com/das2010/perceptron/blob/main/docs/ops/backups.md)

Una vez desplegado, los usuarios entran con el navegador a la URL del servidor. Ver
[Team Server](team-server.md).

## Si algo no anda

| Síntoma | Qué probar |
|---|---|
| «No se pudo conectar con el Engine» | Tocá **Reintentar** en *Estado del Engine*; si persiste, reiniciá la app |
| El primer arranque falla en «Dependencias» o «PyTorch» | Verificá la conexión y el proxy corporativo (PyPI y el índice de PyTorch) y tocá **Reintentar** |
| La GPU no aparece en **Hardware** | Revisá el driver; después cambiá la variante en **Configuración → Motor local (PyTorch)** |
| Tu empresa bloquea ejecutables en `%LOCALAPPDATA%` | Definí `PERCEPTRON_RUNTIME_DIR` apuntando a una carpeta permitida |
