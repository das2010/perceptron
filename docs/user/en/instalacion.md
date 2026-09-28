# Installation

## Desktop

### Requirements

- **Windows 10/11 x64** or **Linux x64** (Ubuntu 22.04+ or Debian 12+).
- You don't need Python installed: the app ships its own runtime.
- **Internet connection on first launch** (Python, the dependencies and PyTorch are downloaded).
- Disk space: set aside several GB; the GPU build of PyTorch (CUDA/ROCm) is the largest.
- GPU optional. Without a GPU, training runs on the CPU. NVIDIA needs its driver installed; ROCm (AMD)
  is only available on Linux.

### Install

| System | Installer |
|---|---|
| Windows | `.exe` (NSIS) or `.msi` |
| Linux | `.deb` (Debian/Ubuntu) or `AppImage` |

1. Run the installer (on Linux: `sudo apt install ./Perceptron_*.deb`, or make the AppImage executable
   and open it).
2. Open **Perceptron** from the start or applications menu.

!!! warning "Unsigned installers (for now)"
    Code signing of the installers and signed automatic updates are **coming soon**. Meanwhile,
    Windows SmartScreen may warn that the publisher is unknown: make sure the file came from Preteco
    before continuing.

### First launch: "Getting Perceptron ready"

The first time, the app prepares its environment. You will see the steps **Python → Environment →
Dependencies → Perceptron Engine → Hardware detection → PyTorch**. Perceptron detects your hardware
(NVIDIA, AMD or Intel GPU, or CPU only) and installs the matching PyTorch build. It can take several
minutes.

- If something fails (for example, no connection or a proxy blocking PyPI), click **Retry**.
- Later launches are fast and don't need internet.

When it finishes, the **Home** screen shows **Engine is running** and the **Hardware** card with the
**Recommended device**, CPU, memory and GPU. Without a GPU you will see "No GPU (training on CPU)".

### Where things are stored

| What | Where (default) | How to change it |
|---|---|---|
| Projects, data and models | Windows: `%LOCALAPPDATA%\Perceptron` · Linux: `~/.local/share/perceptron` | Environment variable `PERCEPTRON_WORKSPACE_DIR` |
| Python runtime and PyTorch | App data directory (`com.preteco.perceptron/runtime`) | Environment variable `PERCEPTRON_RUNTIME_DIR` |
| Keys (LLM, databases, tokens) | System keychain (Credential Manager / Secret Service) | — |

`PERCEPTRON_RUNTIME_DIR` helps on corporate machines that block executables inside `%LOCALAPPDATA%`.
Paths with spaces and accents (for example, OneDrive folders) work.

### Changing the PyTorch build (GPU)

If you added a GPU, changed the driver, or detection didn't pick what you wanted:

1. Go to **Settings → Local engine (PyTorch)**. It shows the **Installed build**, the **PyTorch index**
   and the detected **NVIDIA driver**.
2. Under **Change build**, pick **CPU**, **NVIDIA CUDA**, **AMD ROCm (Linux)** or **Intel XPU**.
3. Click **Apply**. Only PyTorch is reinstalled and the engine restarts; the app is not reinstalled.

If a GPU build does not start, switch back to **CPU** the same way.

### Next steps

- Configure an LLM provider (optional): [LLM and privacy](llm-y-privacidad.md). Without an LLM the app
  works the same, with rule-based recommendations.
- If your team has a Team Server, connect it from **Settings → Team servers**: see
  [Team Server](team-server.md).
- Train your first model: [Your first model](primer-modelo.md).

### Updating and uninstalling

- Installing a new version over the previous one keeps your projects.
- Your projects live in the projects folder (see the table above), separate from the app. Before
  uninstalling, back up that folder if you want to keep them.

## Team Server

The Team Server is deployed with **Docker Compose** (single server) or **Helm** (Kubernetes). The image
includes the API, the web UI and the queue workers; the PyTorch build (CPU, CUDA, ROCm or XPU) is chosen
when building it. This is a platform admin task:

- [Team Server deployment](https://github.com/das2010/perceptron/blob/main/server/deploy/README.md)
- [SSO with Entra ID / Google Workspace](https://github.com/das2010/perceptron/blob/main/docs/ops/sso.md)
- [Backups and restore](https://github.com/das2010/perceptron/blob/main/docs/ops/backups.md)

Once deployed, users open the server URL in a browser. See [Team Server](team-server.md).

## Troubleshooting

| Symptom | What to try |
|---|---|
| "Could not connect to the Engine" | Click **Retry** in *Engine status*; if it persists, restart the app |
| First launch fails at "Dependencies" or "PyTorch" | Check the connection and the corporate proxy (PyPI and the PyTorch index) and click **Retry** |
| The GPU doesn't show up under **Hardware** | Check the driver; then change the build in **Settings → Local engine (PyTorch)** |
| Your company blocks executables in `%LOCALAPPDATA%` | Set `PERCEPTRON_RUNTIME_DIR` to an allowed folder |
