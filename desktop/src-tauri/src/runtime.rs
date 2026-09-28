//! Runtime Python embebido (RF-TRN-02, ADR-0026).
//!
//! En el primer arranque (o al cambiar la versión de la app) instala con `uv`, dentro del
//! directorio de datos de la app: Python 3.12, un venv con las dependencias fijadas del
//! Engine, la wheel del Engine y la variante de PyTorch que corresponde al hardware.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub const PYTHON_VERSION: &str = "3.12";
const HARDWARE_PROBE: &str = "import json\n\
from perceptron.training.hardware import detect_hardware\n\
r = detect_hardware()\n\
print(json.dumps({'variant': r.recommended_torch_variant, 'nvidia_driver': r.nvidia_driver}))\n";

#[derive(Clone, Debug, Serialize)]
pub struct Progress {
    pub step: String,
    pub message: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct RuntimeState {
    pub app_version: String,
    pub torch_variant: String,
    pub torch_index: String,
    pub nvidia_driver: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Hardware {
    variant: String,
    nvidia_driver: Option<String>,
}

/// `torch-indexes.json`: índice de PyTorch por variante (ADR-0026).
#[derive(Debug, Deserialize)]
pub struct TorchIndexes {
    pub base_url: String,
    pub cpu: String,
    pub cuda: Vec<CudaIndex>,
    pub rocm: String,
    pub xpu: String,
}

#[derive(Debug, Deserialize)]
pub struct CudaIndex {
    pub min_driver: u32,
    pub path: String,
}

/// `torch.json`: versiones de torch/torchvision fijadas en `uv.lock`.
#[derive(Debug, Deserialize)]
pub struct TorchVersions {
    pub torch: String,
    pub torchvision: String,
}

impl TorchIndexes {
    /// URL del índice para una variante; CUDA según la versión del driver NVIDIA.
    pub fn url_for(&self, variant: &str, driver: Option<&str>, linux: bool) -> (String, String) {
        let path = match variant {
            "cuda" => {
                let major = driver
                    .and_then(|d| d.split('.').next())
                    .and_then(|m| m.trim().parse::<u32>().ok())
                    .unwrap_or(0);
                match self.cuda.iter().find(|c| major >= c.min_driver) {
                    Some(c) => c.path.clone(),
                    None => return ("cpu".into(), self.join(&self.cpu)),
                }
            }
            "rocm" if linux => self.rocm.clone(),
            "xpu" => self.xpu.clone(),
            _ => return ("cpu".into(), self.join(&self.cpu)),
        };
        (variant.to_string(), self.join(&path))
    }

    fn join(&self, path: &str) -> String {
        format!("{}/{}", self.base_url.trim_end_matches('/'), path.trim_start_matches('/'))
    }
}

pub struct Runtime {
    pub root: PathBuf,
    resources: PathBuf,
}

fn exe(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    }
}

/// Sin ventana de consola en Windows para los procesos hijos.
pub fn hide_console(cmd: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    #[cfg(not(windows))]
    let _ = cmd;
}

fn run(mut cmd: Command, what: &str) -> Result<String, String> {
    hide_console(&mut cmd);
    let out = cmd
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("{what}: no se pudo ejecutar ({e})"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        let tail: String = stderr.chars().rev().take(2000).collect::<String>().chars().rev().collect();
        return Err(format!("{what} falló ({}): {tail}", out.status));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

impl Runtime {
    /// `PERCEPTRON_RUNTIME_DIR` permite reubicar el runtime (equipos corporativos, ADR-0026).
    pub fn new(app_data: &Path, resource_dir: &Path) -> Self {
        let root = std::env::var_os("PERCEPTRON_RUNTIME_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| app_data.join("runtime"));
        Self { root, resources: resource_dir.join("resources") }
    }

    fn uv(&self) -> PathBuf {
        self.resources.join("bin").join(exe("uv"))
    }

    fn venv(&self) -> PathBuf {
        self.root.join("venv")
    }

    pub fn python(&self) -> PathBuf {
        if cfg!(windows) {
            self.venv().join("Scripts").join("python.exe")
        } else {
            self.venv().join("bin").join("python")
        }
    }

    pub fn logs_dir(&self) -> PathBuf {
        self.root.join("logs")
    }

    fn state_path(&self) -> PathBuf {
        self.root.join("state.json")
    }

    pub fn state(&self) -> Option<RuntimeState> {
        let raw = fs::read_to_string(self.state_path()).ok()?;
        serde_json::from_str(&raw).ok()
    }

    fn uv_cmd(&self) -> Command {
        let mut cmd = Command::new(self.uv());
        cmd.env("UV_CACHE_DIR", self.root.join("cache"))
            .env("UV_PYTHON_INSTALL_DIR", self.root.join("python"))
            .env("UV_PYTHON_PREFERENCE", "only-managed")
            .env("UV_NO_CONFIG", "1")
            .env_remove("VIRTUAL_ENV");
        cmd
    }

    fn read_json<T: for<'de> Deserialize<'de>>(&self, name: &str) -> Result<T, String> {
        let path = self.resources.join(name);
        let raw = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        serde_json::from_str(&raw).map_err(|e| format!("{}: {e}", path.display()))
    }

    fn wheel(&self) -> Result<PathBuf, String> {
        let dir = self.resources.join("wheels");
        fs::read_dir(&dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .find(|p| p.extension().is_some_and(|x| x == "whl"))
            .ok_or_else(|| format!("no hay wheel del Engine en {}", dir.display()))
    }

    /// Deja el runtime listo. `variant` fuerza una variante de torch (cambio desde la UI).
    pub fn provision(
        &self,
        app_version: &str,
        variant: Option<&str>,
        emit: &dyn Fn(Progress),
    ) -> Result<RuntimeState, String> {
        let say = |step: &str, message: &str| {
            emit(Progress { step: step.into(), message: message.into() })
        };
        let current = self.state();
        let ready = self.python().is_file()
            && current.as_ref().is_some_and(|s| s.app_version == app_version);
        if ready && variant.is_none() {
            return Ok(current.unwrap_or_default());
        }
        fs::create_dir_all(&self.root).map_err(|e| format!("{}: {e}", self.root.display()))?;
        let python = self.python();

        if !ready {
            say("python", "Instalando Python");
            let mut cmd = self.uv_cmd();
            cmd.args(["python", "install", PYTHON_VERSION]);
            run(cmd, "uv python install")?;

            say("venv", "Creando el entorno");
            let mut cmd = self.uv_cmd();
            cmd.args(["venv", "--clear", "--python", PYTHON_VERSION]).arg(self.venv());
            run(cmd, "uv venv")?;

            // El lock exportado es el cierre completo sin torch: `--no-deps` evita que uv
            // resuelva torch desde PyPI como dependencia de lightning/timm (en Linux sería la
            // build CUDA de >2 GB). Torch se instala después, desde el índice de la variante.
            say("deps", "Instalando dependencias");
            let mut cmd = self.uv_cmd();
            cmd.args(["pip", "install", "--no-deps", "--python"])
                .arg(&python)
                .arg("-r")
                .arg(self.resources.join("requirements.lock.txt"));
            run(cmd, "uv pip install (dependencias)")?;

            say("engine", "Instalando Perceptron Engine");
            let mut cmd = self.uv_cmd();
            cmd.args(["pip", "install", "--no-deps", "--reinstall", "--python"])
                .arg(&python)
                .arg(self.wheel()?);
            run(cmd, "uv pip install (engine)")?;
        }

        say("hardware", "Detectando el hardware");
        let mut probe = Command::new(&python);
        probe.args(["-c", HARDWARE_PROBE]);
        let hw_out = run(probe, "detección de hardware")?;
        let hw: Hardware = serde_json::from_str(hw_out.trim().lines().last().unwrap_or(""))
            .map_err(|e| format!("detección de hardware: {e} ({hw_out})"))?;
        let wanted = variant.unwrap_or(&hw.variant);
        let indexes: TorchIndexes = self.read_json("torch-indexes.json")?;
        let versions: TorchVersions = self.read_json("torch.json")?;
        let (chosen, index) =
            indexes.url_for(wanted, hw.nvidia_driver.as_deref(), cfg!(target_os = "linux"));

        say("torch", &format!("Instalando PyTorch ({chosen})"));
        let mut cmd = self.uv_cmd();
        cmd.args(["pip", "install", "--python"])
            .arg(&python)
            .args(["--reinstall-package", "torch", "--reinstall-package", "torchvision"])
            .args(["--index-url", &index])
            .arg(format!("torch=={}", versions.torch))
            .arg(format!("torchvision=={}", versions.torchvision));
        run(cmd, "uv pip install (torch)")?;

        let state = RuntimeState {
            app_version: app_version.to_string(),
            torch_variant: chosen,
            torch_index: index,
            nvidia_driver: hw.nvidia_driver,
        };
        let raw = serde_json::to_string_pretty(&state).map_err(|e| e.to_string())?;
        fs::write(self.state_path(), raw).map_err(|e| e.to_string())?;
        say("done", "Entorno listo");
        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn indexes() -> TorchIndexes {
        serde_json::from_str(include_str!("../../runtime/torch-indexes.json")).unwrap()
    }

    #[test]
    fn cpu_and_unknown_variants_use_the_cpu_index() {
        let ix = indexes();
        assert_eq!(ix.url_for("cpu", None, true).0, "cpu");
        assert!(ix.url_for("cpu", None, true).1.ends_with("/cpu"));
        assert_eq!(ix.url_for("rara", None, false).0, "cpu");
    }

    #[test]
    fn cuda_index_follows_the_driver() {
        let ix = indexes();
        let (v, new) = ix.url_for("cuda", Some("575.51.02"), false);
        assert_eq!(v, "cuda");
        assert!(new.ends_with("/cu128"), "{new}");
        assert!(ix.url_for("cuda", Some("561.09"), false).1.ends_with("/cu126"));
        // Driver viejo o desconocido: CPU (mejor que una wheel que no carga).
        assert_eq!(ix.url_for("cuda", Some("470.1"), false).0, "cpu");
        assert_eq!(ix.url_for("cuda", None, false).0, "cpu");
    }

    #[test]
    fn rocm_only_on_linux() {
        let ix = indexes();
        assert_eq!(ix.url_for("rocm", None, true).0, "rocm");
        assert_eq!(ix.url_for("rocm", None, false).0, "cpu");
    }
}
