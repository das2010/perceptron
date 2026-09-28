//! Actualización automática firmada (Tauri updater, ADR-0037).
//!
//! La app consulta `latest.json` (GitHub Releases), verifica la firma minisign del paquete con
//! la clave pública de `tauri.conf.json` e instala encima. Los proyectos viven en el workspace
//! del usuario y el runtime Python se reaprovisiona solo cuando cambia la versión de la app, así
//! que una actualización N → N+1 no toca los datos. Antes de instalar se detiene el Engine.
//!
//! Con `--update-only [--report <archivo>]` no abre ventana: busca, descarga e instala la
//! actualización y sale (prueba N → N+1 en CI).

use serde::Serialize;
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager, Runtime};
use tauri_plugin_updater::{Update, Updater, UpdaterExt};

const PROGRESS_EVENT: &str = "updater://progress";

/// Actualización encontrada por `check_update`, a la espera de que el usuario la instale.
#[derive(Default)]
pub struct Pending(Mutex<Option<Update>>);

#[derive(Clone, Debug, Serialize)]
pub struct UpdateInfo {
    pub version: String,
    pub current_version: String,
    pub notes: Option<String>,
    pub date: Option<String>,
}

impl From<&Update> for UpdateInfo {
    fn from(u: &Update) -> Self {
        Self {
            version: u.version.clone(),
            current_version: u.current_version.clone(),
            notes: u.body.clone().filter(|b| !b.trim().is_empty()),
            date: u.raw_json.get("pub_date").and_then(|d| d.as_str()).map(str::to_string),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct DownloadProgress {
    downloaded: u64,
    total: Option<u64>,
}

/// `before_exit` corre justo antes de lanzar el instalador (en Windows el updater cierra la app).
fn updater<R: Runtime>(
    app: &AppHandle<R>,
    before_exit: impl Fn() + Send + Sync + 'static,
) -> Result<Updater, String> {
    app.updater_builder().on_before_exit(before_exit).build().map_err(|e| e.to_string())
}

pub async fn check<R: Runtime>(
    app: &AppHandle<R>,
    before_exit: impl Fn() + Send + Sync + 'static,
) -> Result<Option<Update>, String> {
    updater(app, before_exit)?.check().await.map_err(|e| e.to_string())
}

/// Descarga (verificando la firma) con progreso en `updater://progress`.
pub async fn download<R: Runtime>(app: &AppHandle<R>, update: &Update) -> Result<Vec<u8>, String> {
    let mut downloaded = 0u64;
    update
        .download(
            |chunk, total| {
                downloaded += chunk as u64;
                let _ = app.emit(PROGRESS_EVENT, DownloadProgress { downloaded, total });
            },
            || {},
        )
        .await
        .map_err(|e| e.to_string())
}

pub fn store(app: &AppHandle, update: Option<Update>) -> Option<UpdateInfo> {
    let info = update.as_ref().map(UpdateInfo::from);
    if let Ok(mut pending) = app.state::<Pending>().0.lock() {
        *pending = update;
    }
    info
}

pub fn take(app: &AppHandle) -> Option<Update> {
    app.state::<Pending>().0.lock().ok().and_then(|mut p| p.take())
}
