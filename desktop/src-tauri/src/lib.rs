//! Shell de escritorio de Perceptron (Tauri 2, ADR-0026).
//!
//! Al arrancar prepara el runtime Python embebido (primer arranque o nueva versión), lanza el
//! Engine como sidecar y le entrega a la UI la conexión (`engine_connection`). Con
//! `--provision-only [--report <archivo>]` no abre ventana: aprovisiona, levanta el Engine,
//! verifica `/system/health`, escribe el reporte y sale (smoke test del instalador en CI).
//! Con `--update-only` busca e instala la actualización firmada y sale (`updater.rs`).

mod engine;
mod runtime;
mod secrets;
mod updater;

use engine::{Connection, EngineProcess, Launcher};
use runtime::{Progress, Runtime, RuntimeState};
use serde::Serialize;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use tauri::{AppHandle, Emitter, Manager, RunEvent, WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_updater::Update;
use updater::UpdateInfo;

const PROGRESS_EVENT: &str = "runtime://progress";

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum Status {
    Starting,
    Ready { connection: Connection },
    Failed { error: String },
}

#[derive(Default)]
struct Shared {
    status: Option<Status>,
    engine: Option<EngineProcess>,
    runtime: Option<RuntimeState>,
}

#[derive(Default, Clone)]
struct AppState(Arc<(Mutex<Shared>, Condvar)>);

impl AppState {
    fn set(&self, status: Status) {
        let (lock, cv) = &*self.0;
        lock.lock().expect("estado").status = Some(status);
        cv.notify_all();
    }

    fn wait_ready(&self) -> Result<Connection, String> {
        let (lock, cv) = &*self.0;
        let mut guard = lock.lock().map_err(|e| e.to_string())?;
        loop {
            match &guard.status {
                Some(Status::Ready { connection }) => return Ok(connection.clone()),
                Some(Status::Failed { error }) => return Err(error.clone()),
                _ => guard = cv.wait(guard).map_err(|e| e.to_string())?,
            }
        }
    }

    fn stop_engine(&self) {
        let (lock, _) = &*self.0;
        if let Ok(mut g) = lock.lock() {
            if let Some(mut e) = g.engine.take() {
                e.stop();
            }
        }
    }
}

fn arg_value(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

/// Desarrollo (`tauri dev`): el Engine del repo con `uv run`, sin runtime embebido.
fn dev_repo() -> Option<PathBuf> {
    if !cfg!(debug_assertions) || std::env::var_os("PERCEPTRON_EMBEDDED_RUNTIME").is_some() {
        return None;
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
    root.join("uv.lock").is_file().then_some(root)
}

fn prepare(app: &AppHandle, variant: Option<&str>) -> Result<(Launcher, PathBuf), String> {
    let paths = app.path();
    let data = paths.app_local_data_dir().map_err(|e| e.to_string())?;
    if let Some(root) = dev_repo() {
        return Ok((Launcher::Repo { root }, data.join("logs")));
    }
    let resources = paths.resource_dir().map_err(|e| e.to_string())?;
    let rt = Runtime::new(&data, &resources);
    let emit = |p: Progress| {
        let _ = app.emit(PROGRESS_EVENT, p);
    };
    let version = app.package_info().version.to_string();
    let state = rt.provision(&version, variant, &emit)?;
    app.state::<AppState>().0 .0.lock().map_err(|e| e.to_string())?.runtime = Some(state);
    Ok((Launcher::Embedded { python: rt.python() }, rt.logs_dir()))
}

fn start(app: &AppHandle, variant: Option<&str>) -> Result<Connection, String> {
    let state = app.state::<AppState>().inner().clone();
    state.stop_engine();
    state.set(Status::Starting);
    let result = prepare(app, variant).and_then(|(launcher, logs)| engine::launch(&launcher, &logs));
    match result {
        Ok(process) => {
            let connection = process.connection.clone();
            state.0 .0.lock().map_err(|e| e.to_string())?.engine = Some(process);
            state.set(Status::Ready { connection: connection.clone() });
            Ok(connection)
        }
        Err(error) => {
            state.set(Status::Failed { error: error.clone() });
            Err(error)
        }
    }
}

#[derive(Serialize)]
struct Report {
    ok: bool,
    runtime: Option<RuntimeState>,
    health: Option<serde_json::Value>,
    error: Option<String>,
}

fn write_report<T: Serialize>(report: &T) {
    let json = serde_json::to_string_pretty(report).unwrap_or_default();
    println!("{json}");
    if let Some(path) = arg_value("--report") {
        let _ = std::fs::write(path, &json);
    }
}

fn provision_only(app: &AppHandle) -> i32 {
    let outcome = start(app, None).and_then(|conn| engine::health(&conn));
    let state = app.state::<AppState>();
    let runtime = state.0 .0.lock().ok().and_then(|g| g.runtime.clone());
    let report = match outcome {
        Ok(body) => Report {
            ok: true,
            runtime,
            health: serde_json::from_str(&body).ok(),
            error: None,
        },
        Err(e) => Report { ok: false, runtime, health: None, error: Some(e) },
    };
    write_report(&report);
    state.stop_engine();
    if report.ok {
        0
    } else {
        1
    }
}

#[derive(Serialize)]
struct UpdateReport {
    ok: bool,
    current_version: String,
    available: Option<UpdateInfo>,
    error: Option<String>,
}

/// `--update-only`: busca, descarga (verificando la firma) e instala. En Windows `install`
/// lanza el instalador y cierra la app, por eso el reporte se escribe antes.
fn update_only(app: &AppHandle) -> i32 {
    let mut report = UpdateReport {
        ok: false,
        current_version: app.package_info().version.to_string(),
        available: None,
        error: None,
    };
    let found: Result<Option<(Update, Vec<u8>)>, String> =
        tauri::async_runtime::block_on(async {
            match updater::check(app, || {}).await? {
                None => Ok(None),
                Some(update) => {
                    let bytes = updater::download(app, &update).await?;
                    Ok(Some((update, bytes)))
                }
            }
        });
    match found {
        Err(e) => report.error = Some(e),
        Ok(None) => report.ok = true,
        Ok(Some((update, bytes))) => {
            // sin relanzar la app: el CI verifica la versión nueva con `--provision-only`
            let update = update.restart_after_install(false);
            report.available = Some(UpdateInfo::from(&update));
            report.ok = true;
            write_report(&report);
            if let Err(e) = update.install(&bytes) {
                report.ok = false;
                report.error = Some(e.to_string());
            }
        }
    }
    write_report(&report);
    if report.ok {
        0
    } else {
        1
    }
}

// ------------------------------------------------------------------ comandos para la UI

#[tauri::command]
async fn engine_connection(state: tauri::State<'_, AppState>) -> Result<Connection, String> {
    let state = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || state.wait_ready())
        .await
        .map_err(|e| e.to_string())?
}

#[tauri::command]
fn runtime_state(state: tauri::State<'_, AppState>) -> Option<RuntimeState> {
    state.0 .0.lock().ok().and_then(|g| g.runtime.clone())
}

/// Cambia la variante de PyTorch sin reinstalar la app (RF-TRN-02) y reinicia el Engine.
#[tauri::command]
async fn set_torch_variant(app: AppHandle, variant: String) -> Result<Connection, String> {
    tauri::async_runtime::spawn_blocking(move || start(&app, Some(&variant)))
        .await
        .map_err(|e| e.to_string())?
}

/// Reintenta el aprovisionamiento y el arranque del Engine (p. ej. tras un fallo de red).
#[tauri::command]
async fn restart_engine(app: AppHandle) -> Result<Connection, String> {
    tauri::async_runtime::spawn_blocking(move || start(&app, None))
        .await
        .map_err(|e| e.to_string())?
}

/// Busca una versión nueva en el endpoint configurado; la deja pendiente para `install_update`.
#[tauri::command]
async fn check_update(app: AppHandle) -> Result<Option<UpdateInfo>, String> {
    let state = app.state::<AppState>().inner().clone();
    let update = updater::check(&app, move || state.stop_engine()).await?;
    Ok(updater::store(&app, update))
}

/// Descarga e instala la actualización pendiente y reinicia la app (en Windows la cierra el
/// instalador). Si la instalación falla, vuelve a levantar el Engine.
#[tauri::command]
async fn install_update(app: AppHandle) -> Result<(), String> {
    let update = updater::take(&app).ok_or("no hay una actualización pendiente")?;
    let bytes = updater::download(&app, &update).await?;
    app.state::<AppState>().stop_engine();
    if let Err(e) = update.install(&bytes) {
        let handle = app.clone();
        let _ = tauri::async_runtime::spawn_blocking(move || start(&handle, None)).await;
        return Err(e.to_string());
    }
    app.restart()
}

pub fn run() {
    let provision = std::env::args().any(|a| a == "--provision-only");
    let update = std::env::args().any(|a| a == "--update-only");
    let headless = provision || update;
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .manage(AppState::default())
        .manage(updater::Pending::default())
        .invoke_handler(tauri::generate_handler![
            engine_connection,
            runtime_state,
            set_torch_variant,
            restart_engine,
            check_update,
            install_update,
            secrets::get_secret,
            secrets::set_secret,
            secrets::delete_secret,
        ])
        .setup(move |app| {
            let handle = app.handle().clone();
            if headless {
                std::thread::spawn(move || {
                    let code = if update { update_only(&handle) } else { provision_only(&handle) };
                    handle.exit(code);
                });
                return Ok(());
            }
            WebviewWindowBuilder::new(app, "main", WebviewUrl::default())
                .title("Perceptron")
                .inner_size(1400.0, 900.0)
                .min_inner_size(1024.0, 700.0)
                .build()?;
            std::thread::spawn(move || {
                let _ = start(&handle, None);
            });
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("no se pudo iniciar Perceptron");
    app.run(|handle, event| {
        if let RunEvent::Exit = event {
            handle.state::<AppState>().stop_engine();
        }
    });
}
