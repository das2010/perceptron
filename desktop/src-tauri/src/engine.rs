//! Sidecar del Engine (SPEC §4.2, ADR-0026): `perceptron serve --new-token` en `127.0.0.1`,
//! puerto aleatorio y token efímero, leídos de la línea `ready` que el Engine emite por stdout.

use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use crate::runtime::hide_console;

const READY_TIMEOUT: Duration = Duration::from_secs(180);
const LISTEN_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Connection {
    /// URL base del Engine, sin `/api/v1` (contrato de `PlatformBridge.engine()`).
    #[serde(rename = "baseUrl")]
    pub base_url: String,
    pub token: String,
}

#[derive(Debug, Deserialize)]
struct Ready {
    event: String,
    host: String,
    port: u16,
    token: Option<String>,
}

pub struct EngineProcess {
    child: Child,
    pub connection: Connection,
}

impl EngineProcess {
    pub fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for EngineProcess {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Cómo se lanza el Engine: runtime embebido (release) o el repo con `uv run` (desarrollo).
pub enum Launcher {
    Embedded { python: PathBuf },
    Repo { root: PathBuf },
}

impl Launcher {
    fn command(&self) -> Command {
        let serve = ["serve", "--new-token"];
        match self {
            Launcher::Embedded { python } => {
                let mut cmd = Command::new(python);
                cmd.args(["-m", "perceptron.cli.main"]).args(serve);
                cmd
            }
            Launcher::Repo { root } => {
                let mut cmd = Command::new("uv");
                cmd.args(["run", "perceptron"]).args(serve).current_dir(root);
                cmd
            }
        }
    }
}

fn parse_ready(line: &str) -> Option<Connection> {
    let ready: Ready = serde_json::from_str(line.trim()).ok()?;
    if ready.event != "ready" {
        return None;
    }
    Some(Connection {
        base_url: format!("http://{}:{}", ready.host, ready.port),
        token: ready.token.unwrap_or_default(),
    })
}

fn drain(reader: impl Read + Send + 'static, mut log: File) {
    thread::spawn(move || {
        let mut buf = [0u8; 8192];
        let mut r = reader;
        while let Ok(n) = r.read(&mut buf) {
            if n == 0 || log.write_all(&buf[..n]).is_err() {
                break;
            }
        }
    });
}

pub fn launch(launcher: &Launcher, logs_dir: &Path) -> Result<EngineProcess, String> {
    fs::create_dir_all(logs_dir).map_err(|e| e.to_string())?;
    let open_log = |name: &str| {
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(logs_dir.join(name))
            .map_err(|e| e.to_string())
    };
    let mut cmd = launcher.command();
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    cmd.env("PYTHONUNBUFFERED", "1").env("PYTHONIOENCODING", "utf-8");
    hide_console(&mut cmd);
    let mut child = cmd.spawn().map_err(|e| format!("no se pudo lanzar el Engine: {e}"))?;
    let stdout = child.stdout.take().ok_or("sin stdout del Engine")?;
    let stderr = child.stderr.take().ok_or("sin stderr del Engine")?;
    drain(stderr, open_log("engine.log")?);

    let (tx, rx) = mpsc::channel::<Connection>();
    let mut out_log = open_log("engine.out.log")?;
    thread::spawn(move || {
        let mut sent = false;
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            let _ = writeln!(out_log, "{line}");
            if !sent {
                if let Some(conn) = parse_ready(&line) {
                    sent = tx.send(conn).is_ok();
                }
            }
        }
    });
    match rx.recv_timeout(READY_TIMEOUT) {
        Ok(connection) => {
            // El Engine emite `ready` antes de que uvicorn abra el puerto: esperar a que acepte.
            if let Err(e) = wait_listening(&connection, LISTEN_TIMEOUT) {
                let _ = child.kill();
                return Err(e);
            }
            Ok(EngineProcess { child, connection })
        }
        Err(_) => {
            let _ = child.kill();
            Err(format!(
                "el Engine no respondió en {} s (ver {})",
                READY_TIMEOUT.as_secs(),
                logs_dir.join("engine.log").display()
            ))
        }
    }
}

fn wait_listening(conn: &Connection, timeout: Duration) -> Result<(), String> {
    let addr = conn.base_url.trim_start_matches("http://");
    let target: SocketAddr = addr.parse().map_err(|e| format!("{addr}: {e}"))?;
    let deadline = Instant::now() + timeout;
    loop {
        if TcpStream::connect_timeout(&target, Duration::from_millis(500)).is_ok() {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(format!("el Engine no abrió {addr} en {} s", timeout.as_secs()));
        }
        thread::sleep(Duration::from_millis(200));
    }
}

/// `GET /api/v1/system/health` sin dependencias HTTP: alcanza para el smoke test.
pub fn health(conn: &Connection) -> Result<String, String> {
    let addr = conn.base_url.trim_start_matches("http://");
    let mut stream = TcpStream::connect(addr).map_err(|e| e.to_string())?;
    stream.set_read_timeout(Some(Duration::from_secs(30))).map_err(|e| e.to_string())?;
    let req = format!(
        "GET /api/v1/system/health HTTP/1.1\r\nHost: {addr}\r\nX-Perceptron-Token: {}\r\nConnection: close\r\n\r\n",
        conn.token
    );
    stream.write_all(req.as_bytes()).map_err(|e| e.to_string())?;
    let mut resp = String::new();
    stream.read_to_string(&mut resp).map_err(|e| e.to_string())?;
    let status = resp.lines().next().unwrap_or_default().to_string();
    if !status.contains(" 200 ") {
        return Err(format!("health: {status}"));
    }
    Ok(resp.split("\r\n\r\n").nth(1).unwrap_or_default().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_ready_handshake() {
        let conn = parse_ready(r#"{"event":"ready","host":"127.0.0.1","port":51234,"token":"abc"}"#)
            .unwrap();
        assert_eq!(conn.base_url, "http://127.0.0.1:51234");
        assert_eq!(conn.token, "abc");
        assert!(parse_ready(r#"{"event":"log","host":"x","port":1}"#).is_none());
        assert!(parse_ready("INFO: arrancando").is_none());
    }
}
