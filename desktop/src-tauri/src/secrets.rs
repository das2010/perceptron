//! Secretos en el keychain del SO (SPEC §4.2): Credential Manager en Windows, Secret Service
//! en Linux. La UI nunca los guarda en el navegador.

const SERVICE: &str = "com.preteco.perceptron";
/// La UI solo toca sus propias entradas (`ui.<nombre>`): nada de lo que el webview ejecute
/// puede leer o pisar otras credenciales del servicio.
const PREFIX: &str = "ui.";

fn valid_key(key: &str) -> bool {
    key.len() <= 128
        && key.strip_prefix(PREFIX).is_some_and(|rest| {
            !rest.is_empty()
                && rest.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '/'))
        })
}

fn entry(key: &str) -> Result<keyring::Entry, String> {
    if !valid_key(key) {
        return Err(format!("clave no permitida: las de la UI empiezan con {PREFIX}"));
    }
    keyring::Entry::new(SERVICE, key).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn get_secret(key: String) -> Result<Option<String>, String> {
    match entry(&key)?.get_password() {
        Ok(v) => Ok(Some(v)),
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(e) => Err(e.to_string()),
    }
}

#[tauri::command]
pub fn set_secret(key: String, value: String) -> Result<(), String> {
    entry(&key)?.set_password(&value).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn delete_secret(key: String) -> Result<(), String> {
    match entry(&key)?.delete_credential() {
        Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
        Err(e) => Err(e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::valid_key;

    #[test]
    fn only_ui_prefixed_keys() {
        assert!(valid_key("ui.servidor/principal"));
        assert!(valid_key("ui.token_1"));
        let long = "ui.x".repeat(40);
        for bad in ["", "ui.", "llm/anthropic", "perceptron", "ui.a b", "ui.ñ", long.as_str()] {
            assert!(!valid_key(bad), "{bad}");
        }
    }
}
