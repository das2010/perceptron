# Perceptron Desktop (Tauri 2)

Se implementa en la **Capa 3** (SPEC §14): `src-tauri/` (sidecar del Engine, keychain,
updater, deep links) y `runtime/` (runtime Python embebido + selección de la variante de PyTorch).

Contrato con el Engine ya disponible en Capa 0: `perceptron serve --new-token` emite por stdout
una línea JSON `{"event":"ready","host":"127.0.0.1","port":N,"token":"..."}`; la UI envía el
token en la cabecera `X-Perceptron-Token`.
