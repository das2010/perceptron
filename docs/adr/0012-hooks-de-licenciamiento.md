# ADR-0012: Hooks de licenciamiento sin enforcement en v1
- Estado: aceptado
- Fecha: 2026-09-26
- Contexto: SPEC §7.17: modelo comercial a definir (D1).
- Decisión: `LicenseProvider` (Protocol) + `DevLicenseProvider` (todo habilitado). Claves de feature únicamente en `engine/perceptron/licensing/features.yaml`; el código consulta `features.is_enabled(key)` / `features.require(key)`; una clave no declarada es error.
- Consecuencias: Agregar enforcement luego (Ed25519 offline, suscripción, asiento/servidor/GPU) no requiere tocar los llamadores.
- Alternativas consideradas: —
