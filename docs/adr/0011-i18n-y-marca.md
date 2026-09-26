# ADR-0011: i18n es/en y marca Preteco
- Estado: aceptado
- Fecha: 2026-09-26
- Contexto: SPEC §2 y §11: producto Preteco, Titillium Web, paleta del Manual de Marca, temas claro y oscuro.
- Decisión: i18next con español por defecto e inglés; ningún string hardcodeado (test de paridad de claves). Tokens de marca en `ui/src/styles/tokens.css` con tests de valor y de contraste WCAG AA. Fuentes Titillium Web empaquetadas localmente vía `@fontsource/titillium-web` (OFL). Logo provisional tipográfico hasta recibir el vectorial (D8).
- Consecuencias: shadcn/ui + Tailwind se incorporan en Capa 3 mapeados a estos tokens.
- Alternativas consideradas: —
