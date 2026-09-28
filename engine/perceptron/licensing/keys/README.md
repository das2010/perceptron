# Claves públicas de licencia (ADR-0035)

Acá va la clave **pública** Ed25519 de Preteco con la que se firman las licencias, como
`<key_id>.pub` (PEM). Se genera una vez en una máquina segura:

    perceptron license keygen --out <carpeta-segura> --key-id preteco-2026

La **privada** (`preteco-2026.key`) nunca entra al repositorio ni al producto: queda en el
gestor de secretos de Preteco y se usa solo para `perceptron license issue`.
Para rotar la clave se agrega un `.pub` nuevo (las licencias viejas siguen validando con la
anterior hasta que se retire).
