# Emisión de licencias (uso interno de Preteco)

Las licencias son archivos `license.json` firmados con Ed25519 (ADR-0035). Se emiten **localmente**, en una máquina de Preteco con la clave privada, y el cliente las instala en Configuración → Licencia, o el Admin del servidor en el Team Server.

## Requisitos
- La clave privada `preteco-2026.key`, desde la bóveda de Preteco. Nunca va al repositorio ni se envía por mail o chat.
- Una máquina donde Python pueda ejecutarse, con [uv](https://docs.astral.sh/uv/) y una copia del repositorio. No hace falta instalar Perceptron: el script solo necesita `cryptography`.

## Emitir

```bash
uv run --no-project --with cryptography scripts/issue_license.py \
  --key /ruta/segura/preteco-2026.key --key-id preteco-2026 \
  --licensee "Acme SA" --seats 10 --servers 1 --gpus 2 --days 365 \
  --out license-acme.json
```

| Opción | Qué define |
|---|---|
| `--licensee` | Nombre del cliente (se muestra en la app). |
| `--seats` | Usuarios activos del Team Server (sin la opción: sin tope). |
| `--servers` | Instalaciones del Team Server. |
| `--gpus` | GPUs en workers del servidor. |
| `--features` | Funciones habilitadas, separadas por coma (`features.yaml`); `*` = todas. |
| `--days` | Vigencia; sin la opción, la licencia es perpetua. |
| `--edition` | Nombre de la edición (por defecto `team`). |

Cada licencia tiene un `id` único; conviene registrar en un listado interno a quién se emitió y con qué topes. Si Perceptron está instalado en esa máquina, `perceptron license issue` hace lo mismo con las mismas opciones.

## Verificar antes de enviar

```bash
perceptron license verify license-acme.json
```

Da `valid` con la clave pública empaquetada `preteco-2026`. Sin Perceptron instalado, se verifica al instalarla: la app rechaza licencias con firma inválida o fuera de vigencia.

## Qué pasa en el cliente
- En v1 la licencia **se informa pero no limita**: el estado, el titular, los topes y el vencimiento se ven en Configuración. En el Team Server, la consola muestra el uso real contra los topes (usuarios, GPUs, servidores) y marca los excedidos.
- Para renovar o ampliar, se emite una licencia nueva y se instala encima.

## Rotación de la clave
Si la clave privada se pierde o se compromete: generar un par nuevo (ver ADR-0035), agregar el `.pub` nuevo al producto en `engine/perceptron/licensing/keys/`, publicar un release y emitir las licencias nuevas con el `--key-id` nuevo. Si la clave se comprometió, hay que quitar el `.pub` viejo en ese release.
