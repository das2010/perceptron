# Despliegue del Team Server

La imagen (`server/deploy/Dockerfile`) trae la API, la UI web y el worker de la cola. Con `TORCH_VARIANT` se elige la variante de PyTorch: `cpu` (por defecto), `cu128`, `cu126`, `rocm6.4` o `xpu`.

## Docker Compose (un servidor)

```bash
cp server/deploy/.env.example server/deploy/.env   # completar claves y el primer Admin
docker compose -f server/deploy/compose.yaml up -d --build
docker compose -f server/deploy/compose.yaml --profile gpu up -d   # + worker GPU (NVIDIA)
```

Servicios:
- PostgreSQL 16: metadata, más la base `mlflow`.
- Valkey: la cola.
- MLflow server.
- `server`: API + UI en `:8080`.
- `worker`: CPU. `worker-gpu` va en el perfil `gpu`.

Servidor y workers comparten el volumen `workspace`. La carpeta `PERCEPTRON_SOURCES_DIR` se monta de solo lectura como «fuente del servidor».

Pruebas de humo, las mismas que corre el CI:

```bash
PERCEPTRON_ADMIN_EMAIL=… PERCEPTRON_ADMIN_PASSWORD=… sh server/deploy/smoke.sh
PERCEPTRON_ADMIN_EMAIL=… PERCEPTRON_ADMIN_PASSWORD=… python3 server/deploy/smoke_study.py
```

## Kubernetes (Helm)

```bash
helm install perceptron server/deploy/helm/perceptron \
  --set existingSecret=perceptron-secrets \
  --set ingress.enabled=true --set ingress.host=perceptron.empresa.com \
  --set server.publicUrl=https://perceptron.empresa.com \
  --set workers.gpu.enabled=true
```

- PostgreSQL es externo. El Secret lleva `database-url`, `mlflow-database-url`, `secret-key`, `master-key` y, opcionalmente, `admin-email` y `admin-password`.
- El workspace necesita un PVC `ReadWriteMany` (NFS, CephFS, EFS…), compartido por el servidor y los workers.
- El servidor corre con una réplica: los jobs se siguen en memoria (ADR-0031).

## SSO y backups

- [SSO con Entra ID / Google](../../docs/ops/sso.md)
- [Backups y restauración](../../docs/ops/backups.md)
