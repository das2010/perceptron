#!/usr/bin/env sh
# Backup del Team Server con Docker Compose (RF-SRV-08).
#
#   server/deploy/backup.sh [carpeta-destino]
#
# Genera <destino>/perceptron-<fecha>/ con:
#   perceptron.dump   metadata (pg_dump -Fc): proyectos, usuarios, roles, auditoría
#   mlflow.dump       tracking de MLflow
#   workspace.tar.gz  datasets, runs, modelos y exportaciones (volumen /data)
#   mlartifacts.tar.gz artefactos del MLflow server
#   SHA256SUMS
# Consistencia: se hace con el servidor y los workers detenidos (la base queda coherente con
# los archivos). Postgres, Valkey y MLflow siguen arriba. Restaurar: restore.sh.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
compose="docker compose -f $here/compose.yaml"
dest_root=${1:-$here/backups}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
dest="$dest_root/perceptron-$stamp"
mkdir -p "$dest"
dest=$(cd "$dest" && pwd)
db_user=${POSTGRES_USER:-perceptron}
db_name=${POSTGRES_DB:-perceptron}

echo "deteniendo servidor y workers…"
$compose stop server worker >/dev/null
trap '$compose start server worker >/dev/null 2>&1 || true' EXIT

echo "metadata ($db_name) y tracking (mlflow)…"
$compose exec -T postgres pg_dump -U "$db_user" -Fc "$db_name" > "$dest/perceptron.dump"
$compose exec -T postgres pg_dump -U "$db_user" -Fc mlflow > "$dest/mlflow.dump"

echo "workspace y artefactos…"
$compose run --rm --no-deps --user 0 --entrypoint tar -v "$dest:/backup" server \
  czf /backup/workspace.tar.gz -C /data .
$compose run --rm --no-deps --user 0 --entrypoint tar -v "$dest:/backup" -v perceptron_mlartifacts:/mlartifacts server \
  czf /backup/mlartifacts.tar.gz -C /mlartifacts .

(cd "$dest" && sha256sum perceptron.dump mlflow.dump workspace.tar.gz mlartifacts.tar.gz > SHA256SUMS)
echo "backup listo en $dest"
