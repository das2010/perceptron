#!/usr/bin/env sh
# Restauración de un backup de backup.sh (RF-SRV-08). REEMPLAZA los datos actuales.
#
#   server/deploy/restore.sh <carpeta perceptron-AAAAMMDDTHHMMSSZ>
#
# Verifica los checksums, detiene servidor y workers, restaura la metadata y el tracking
# (pg_restore --clean) y reemplaza el contenido de los volúmenes del workspace y de MLflow.
set -eu
src=${1:?uso: restore.sh <carpeta del backup>}
src=$(cd "$src" && pwd)
here=$(cd "$(dirname "$0")" && pwd)
compose="docker compose -f $here/compose.yaml"
db_user=${POSTGRES_USER:-perceptron}
db_name=${POSTGRES_DB:-perceptron}

(cd "$src" && sha256sum -c SHA256SUMS)

echo "deteniendo servidor, workers y MLflow…"
$compose stop server worker mlflow >/dev/null

echo "metadata y tracking…"
$compose exec -T postgres pg_restore -U "$db_user" -d "$db_name" --clean --if-exists --no-owner \
  < "$src/perceptron.dump"
$compose exec -T postgres pg_restore -U "$db_user" -d mlflow --clean --if-exists --no-owner \
  < "$src/mlflow.dump"

echo "workspace y artefactos…"
$compose run --rm --no-deps --user 0 --entrypoint sh -v "$src:/backup:ro" server \
  -c 'find /data -mindepth 1 -delete && tar xzf /backup/workspace.tar.gz -C /data'
$compose run --rm --no-deps --user 0 --entrypoint sh -v "$src:/backup:ro" -v perceptron_mlartifacts:/mlartifacts server \
  -c 'find /mlartifacts -mindepth 1 -delete && tar xzf /backup/mlartifacts.tar.gz -C /mlartifacts'

$compose start mlflow server worker >/dev/null
echo "restaurado desde $src"
