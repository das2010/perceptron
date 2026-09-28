#!/usr/bin/env sh
# Smoke del Team Server desplegado: health, UI, login, CSRF, RBAC y fuentes del servidor.
# Uso: server/deploy/smoke.sh [URL] (admin según PERCEPTRON_ADMIN_EMAIL/PASSWORD).
set -eu
base=${1:-http://localhost:8080}
email=${PERCEPTRON_ADMIN_EMAIL:?}
password=${PERCEPTRON_ADMIN_PASSWORD:?}
jar=$(mktemp)
code() { curl -s -o /dev/null -w '%{http_code}' "$@"; }

curl -fs "$base/api/v1/system/health" >/dev/null
curl -fs "$base/" | grep -q 'id="root"'
curl -fsI "$base/" | grep -qi 'content-security-policy'
test "$(code "$base/api/v1/projects")" = 401

curl -fs -c "$jar" -H 'Content-Type: application/json' \
  -d "{\"email\":\"$email\",\"password\":\"$password\"}" "$base/api/v1/auth/login" >/dev/null
csrf=$(awk '$6 == "pt_csrf" {print $7}' "$jar")
test -n "$csrf"
# Sin cabecera CSRF una escritura con cookies se rechaza.
test "$(code -b "$jar" -H 'Content-Type: application/json' -d '{"name":"x"}' "$base/api/v1/projects")" = 403

pid=$(curl -fs -b "$jar" -H "X-CSRF-Token: $csrf" -H 'Content-Type: application/json' \
  -d '{"name":"Smoke"}' "$base/api/v1/projects" | sed -E 's/.*"id":"([^"]+)".*/\1/')
curl -fs -b "$jar" "$base/api/v1/server/sources" | grep -q '/sources'
curl -fs -b "$jar" -H "X-CSRF-Token: $csrf" -H 'Content-Type: application/json' \
  -d '{"path":"/sources/demanda.csv"}' "$base/api/v1/projects/$pid/sources" >/dev/null
# Fuera de las fuentes habilitadas: prohibido.
test "$(code -b "$jar" -H "X-CSRF-Token: $csrf" -H 'Content-Type: application/json' \
  -d '{"path":"/etc/passwd"}' "$base/api/v1/projects/$pid/sources")" = 403
rm -f "$jar"
echo "smoke OK ($base)"
