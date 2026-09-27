#!/usr/bin/env bash
# Espera el último run de CI de una rama y resume el resultado y los errores.
# Uso: scripts/ci-watch.sh [rama]
set -u
GH="${GH:-$LOCALAPPDATA/Microsoft/WinGet/Packages/GitHub.cli_Microsoft.Winget.Source_8wekyb3d8bbwe/bin/gh.exe}"
command -v gh >/dev/null 2>&1 && GH=gh
BRANCH="${1:-$(git branch --show-current)}"
SHA="$(git rev-parse HEAD)"

# Esperar a que exista el run del commit actual
for _ in $(seq 1 30); do
  ID="$("$GH" run list -b "$BRANCH" -c "$SHA" -L1 --json databaseId -q '.[0].databaseId' 2>/dev/null)"
  [ -n "$ID" ] && break
  sleep 5
done
[ -z "${ID:-}" ] && { echo "no se encontró run para $SHA"; exit 2; }

"$GH" run watch "$ID" --interval 20 >/dev/null 2>&1
"$GH" run view "$ID" --json conclusion,jobs -q '"RUN \(.conclusion) (id \(.databaseId // ""))", (.jobs[] | "\(.conclusion)\t\(.name)")'
echo "run_id=$ID"
"$GH" run view "$ID" --log-failed 2>/dev/null \
  | sed -E 's/\t[0-9TZ:.-]+ /\t/' \
  | grep -aE "FAILED|Error|error:|error\[|assert|Traceback|\.py:[0-9]+|##\[error\]" \
  | cut -c1-260 | head -80
