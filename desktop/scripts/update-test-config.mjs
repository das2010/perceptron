// Configuraciones para la prueba de actualización N → N+1 en CI (ADR-0037).
// Con una clave efímera (nunca la de Preteco) y un endpoint local, genera en <dir>:
//   n.conf.json   versión de tauri.conf.json
//   n1.conf.json  la misma con el patch + 1
// y exporta V_N y V_N1 a $GITHUB_ENV.
// Uso: node desktop/scripts/update-test-config.mjs <dir> <clave.pub> <endpoint>
import { appendFileSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const [dir, pubFile, endpoint] = process.argv.slice(2);
if (!dir || !pubFile || !endpoint) throw new Error("uso: <dir> <clave.pub> <endpoint>");

const here = dirname(fileURLToPath(import.meta.url));
const conf = JSON.parse(readFileSync(resolve(here, "..", "src-tauri", "tauri.conf.json"), "utf8"));
const [major, minor, patch] = conf.version.split(".").map(Number);
const versions = { n: conf.version, n1: `${major}.${minor}.${patch + 1}` };

for (const [name, version] of Object.entries(versions)) {
  const override = {
    version,
    bundle: { createUpdaterArtifacts: true },
    plugins: {
      updater: {
        pubkey: readFileSync(pubFile, "utf8").trim(),
        endpoints: [endpoint],
        dangerousInsecureTransportProtocol: true,
        windows: { installMode: "quiet" },
      },
    },
  };
  writeFileSync(join(dir, `${name}.conf.json`), JSON.stringify(override, null, 2));
}
if (process.env.GITHUB_ENV) {
  appendFileSync(process.env.GITHUB_ENV, `V_N=${versions.n}\nV_N1=${versions.n1}\n`);
}
console.log(`N=${versions.n} N+1=${versions.n1}`);
