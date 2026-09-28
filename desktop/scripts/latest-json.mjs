// Arma `latest.json` para el updater de Tauri (ADR-0037) a partir de los paquetes firmados.
//   windows-x86_64 → *-setup.exe (NSIS) + .sig
//   linux-x86_64   → *.AppImage + .sig
// Uso: node desktop/scripts/latest-json.mjs --version 0.2.0 --base-url <url de descarga>
//        [--notes-file notas.md] --out latest.json <carpeta con los instaladores>...
import { existsSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { basename, join } from "node:path";
import { parseArgs } from "node:util";

const PLATFORMS = [
  { key: "windows-x86_64", match: (f) => f.endsWith("-setup.exe") },
  { key: "linux-x86_64", match: (f) => f.endsWith(".AppImage") },
];

function walk(dir) {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    return statSync(path).isDirectory() ? walk(path) : [path];
  });
}

function buildManifest({ version, baseUrl, notes, files, now = new Date() }) {
  const base = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  const platforms = {};
  for (const { key, match } of PLATFORMS) {
    const found = files.filter((f) => match(basename(f)) && existsSync(`${f}.sig`));
    if (found.length > 1) throw new Error(`más de un paquete para ${key}: ${found.join(", ")}`);
    if (!found.length) continue;
    platforms[key] = {
      signature: readFileSync(`${found[0]}.sig`, "utf8").trim(),
      url: base + encodeURIComponent(basename(found[0])),
    };
  }
  if (!Object.keys(platforms).length) throw new Error("no hay paquetes firmados (.sig)");
  return {
    version,
    notes: notes ?? "",
    pub_date: now.toISOString().replace(/\.\d{3}Z$/, "Z"),
    platforms,
  };
}

function main() {
  const { values, positionals } = parseArgs({
    allowPositionals: true,
    options: {
      version: { type: "string" },
      "base-url": { type: "string" },
      "notes-file": { type: "string" },
      out: { type: "string" },
    },
  });
  if (!values.version || !values["base-url"] || !values.out || !positionals.length) {
    throw new Error("uso: --version V --base-url URL --out latest.json <carpeta>...");
  }
  const manifest = buildManifest({
    version: values.version.replace(/^v/, ""),
    baseUrl: values["base-url"],
    notes: values["notes-file"] ? readFileSync(values["notes-file"], "utf8") : "",
    files: positionals.flatMap(walk),
  });
  writeFileSync(values.out, JSON.stringify(manifest, null, 2) + "\n");
  console.log(`latest.json ${manifest.version}: ${Object.keys(manifest.platforms).join(", ")}`);
}

main();
