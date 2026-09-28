// Arma `src-tauri/resources/` para el instalador (ADR-0026):
//   bin/uv[.exe]            binario de uv (el del PATH o $UV_BIN; en CI, el de setup-uv)
//   wheels/*.whl            wheel de perceptron-engine
//   requirements.lock.txt   dependencias fijadas de uv.lock, sin torch/torchvision/torchaudio
//   torch.json              versiones de torch y torchvision del lock
//   torch-indexes.json      índice de PyTorch por variante
// Uso: pnpm -C desktop prepare:runtime
import { execFileSync } from "node:child_process";
import { copyFileSync, mkdirSync, readFileSync, rmSync, writeFileSync, chmodSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "..", "..");
const out = resolve(here, "..", "src-tauri", "resources");
const win = process.platform === "win32";
const TORCH_PACKAGES = new Set(["torch", "torchvision", "torchaudio"]);

function uvPath() {
  if (process.env.UV_BIN) return process.env.UV_BIN;
  const finder = win ? "where" : "which";
  return execFileSync(finder, ["uv"], { encoding: "utf8" }).split(/\r?\n/)[0].trim();
}

function uv(args) {
  return execFileSync(uvPath(), args, { cwd: repo, encoding: "utf8", stdio: ["ignore", "pipe", "inherit"] });
}

/** Mayor versión fijada de un paquete en uv.lock, sin la etiqueta local (+cpu). */
function lockedVersion(lock, name) {
  const re = new RegExp(`\\[\\[package\\]\\]\\nname = "${name}"\\nversion = "([^"]+)"`, "g");
  const versions = [...lock.matchAll(re)].map((m) => m[1].split("+")[0]);
  if (!versions.length) throw new Error(`${name} no está en uv.lock`);
  return versions.sort((a, b) => a.localeCompare(b, undefined, { numeric: true })).at(-1);
}

export function filterRequirements(text) {
  return (
    text
      .split(/\r?\n/)
      .filter((line) => {
        const t = line.trim();
        if (!t || t.startsWith("#") || t.startsWith("--")) return false;
        const name = t.split(/[=<>~!; [(]/)[0].toLowerCase();
        return !TORCH_PACKAGES.has(name);
      })
      .join("\n") + "\n"
  );
}

function main() {
  rmSync(out, { recursive: true, force: true });
  mkdirSync(join(out, "bin"), { recursive: true });
  mkdirSync(join(out, "wheels"), { recursive: true });

  const uvBin = join(out, "bin", win ? "uv.exe" : "uv");
  copyFileSync(uvPath(), uvBin);
  if (!win) chmodSync(uvBin, 0o755);

  uv(["build", "--package", "perceptron-engine", "--wheel", "--out-dir", join(out, "wheels")]);

  const exported = uv([
    "export",
    "--frozen",
    "--no-dev",
    "--no-hashes",
    "--no-header",
    "--no-annotate",
    "--no-emit-workspace",
    "--package",
    "perceptron-engine",
    "--extra",
    "ml",
    "--extra",
    "llm",
    "--extra",
    "export",
  ]);
  writeFileSync(join(out, "requirements.lock.txt"), filterRequirements(exported));

  const lock = readFileSync(join(repo, "uv.lock"), "utf8").replace(/\r\n/g, "\n");
  const versions = {
    torch: lockedVersion(lock, "torch"),
    torchvision: lockedVersion(lock, "torchvision"),
  };
  writeFileSync(join(out, "torch.json"), JSON.stringify(versions, null, 2) + "\n");
  copyFileSync(join(here, "torch-indexes.json"), join(out, "torch-indexes.json"));
  console.log(`recursos listos en ${out}: torch ${versions.torch}, torchvision ${versions.torchvision}`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
