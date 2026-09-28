// Verifica que cada índice de `desktop/runtime/torch-indexes.json` publique la versión de torch y
// torchvision fijada (`src-tauri/resources/torch.json`, la genera `prepare:runtime`) para
// CPython 3.12 en Windows y Linux x86_64. Sin esto, subir torch puede dejar una variante sin
// wheels y el primer arranque falla recién en la PC del usuario ("No solution found").
// Uso: node desktop/scripts/check-torch-indexes.mjs
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const indexes = JSON.parse(readFileSync(resolve(here, "..", "runtime", "torch-indexes.json"), "utf8"));
const versions = JSON.parse(
  readFileSync(resolve(here, "..", "src-tauri", "resources", "torch.json"), "utf8"),
);

const PLATFORMS = { windows: "win_amd64", linux: "manylinux_2_28_x86_64" };
const variants = [
  { path: indexes.cpu, platforms: ["windows", "linux"] },
  ...indexes.cuda.map((c) => ({ path: c.path, platforms: ["windows", "linux"] })),
  { path: indexes.rocm, platforms: ["linux"] }, // ROCm solo en Linux
  { path: indexes.xpu, platforms: ["windows", "linux"] },
];

async function listing(path, pkg) {
  const url = `${indexes.base_url.replace(/\/$/, "")}/${path}/${pkg}/`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return decodeURIComponent(await res.text());
}

const missing = [];
for (const v of variants) {
  for (const [pkg, version] of [
    ["torch", versions.torch],
    ["torchvision", versions.torchvision],
  ]) {
    const page = await listing(v.path, pkg);
    for (const platform of v.platforms) {
      const wheel = `${pkg}-${version}+${v.path}-cp312-cp312-${PLATFORMS[platform]}.whl`;
      if (!page.includes(wheel)) missing.push(`${v.path}: ${wheel}`);
    }
  }
}
if (missing.length) {
  console.error("Faltan wheels de PyTorch en índices de torch-indexes.json:\n  " + missing.join("\n  "));
  process.exit(1);
}
console.log(`torch ${versions.torch} / torchvision ${versions.torchvision}: ${variants.map((v) => v.path).join(", ")} OK`);
