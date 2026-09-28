"""Auditoría de licencias para la distribución comercial (Capa 7, ADR-0036).

Inventaría lo que se entrega y lo clasifica según `docs/licenses/policy.yaml`:
- Python: dependencias de runtime (`uv export --no-dev`), con la licencia de sus metadatos;
- JavaScript: dependencias de producción de la UI y el desktop (`pnpm licenses list --prod`);
- Rust: crates del desktop (`cargo metadata`);
- pesos preentrenados del catálogo (licencia y uso comercial declarado).

Escribe `report.md` y `report.json` y termina con error si hay algo prohibido sin excepción.

    uv run python scripts/license_audit.py --out build/licenses [--skip-js] [--skip-rust]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

# Textos de licencia frecuentes → SPDX.
ALIASES = [
    (r"\bapache\b.*2", "Apache-2.0"),
    (r"^apache$", "Apache-2.0"),
    (r"\bmit\b", "MIT"),
    (r"bsd.?3|new bsd|modified bsd|revised bsd", "BSD-3-Clause"),
    (r"bsd.?2|simplified bsd|freebsd", "BSD-2-Clause"),
    (r"^bsd( license)?$", "BSD-3-Clause"),
    (r"\bisc\b", "ISC"),
    (r"python software foundation|\bpsf\b", "PSF-2.0"),
    (r"mozilla public license 2|mpl.?2", "MPL-2.0"),
    (r"lesser general public license v3|lgpl.?3", "LGPL-3.0-or-later"),
    (r"lesser general public license v2|lgpl.?2", "LGPL-2.1-or-later"),
    (r"affero", "AGPL-3.0"),
    (r"general public license v3|\bgpl.?3", "GPL-3.0"),
    (r"general public license v2|\bgpl.?2", "GPL-2.0"),
    (r"unlicense", "Unlicense"),
    (r"public domain|cc0", "CC0-1.0"),
    (r"zlib", "Zlib"),
    (r"historical permission", "HPND"),
    (r"eclipse public license 2|epl.?2", "EPL-2.0"),
    (r"eclipse distribution", "EDL-1.0"),
    (r"boost", "BSL-1.0"),
    (r"cc.?by.?4", "CC-BY-4.0"),
]


@dataclass
class Item:
    ecosystem: str
    name: str
    version: str
    license: str
    verdict: str  # allowed | review | forbidden | unknown | exception
    note: str = ""


def spdx(text: str) -> str:
    raw = " ".join(text.split())
    if re.fullmatch(r"[A-Za-z0-9.\-+ ()]+", raw) and any(
        tok in raw for tok in (" OR ", " AND ", "-")
    ):
        return raw  # ya parece una expresión SPDX
    low = raw.lower()
    for pattern, name in ALIASES:
        if re.search(pattern, low):
            return name
    return raw or "UNKNOWN"


def classify(expr: str, policy: dict[str, Any]) -> str:
    allowed = set(policy["allowed"])
    review = set(policy["review"])
    forbidden = [p.lower() for p in policy["forbidden_patterns"]]

    def one(term: str) -> str:
        t = term.strip().strip("()").replace("+", "-or-later").strip()
        if (
            t in allowed
            or t.removesuffix("-only") in allowed
            or t.removesuffix("-or-later") in allowed
        ):
            return "allowed"
        if t in review:
            return "review"
        low = t.lower()
        if any(p in low and not low.startswith("l" + p) for p in forbidden):
            return "forbidden"
        return "unknown"

    rank = {"allowed": 0, "review": 1, "unknown": 2, "forbidden": 3}
    alternatives = re.split(r"\s+OR\s+", expr)
    best = "forbidden"
    for alt in alternatives:
        parts = [one(p) for p in re.split(r"\s+AND\s+", alt)]
        worst = max(parts, key=rank.__getitem__)
        if rank[worst] < rank[best]:
            best = worst
    return best


def python_items(policy: dict[str, Any]) -> list[Item]:
    uv = shutil.which("uv") or "uv"
    exported = subprocess.run(  # noqa: S603 - comando fijo
        [
            uv,
            "export",
            "--no-dev",
            "--all-packages",
            "--no-hashes",
            "--format",
            "requirements-txt",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    wanted = {
        re.split(r"[=<>\[; ]", line, maxsplit=1)[0].strip().lower().replace("_", "-")
        for line in exported.splitlines()
        if line and not line.startswith(("#", " ", "-e", "."))
    }
    items = []
    for dist in metadata.distributions():
        name = (dist.metadata["Name"] or "").lower().replace("_", "-")
        if name not in wanted or name.startswith("perceptron"):
            continue
        meta = dist.metadata
        text = meta.get("License-Expression") or ""
        if not text or len(text) > 80:
            classifiers = [
                c.split("::")[-1].strip()
                for c in meta.get_all("Classifier") or []
                if c.startswith("License ::")
            ]
            short = (meta.get("License") or "").strip()
            text = (
                " OR ".join(classifiers)
                if classifiers
                else (short if len(short) < 80 else short[:60])
            )
        lic = " OR ".join(spdx(x) for x in text.split(" OR ")) if text else "UNKNOWN"
        items.append(Item("python", name, dist.version, lic, classify(lic, policy)))
    return sorted(items, key=lambda i: i.name)


def js_items(policy: dict[str, Any]) -> list[Item]:
    pnpm = shutil.which("pnpm")
    if pnpm is None:
        raise RuntimeError("pnpm no está instalado (usar --skip-js)")
    out = subprocess.run(  # noqa: S603
        [pnpm, "licenses", "list", "--json", "--prod", "-r"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    data: dict[str, list[dict[str, Any]]] = json.loads(out)
    items = []
    for lic, pkgs in data.items():
        for p in pkgs:
            expr = spdx(lic)
            for version in p.get("versions") or [p.get("version", "?")]:
                items.append(Item("js", p["name"], str(version), expr, classify(expr, policy)))
    return sorted(items, key=lambda i: (i.name, i.version))


def rust_items(policy: dict[str, Any]) -> list[Item]:
    cargo = shutil.which("cargo")
    if cargo is None:
        raise RuntimeError("cargo no está instalado (usar --skip-rust)")
    out = subprocess.run(  # noqa: S603
        [
            cargo,
            "metadata",
            "--format-version",
            "1",
            "--locked",
            "--manifest-path",
            "desktop/src-tauri/Cargo.toml",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    meta = json.loads(out)
    own = set(meta.get("workspace_members") or [])
    items = []
    for pkg in meta["packages"]:
        if pkg["id"] in own:
            continue
        expr = (pkg.get("license") or "UNKNOWN").replace("/", " OR ")
        items.append(Item("rust", pkg["name"], pkg["version"], expr, classify(expr, policy)))
    return sorted(items, key=lambda i: (i.name, i.version))


def weight_items(policy: dict[str, Any]) -> list[Item]:
    from perceptron.catalog.registry import HF_TEXT_MODELS, TIMM_WEIGHTS

    items = []
    for name, w in TIMM_WEIGHTS.items():
        verdict = classify(spdx(w.license), policy) if w.commercial_ok else "forbidden"
        items.append(Item("weights", f"timm:{name}", w.pretrained_tag, w.license, verdict))
    for name, t in HF_TEXT_MODELS.items():
        lic = str(getattr(t, "license", "UNKNOWN"))
        ok = bool(getattr(t, "commercial_ok", False))
        verdict = classify(spdx(lic), policy) if ok else "forbidden"
        items.append(Item("weights", f"hf:{name}", "", lic, verdict))
    return items


def apply_exceptions(items: list[Item], policy: dict[str, Any]) -> None:
    exceptions: dict[str, str] = policy.get("exceptions") or {}
    for item in items:
        if item.name in exceptions and item.verdict != "allowed":
            item.note = exceptions[item.name]
            item.verdict = "exception"


def render(items: list[Item]) -> str:
    counts: dict[str, int] = {}
    for i in items:
        counts[i.verdict] = counts.get(i.verdict, 0) + 1
    lines = [
        "# Informe de licencias de Perceptron",
        "",
        "Generado por `scripts/license_audit.py` con la política `docs/licenses/policy.yaml`.",
        "",
        "| Veredicto | Cantidad |",
        "|---|---|",
        *(f"| {k} | {v} |" for k, v in sorted(counts.items())),
        "",
    ]
    for eco in ("python", "js", "rust", "weights"):
        group = [i for i in items if i.ecosystem == eco]
        if not group:
            continue
        lines += [
            f"## {eco}",
            "",
            "| Componente | Versión | Licencia | Veredicto | Nota |",
            "|---|---|---|---|---|",
        ]
        lines += [
            f"| {i.name} | {i.version} | {i.license} | {i.verdict} | {i.note} |" for i in group
        ]
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "licenses")
    parser.add_argument("--skip-js", action="store_true")
    parser.add_argument("--skip-rust", action="store_true")
    args = parser.parse_args(argv)
    policy = yaml.safe_load(
        (ROOT / "docs" / "licenses" / "policy.yaml").read_text(encoding="utf-8")
    )
    items = python_items(policy) + weight_items(policy)
    if not args.skip_js:
        items += js_items(policy)
    if not args.skip_rust:
        items += rust_items(policy)
    apply_exceptions(items, policy)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(
        json.dumps([asdict(i) for i in items], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.out / "report.md").write_text(render(items), encoding="utf-8")
    bad = [i for i in items if i.verdict == "forbidden"]
    unknown = [i for i in items if i.verdict == "unknown"]
    review = [i for i in items if i.verdict == "review"]
    print(
        f"{len(items)} componentes · {len(bad)} prohibidos · {len(review)} a revisar · "
        f"{len(unknown)} sin licencia clara"
    )
    for i in bad:
        print(f"PROHIBIDO  {i.ecosystem}:{i.name} {i.version} ({i.license})")
    for i in review + unknown:
        print(f"REVISAR    {i.ecosystem}:{i.name} {i.version} ({i.license})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
