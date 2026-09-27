"""Genera los fixtures sintéticos de los casos de uso UC-01…UC-10 (SPEC §3.3, §15.2).

Solo biblioteca estándar: determinístico (semilla fija), sin dependencias y
multiplataforma. Los datasets son diminutos para que el pipeline completo corra
en CPU en < 2 min. Uso:

    python fixtures/generate.py            # regenera todo en fixtures/
    python fixtures/generate.py --check    # verifica que los archivos coinciden (CI)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import struct
import sys
import wave
import zlib
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED = 20260926

Files = dict[str, bytes]


# ----------------------------------------------------------------- helpers binarios


def zlib_stored(data: bytes) -> bytes:
    """Stream zlib con bloques deflate sin compresión (RFC 1950/1951).

    No usa `zlib.compress`: su salida depende de la implementación (zlib vs zlib-ng),
    lo que rompía el determinismo entre máquinas y versiones de Python.
    """
    out = bytearray(b"\x78\x01")
    step = 0xFFFF
    for i in range(0, max(len(data), 1), step):
        block = data[i : i + step]
        final = 1 if i + step >= len(data) else 0
        out += struct.pack("<BHH", final, len(block), len(block) ^ 0xFFFF) + block
    out += struct.pack(">I", zlib.adler32(data))
    return bytes(out)


def png_bytes(pixels: list[list[tuple[int, int, int]]]) -> bytes:
    """PNG RGB 8-bit, determinístico en cualquier plataforma."""
    height, width = len(pixels), len(pixels[0])
    raw = b"".join(b"\x00" + bytes(c for px in row for c in px) for row in pixels)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib_stored(raw)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def png_gray(mask: list[list[int]]) -> bytes:
    return png_bytes([[(v, v, v) for v in row] for row in mask])


def wav_bytes(samples: list[float], rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(
            b"".join(struct.pack("<h", int(max(-1.0, min(1.0, s)) * 32767)) for s in samples)
        )
    return buf.getvalue()


def csv_bytes(header: list[str], rows: list[list[object]]) -> bytes:
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def jsonl_bytes(records: list[dict[str, object]]) -> bytes:
    return "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records
    ).encode("utf-8")


# ----------------------------------------------------------------- casos de uso


def uc01_churn(rng: random.Random) -> Files:
    """Tabular, clasificación binaria. Incluye una columna id (alerta de leakage) y nulos."""
    plans = ["básico", "estándar", "premium"]
    regions = ["AMBA", "Córdoba", "Santa Fe", "Mendoza"]
    rows = []
    for i in range(400):
        tenure = rng.randint(1, 72)
        plan = rng.choice(plans)
        monthly = round(
            {"básico": 20, "estándar": 45, "premium": 80}[plan] * rng.uniform(0.8, 1.2), 2
        )
        tickets = rng.randint(0, 8)
        logit = (
            -1.2
            - 0.05 * tenure
            + 0.35 * tickets
            + (0.8 if plan == "básico" else 0)
            + rng.gauss(0, 0.6)
        )
        churn = int(1 / (1 + math.exp(-logit)) > 0.5)
        age = "" if rng.random() < 0.05 else rng.randint(18, 80)
        rows.append([f"C{i:05d}", age, rng.choice(regions), plan, tenure, monthly, tickets, churn])
    header = [
        "customer_id",
        "edad",
        "region",
        "plan",
        "antiguedad_meses",
        "cargo_mensual",
        "tickets_90d",
        "churn",
    ]
    return {"uc01_churn/churn.csv": csv_bytes(header, rows)}


_TICKET_TEMPLATES = {
    "facturación": [
        "No puedo descargar la factura de {m}",
        "Me cobraron dos veces en {m}",
        "Error en el importe de la factura",
    ],
    "acceso": [
        "No puedo iniciar sesión desde {d}",
        "Olvidé mi contraseña",
        "La cuenta quedó bloqueada",
    ],
    "rendimiento": [
        "El sistema está muy lento en {d}",
        "La página tarda en cargar",
        "Se congela al exportar reportes",
    ],
    "funcionalidad": [
        "Quisiera agregar usuarios al equipo",
        "¿Cómo configuro alertas por email?",
        "Falta la opción de exportar a Excel",
    ],
}
_MONTHS = ["enero", "febrero", "marzo", "abril", "mayo", "junio"]
_DEVICES = ["el celular", "la notebook", "Chrome", "la app de escritorio"]


def _ticket_text(rng: random.Random, category: str) -> str:
    return rng.choice(_TICKET_TEMPLATES[category]).format(
        m=rng.choice(_MONTHS), d=rng.choice(_DEVICES)
    )


def uc02_tickets_time(rng: random.Random) -> Files:
    """Tabular + texto, regresión (horas de resolución)."""
    rows = []
    base = {"facturación": 6, "acceso": 2, "rendimiento": 20, "funcionalidad": 12}
    for i in range(300):
        cat = rng.choice(list(base))
        priority = rng.choice(["baja", "media", "alta"])
        hours = (
            base[cat]
            * {"baja": 1.5, "media": 1.0, "alta": 0.6}[priority]
            * rng.lognormvariate(0, 0.3)
        )
        rows.append(
            [f"T{i:05d}", _ticket_text(rng, cat), cat, priority, rng.randint(1, 5), round(hours, 2)]
        )
    header = [
        "ticket_id",
        "descripcion",
        "categoria",
        "prioridad",
        "nivel_cliente",
        "horas_resolucion",
    ]
    return {"uc02_tickets_time/tickets.csv": csv_bytes(header, rows)}


def uc03_tickets_es(rng: random.Random) -> Files:
    """Texto en español, clasificación multiclase."""
    records = []
    for i in range(240):
        cat = list(_TICKET_TEMPLATES)[i % len(_TICKET_TEMPLATES)]
        records.append({"id": f"T{i:05d}", "texto": _ticket_text(rng, cat), "categoria": cat})
    rng.shuffle(records)
    return {"uc03_tickets_es/tickets.jsonl": jsonl_bytes(records)}


def _part_image(
    rng: random.Random, defect: bool, size: int = 32
) -> tuple[list[list[tuple[int, int, int]]], tuple[int, int, int, int] | None]:
    g = [[(150 + rng.randint(-8, 8),) * 3 for _ in range(size)] for _ in range(size)]
    pixels = [[(v[0], v[1], v[2]) for v in row] for row in g]
    box = None
    if defect:
        w, h = rng.randint(4, 9), rng.randint(2, 6)
        x, y = rng.randint(1, size - w - 1), rng.randint(1, size - h - 1)
        for yy in range(y, y + h):
            for xx in range(x, x + w):
                pixels[yy][xx] = (40 + rng.randint(0, 20), 30, 30)
        box = (x, y, w, h)
    return pixels, box


def uc04_defects(rng: random.Random) -> Files:
    """Imagen: clasificación (carpetas clase/archivo) + detección (anotaciones COCO)."""
    files: Files = {}
    images, annotations = [], []
    for i in range(40):
        defect = i % 2 == 1
        pixels, box = _part_image(rng, defect)
        name = f"{'defect' if defect else 'ok'}/pieza_{i:03d}.png"
        files[f"uc04_defects/{name}"] = png_bytes(pixels)
        images.append({"id": i, "file_name": name, "width": 32, "height": 32})
        if box:
            annotations.append(
                {
                    "id": len(annotations),
                    "image_id": i,
                    "category_id": 1,
                    "bbox": list(box),
                    "area": box[2] * box[3],
                    "iscrowd": 0,
                }
            )
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "defecto"}],
    }
    files["uc04_defects/annotations_coco.json"] = (
        json.dumps(coco, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return files


def uc05_masks(rng: random.Random) -> Files:
    """Imagen: segmentación (imagen + máscara PNG, 0 = fondo, 255 = daño)."""
    files: Files = {}
    for i in range(20):
        size = 32
        cx, cy, r = rng.randint(8, 24), rng.randint(8, 24), rng.randint(3, 7)
        mask = [
            [255 if (x - cx) ** 2 + (y - cy) ** 2 <= r * r else 0 for x in range(size)]
            for y in range(size)
        ]
        img = [
            [
                (90, 60, 40) if mask[y][x] else (170 + rng.randint(-10, 10), 170, 175)
                for x in range(size)
            ]
            for y in range(size)
        ]
        files[f"uc05_masks/images/foto_{i:03d}.png"] = png_bytes(img)
        files[f"uc05_masks/masks/foto_{i:03d}.png"] = png_gray(mask)
    return files


# Fuente 3x5 para dígitos (OCR sintético de remitos).
_DIGITS = {
    "0": ["111", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "111"],
    "2": ["111", "001", "111", "100", "111"],
    "3": ["111", "001", "111", "001", "111"],
    "4": ["101", "101", "111", "001", "001"],
    "5": ["111", "100", "111", "001", "111"],
    "6": ["111", "100", "111", "101", "111"],
    "7": ["111", "001", "010", "010", "010"],
    "8": ["111", "101", "111", "101", "111"],
    "9": ["111", "101", "111", "001", "111"],
    "-": ["000", "000", "111", "000", "000"],
}


def uc06_ocr(rng: random.Random) -> Files:
    """Imagen: OCR de números de remito (texto renderizado con fuente de píxeles)."""
    files: Files = {}
    labels = []
    scale = 2
    for i in range(20):
        text = f"{rng.randint(1, 9999):04d}-{rng.randint(0, 99999999):08d}"
        width, height = (len(text) * 4 + 2) * scale, 9 * scale
        img = [[(245, 245, 240) for _ in range(width)] for _ in range(height)]
        for k, ch in enumerate(text):
            for gy, line in enumerate(_DIGITS[ch]):
                for gx, bit in enumerate(line):
                    if bit == "1":
                        for sy in range(scale):
                            for sx in range(scale):
                                img[(2 + gy) * scale + sy][(1 + k * 4 + gx) * scale + sx] = (
                                    20,
                                    20,
                                    30,
                                )
        name = f"remito_{i:03d}.png"
        files[f"uc06_ocr/images/{name}"] = png_bytes(img)
        labels.append([name, text])
    files["uc06_ocr/labels.csv"] = csv_bytes(["file_name", "texto"], labels)
    return files


def uc07_demand(rng: random.Random) -> Files:
    """Serie temporal multi-serie: demanda semanal por SKU (tendencia + estacionalidad)."""
    rows = []
    start = date(2024, 1, 1)
    for s in range(5):
        base, trend, amp = rng.uniform(80, 200), rng.uniform(-0.2, 0.6), rng.uniform(10, 30)
        for w in range(104):
            value = base + trend * w + amp * math.sin(2 * math.pi * w / 52) + rng.gauss(0, 5)
            rows.append(
                [f"SKU-{s:03d}", (start + timedelta(weeks=w)).isoformat(), max(0, round(value))]
            )
    return {"uc07_demand/demanda.csv": csv_bytes(["sku", "semana", "unidades"], rows)}


def uc08_telemetry(rng: random.Random) -> Files:
    """Serie temporal: telemetría de sensores con anomalías etiquetadas."""
    rows = []
    for t in range(1440):  # 1 día a 1 muestra/min
        temp = 60 + 5 * math.sin(2 * math.pi * t / 1440) + rng.gauss(0, 0.5)
        vib = 0.3 + rng.gauss(0, 0.03)
        anomaly = 0
        if 600 <= t < 615 or 1100 <= t < 1105:
            temp += 12
            vib += 0.5
            anomaly = 1
        ts = f"2026-01-15T{t // 60:02d}:{t % 60:02d}:00"
        rows.append([ts, "S1", round(temp, 3), round(vib, 4), anomaly])
    return {
        "uc08_telemetry/telemetria.csv": csv_bytes(
            ["timestamp", "sensor", "temperatura", "vibracion", "anomalia"], rows
        )
    }


def uc09_motor_audio(rng: random.Random) -> Files:
    """Audio: clips de 1 s a 8 kHz de un motor normal / con falla (armónicos + golpeteo)."""
    rate, n = 8000, 8000
    files: Files = {}
    events = []
    classes = ["normal", "rodamiento", "desbalance", "cavitacion"]
    for i in range(40):
        cls = classes[i % len(classes)]
        f0 = rng.uniform(95, 105)
        samples = []
        for k in range(n):
            t = k / rate
            s = 0.4 * math.sin(2 * math.pi * f0 * t) + 0.1 * math.sin(2 * math.pi * 2 * f0 * t)
            if cls == "rodamiento" and (k % 800) < 40:
                s += 0.5 * math.sin(2 * math.pi * 1800 * t)
            elif cls == "desbalance":
                s *= 1 + 0.5 * math.sin(2 * math.pi * 4 * t)
            elif cls == "cavitacion":
                s += rng.gauss(0, 0.15)
            samples.append(s + rng.gauss(0, 0.02))
        name = f"{cls}/clip_{i:03d}.wav"
        files[f"uc09_motor_audio/{name}"] = wav_bytes(samples, rate)
        if cls != "normal":
            events.append([name, 0.0, 1.0, cls])
    files["uc09_motor_audio/eventos.csv"] = csv_bytes(
        ["file_name", "inicio_s", "fin_s", "etiqueta"], events
    )
    return files


def uc10_stream(rng: random.Random) -> Files:
    """Stream de nuevos clientes para UC-10: los últimos lotes tienen drift sintético."""
    records = []
    for batch in range(10):
        drift = batch >= 7
        for j in range(50):
            tenure = rng.randint(1, 24 if drift else 72)
            tickets = rng.randint(3, 10) if drift else rng.randint(0, 8)
            records.append(
                {
                    "batch": batch,
                    "customer_id": f"N{batch:02d}{j:03d}",
                    "plan": rng.choice(["básico", "estándar", "premium"]),
                    "antiguedad_meses": tenure,
                    "tickets_90d": tickets,
                    "cargo_mensual": round(rng.uniform(15, 100), 2),
                }
            )
    return {"uc10_stream/stream.jsonl": jsonl_bytes(records)}


GENERATORS: list[Callable[[random.Random], Files]] = [
    uc01_churn,
    uc02_tickets_time,
    uc03_tickets_es,
    uc04_defects,
    uc05_masks,
    uc06_ocr,
    uc07_demand,
    uc08_telemetry,
    uc09_motor_audio,
    uc10_stream,
]


def generate_all() -> Files:
    out: Files = {}
    for gen in GENERATORS:
        out.update(gen(random.Random(f"{SEED}:{gen.__name__}")))
    index = {path: hashlib.sha256(data).hexdigest() for path, data in sorted(out.items())}
    out["SHA256SUMS.json"] = (json.dumps(index, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verifica sin escribir")
    args = parser.parse_args(argv)

    files = generate_all()
    mismatched = []
    for rel, data in files.items():
        target = ROOT / rel
        if args.check:
            if not target.is_file() or target.read_bytes() != data:
                mismatched.append(rel)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    if mismatched:
        print(
            f"{len(mismatched)} fixtures desactualizados, p. ej.: {mismatched[:5]}", file=sys.stderr
        )
        return 1
    print(f"{len(files)} archivos {'verificados' if args.check else 'generados'} en {ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
