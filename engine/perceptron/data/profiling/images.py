"""Profiling de imágenes (RF-PRF-02): resoluciones, canales, formatos, corruptas
y casi duplicados por hash perceptual (dHash)."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image

from perceptron.data.profiling.card import K_ANONYMITY, CategoryCount, ImageProfile, sig

HASH_SIZE = 8
NEAR_DUP_MAX_DISTANCE = 3  # con 4 bandas de 16 bits, distancia ≤ 3 garantiza compartir una banda
_BANDS = 4


def dhash(path: Path) -> int | None:
    """Difference hash de 64 bits."""
    try:
        with Image.open(path) as im:
            small = im.convert("L").resize((HASH_SIZE + 1, HASH_SIZE), Image.Resampling.LANCZOS)
    except OSError:
        return None
    px = np.asarray(small, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def near_duplicate_pairs(hashes: list[int]) -> list[tuple[int, int]]:
    """Pares de índices con distancia de Hamming ≤ NEAR_DUP_MAX_DISTANCE (LSH por bandas)."""
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, h in enumerate(hashes):
        for b in range(_BANDS):
            buckets[(b, (h >> (16 * b)) & 0xFFFF)].append(i)
    pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        for i, j in combinations(members, 2):
            if (hashes[i] ^ hashes[j]).bit_count() <= NEAR_DUP_MAX_DISTANCE:
                pairs.add((i, j))
    return sorted(pairs)


def _counts(s: pl.Series) -> list[CategoryCount]:
    vc = s.drop_nulls().cast(pl.String).value_counts(sort=True, name="n")
    return [CategoryCount(value=v, count=int(n)) for v, n in vc.iter_rows() if n >= K_ANONYMITY]


def _quantiles(s: pl.Series) -> dict[str, float | None]:
    x = s.drop_nulls()
    if x.is_empty():
        return {"p05": None, "p50": None, "p95": None}
    return {
        k: sig(float(x.quantile(v) or 0)) for k, v in (("p05", 0.05), ("p50", 0.5), ("p95", 0.95))
    }


def profile_images(index: pl.DataFrame, files_dir: Path) -> ImageProfile:
    ok = index.filter(~pl.col("corrupt"))
    hashes = [h for p in ok["path"] if (h := dhash(files_dir / p)) is not None]
    pairs = near_duplicate_pairs(hashes)
    in_pairs = {i for pair in pairs for i in pair}
    res = ok.select(
        (pl.col("width").cast(pl.String) + "x" + pl.col("height").cast(pl.String)).alias("r")
    )
    return ImageProfile(
        count=index.height,
        corrupt=index.filter(pl.col("corrupt")).height,
        resolutions=_counts(res["r"]),
        width_quantiles=_quantiles(ok["width"]),
        height_quantiles=_quantiles(ok["height"]),
        channels=_counts(ok["channels"]),
        formats=_counts(ok["format"]),
        near_duplicate_pairs=len(pairs),
        near_duplicate_fraction=round(len(in_pairs) / max(len(hashes), 1), 4),
    )
