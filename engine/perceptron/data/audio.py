"""Audio (RF-ING-01 audio, RF-PRF-05, RF-PIP-03 audio). ADR-0018.

IO con `soundfile` (libsndfile: wav/flac/ogg/mp3). DSP propio sobre `torch.stft`
(mel, MFCC, SpecAugment) y resample con `scipy.signal.resample_poly`: sin
depender de torchaudio, que está en modo mantenimiento.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

AUDIO_EXTENSIONS = frozenset({".wav", ".flac", ".mp3", ".ogg"})
SILENCE_DBFS = -50.0
CLIP_LEVEL = 0.999


@dataclass(frozen=True)
class AudioInfo:
    sample_rate: int
    channels: int
    frames: int
    format: str

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate if self.sample_rate else 0.0


def info(path: Path) -> AudioInfo:
    import soundfile as sf

    i = sf.info(str(path))
    return AudioInfo(
        sample_rate=int(i.samplerate),
        channels=int(i.channels),
        frames=int(i.frames),
        format=str(i.format),
    )


def load(
    path: Path, *, sample_rate: int | None = None, mono: bool = True
) -> tuple[np.ndarray, int]:
    """Audio float32 en [-1, 1], forma [canales, muestras] (o [1, muestras] si mono)."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = data.T  # [C, N]
    if mono and x.shape[0] > 1:
        x = x.mean(axis=0, keepdims=True)
    if sample_rate and sr != sample_rate:
        x = resample(x, sr, sample_rate)
        sr = sample_rate
    return x.astype(np.float32), int(sr)


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    from scipy.signal import resample_poly

    frac = Fraction(sr_to, sr_from).limit_denominator(1000)
    out: np.ndarray = resample_poly(x, frac.numerator, frac.denominator, axis=-1)
    return out.astype(np.float32)


def fix_length(x: np.ndarray, samples: int) -> np.ndarray:
    """Recorta o rellena con ceros al final hasta `samples`."""
    if x.shape[-1] >= samples:
        return x[..., :samples]
    return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, samples - x.shape[-1])])


def frame_rms_db(x: np.ndarray, frame: int = 1024) -> np.ndarray:
    mono = x.mean(axis=0) if x.ndim > 1 else x
    n = max(len(mono) // frame, 1)
    frames = mono[: n * frame].reshape(n, -1) if len(mono) >= frame else mono[None, :]
    rms = np.sqrt((frames**2).mean(axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def quality(x: np.ndarray) -> dict[str, float]:
    """Silencio, clipping y SNR estimado (percentil 90 vs 10 de energía por frame, en dB)."""
    db = frame_rms_db(x)
    return {
        "silence_fraction": float((db < SILENCE_DBFS).mean()),
        "clipping_fraction": float((np.abs(x) >= CLIP_LEVEL).mean()),
        "snr_db": float(np.percentile(db, 90) - np.percentile(db, 10)),
    }


# ------------------------------------------------------------------ features


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


@lru_cache(maxsize=16)
def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> torch.Tensor:
    """Banco de filtros triangulares HTK [n_mels, n_fft // 2 + 1]."""
    fmax = sample_rate / 2
    mels = np.linspace(_hz_to_mel(np.array(0.0)), _hz_to_mel(np.array(fmax)), n_mels + 2)
    hz = _mel_to_hz(mels)
    bins = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    fb = np.zeros((n_mels, len(bins)), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = hz[i], hz[i + 1], hz[i + 2]
        up = (bins - lo) / max(mid - lo, 1e-9)
        down = (hi - bins) / max(hi - mid, 1e-9)
        fb[i] = np.clip(np.minimum(up, down), 0, None)
    return torch.from_numpy(fb)


def stft_params(sample_rate: int, win_ms: float = 32.0, hop_ms: float = 10.0) -> tuple[int, int]:
    n_fft = 2 ** math.ceil(math.log2(sample_rate * win_ms / 1000))
    return n_fft, max(1, int(sample_rate * hop_ms / 1000))


def log_mel(x: np.ndarray | torch.Tensor, sample_rate: int, n_mels: int = 64) -> torch.Tensor:
    """[1, N] → log-mel [1, n_mels, frames]."""
    wav = torch.as_tensor(x, dtype=torch.float32).reshape(-1)
    n_fft, hop = stft_params(sample_rate)
    spec = torch.stft(
        wav, n_fft=n_fft, hop_length=hop, window=torch.hann_window(n_fft), return_complex=True
    )
    power = spec.abs() ** 2
    mel = mel_filterbank(sample_rate, n_fft, n_mels) @ power
    return torch.log(mel + 1e-6).unsqueeze(0)


def mfcc(
    x: np.ndarray | torch.Tensor, sample_rate: int, n_mfcc: int = 20, n_mels: int = 64
) -> torch.Tensor:
    lm = log_mel(x, sample_rate, n_mels)[0]  # [M, T]
    n = torch.arange(n_mels, dtype=torch.float32)
    k = torch.arange(n_mfcc, dtype=torch.float32)[:, None]
    dct = torch.cos(math.pi / n_mels * (n + 0.5) * k)  # DCT-II
    return (dct @ lm).unsqueeze(0)


def frames_for(duration_s: float, sample_rate: int) -> int:
    _, hop = stft_params(sample_rate)
    return round(duration_s * sample_rate) // hop + 1


def spec_augment(
    s: torch.Tensor, freq_mask: int, time_mask: int, rng: torch.Generator | None = None
) -> torch.Tensor:
    """SpecAugment (Park et al., 2019): una máscara de frecuencia y una de tiempo.

    La zona tapada se rellena con el espectro promedio del propio clip (media por banda
    en la máscara temporal; media por instante en la de frecuencia). Rellenar con una
    constante global fabrica un "pozo" de energía que imita patrones reales (p. ej. una
    modulación de amplitud) y confunde clases que solo se distinguen en el tiempo.
    """
    s = s.clone()
    _, m, t = s.shape
    if freq_mask and m > 1:
        w = int(torch.randint(0, freq_mask + 1, (1,), generator=rng))
        f0 = int(torch.randint(0, max(m - w, 1), (1,), generator=rng))
        per_frame = s.mean(dim=1, keepdim=True)  # [C, 1, T]
        s[:, f0 : f0 + w, :] = per_frame.expand(-1, w, -1)[:, : s[:, f0 : f0 + w, :].shape[1], :]
    if time_mask and t > 1:
        w = int(torch.randint(0, time_mask + 1, (1,), generator=rng))
        t0 = int(torch.randint(0, max(t - w, 1), (1,), generator=rng))
        per_band = s.mean(dim=2, keepdim=True)  # [C, M, 1]
        s[:, :, t0 : t0 + w] = per_band.expand(-1, -1, w)[:, :, : s[:, :, t0 : t0 + w].shape[2]]
    return s
