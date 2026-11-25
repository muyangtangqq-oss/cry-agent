"""Audio augmentation for minority cry classes.

All transforms operate on time-domain float arrays so they can be applied
inside a CV fold (training only) instead of leaking augmented copies into
validation.
"""
import numpy as np
import librosa
from scipy.signal import fftconvolve


def _fit_length(y: np.ndarray, n: int) -> np.ndarray:
    if len(y) == n:
        return y
    if len(y) > n:
        return y[:n]
    return np.pad(y, (0, n - len(y)))


def pitch_shift(y, sr, semitones):
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=float(semitones))


def time_stretch(y, sr, rate):
    stretched = librosa.effects.time_stretch(y=y, rate=float(rate))
    return _fit_length(stretched, len(y))


def add_noise(y, sr, snr_db, rng):
    signal_power = np.mean(y ** 2) + 1e-10
    noise_power = signal_power / (10 ** (float(snr_db) / 10))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=len(y))
    return y + noise


def random_gain(y, db):
    return y * (10 ** (float(db) / 20))


def spectral_eq(y, sr, rng, max_db=6.0):
    """Smooth random microphone/room EQ in the frequency domain."""
    n = len(y)
    spec = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    lo = 80.0
    hi = min(sr / 2 * 0.9, 7000.0)
    points = np.geomspace(lo, hi, 5)
    gains_db = rng.uniform(-max_db, max_db, len(points))
    curve = np.interp(
        np.log(np.maximum(freqs, 1e-6)),
        np.log(points),
        gains_db,
        left=gains_db[0],
        right=gains_db[-1],
    )
    return np.fft.irfft(spec * (10 ** (curve / 20.0)), n=n)


def bandlimit(y, sr, rng):
    """Soft high-pass / low-pass to mimic small mics and codecs."""
    n = len(y)
    spec = np.fft.rfft(y)
    freqs = np.maximum(np.fft.rfftfreq(n, 1.0 / sr), 1e-6)
    hp = rng.uniform(50.0, 180.0)
    lp = rng.uniform(3800.0, min(7000.0, sr / 2 * 0.9))
    mask = (1.0 / (1.0 + (hp / freqs) ** 4)) * (
        1.0 / (1.0 + (freqs / lp) ** 4)
    )
    return np.fft.irfft(spec * mask, n=n)


def add_reverb(y, sr, rng, max_ms=300.0):
    """Convolve with a short synthetic room impulse response."""
    length = max(1, int(sr * rng.uniform(50.0, max_ms) / 1000.0))
    rir = rng.normal(size=length) * np.exp(
        -np.linspace(0.0, 6.0, length)
    )
    rir[0] = 1.0
    rir /= np.sqrt(np.sum(rir ** 2)) + 1e-8
    wet = fftconvolve(y, rir, mode="full")[:len(y)]
    mix = rng.uniform(0.10, 0.35)
    return (1.0 - mix) * y + mix * wet


def soft_clip(y, rng):
    """Random gain followed by a soft saturation curve."""
    gain = 10 ** (rng.uniform(-6.0, 9.0) / 20.0)
    threshold = rng.uniform(0.7, 1.2)
    return np.tanh(y * gain / threshold) * threshold


def channel_augment(y, sr, seed=0):
    """Channel/gain augmentation that survives feature extraction.

    Unlike `random_gain`, these transforms change the spectral shape (EQ,
    band limit, reverb) or introduce saturation, so peak/RMS normalization
    cannot undo them.
    """
    rng = np.random.default_rng(seed)
    out = np.asarray(y, dtype=np.float32).copy()
    out = spectral_eq(out, sr, rng)
    if rng.random() < 0.60:
        out = bandlimit(out, sr, rng)
    if rng.random() < 0.35:
        out = add_reverb(out, sr, rng)
    if rng.random() < 0.70:
        out = soft_clip(out, rng)
    if not np.isfinite(out).all():
        return np.asarray(y, dtype=np.float32).copy()
    return out.astype(np.float32)


def augment(y, sr, seed=0, strength=1.0):
    """Apply a random chain of pitch / stretch / gain / noise.

    `strength` scales the ranges (1.0 = training default, 0.5 = mild).
    """
    rng = np.random.default_rng(seed)
    out = np.asarray(y, dtype=np.float32).copy()

    if rng.random() < 0.85:
        out = pitch_shift(out, sr, rng.uniform(-3.0, 3.0) * strength)
    if rng.random() < 0.70:
        out = time_stretch(out, sr, rng.uniform(0.90, 1.10) ** strength)
    if rng.random() < 0.80:
        out = random_gain(out, rng.uniform(-6.0, 3.0) * strength)
    if rng.random() < 0.85:
        out = add_noise(out, sr, rng.uniform(15.0, 35.0), rng)

    peak = np.max(np.abs(out)) + 1e-8
    if not np.isfinite(out).all():
        return np.asarray(y, dtype=np.float32).copy()
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32)


def generate_variants(y, sr, count, seed=0, strength=1.0):
    """Return `count` deterministic augmented copies of one segment."""
    return [
        augment(y, sr, seed=seed + i * 7919, strength=strength)
        for i in range(count)
    ]
