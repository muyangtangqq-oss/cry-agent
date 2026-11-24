"""Audio augmentation for minority cry classes.

All transforms operate on time-domain float arrays so they can be applied
inside a CV fold (training only) instead of leaking augmented copies into
validation.
"""
import numpy as np
import librosa


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
