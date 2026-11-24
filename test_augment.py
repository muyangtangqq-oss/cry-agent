"""Sanity tests for audio augmentation."""
import numpy as np

from augment import add_noise, augment, generate_variants, pitch_shift, time_stretch

SR = 16000


def _tone():
    t = np.arange(SR, dtype=np.float32) / SR
    return 0.4 * np.sin(2 * np.pi * 300 * t).astype(np.float32)


def test_shapes_and_finiteness():
    y = _tone()
    for i in range(8):
        out = augment(y, SR, seed=i)
        assert out.shape == y.shape
        assert out.dtype == np.float32
        assert np.isfinite(out).all()
        assert np.max(np.abs(out)) <= 1.0 + 1e-6
    print("  shapes/finiteness OK")


def test_determinism():
    y = _tone()
    a = augment(y, SR, seed=7)
    b = augment(y, SR, seed=7)
    c = augment(y, SR, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    print("  determinism OK")


def test_transforms_change_signal():
    y = _tone()
    stretched = time_stretch(y, SR, 0.8)
    assert len(stretched) == len(y)
    shifted = pitch_shift(y, SR, 3)
    noisy = add_noise(y, SR, 10, np.random.default_rng(0))
    assert not np.allclose(shifted, y)
    assert np.mean(noisy ** 2) > np.mean(y ** 2)
    print("  pitch/stretch/noise OK")


def test_generate_variants():
    variants = generate_variants(_tone(), SR, 4, seed=0)
    assert len(variants) == 4
    assert len({v.tobytes() for v in variants}) == 4
    print("  variants OK")


if __name__ == "__main__":
    test_shapes_and_finiteness()
    test_determinism()
    test_transforms_change_signal()
    test_generate_variants()
    print("All augmentation tests passed.")
