"""Tests for per-family calibration (adapt.FamilyCalibrator)."""
import os
import tempfile

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from adapt import CoralAdapter, FamilyCalibrator, PrototypeAdapter


def test_synthetic_shift():
    rng = np.random.default_rng(0)
    n = 200
    X0 = rng.normal([0, 0], 1.0, size=(n, 2))
    X1 = rng.normal([3, 0], 1.0, size=(n, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * n + [1] * n)

    order = rng.permutation(len(X))
    X, y = X[order], y[order]
    split = int(len(X) * 0.7)
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]

    scaler = StandardScaler().fit(X_tr)
    model = LogisticRegression().fit(scaler.transform(X_tr), y_tr)

    # Target: same classes, shifted and stretched by the recording setup.
    X_t = (X_te * 2.0) + np.array([10.0, 5.0])
    global_pred = model.predict(scaler.transform(X_t))
    global_bal = balanced_accuracy_score(y_te, global_pred)
    assert global_bal < 0.65, global_bal

    coral = CoralAdapter().fit(scaler.transform(X_tr), scaler.transform(X_t))
    coral_pred = model.predict(coral.transform(scaler.transform(X_t)))
    assert balanced_accuracy_score(y_te, coral_pred) > 0.9

    idx0 = np.where(y_te == 0)[0][:20]
    idx1 = np.where(y_te == 1)[0][:20]
    idx = np.concatenate([idx0, idx1])
    proto = PrototypeAdapter([0, 1], metric="euclidean").fit(
        scaler.transform(X_t[idx]), y_te[idx]
    )
    assert balanced_accuracy_score(
        y_te, proto.predict(scaler.transform(X_t))
    ) > 0.9
    return global_bal


def test_calibrator_end_to_end():
    rng = np.random.default_rng(1)
    n = 120
    X_tr = np.vstack([
        rng.normal([0, 0], 1.0, size=(n, 2)),
        rng.normal([3, 0], 1.0, size=(n, 2)),
    ])
    y_tr = np.array([0] * n + [1] * n)
    scaler = StandardScaler().fit(X_tr)
    model = LogisticRegression().fit(scaler.transform(X_tr), y_tr)

    # Family audio: shifted, 20 labelled clips/class + unlabeled pool.
    X_fam = np.vstack([
        rng.normal([10, 5], 1.0, size=(30, 2)),
        rng.normal([13, 5], 1.0, size=(30, 2)),
    ])
    y_fam = np.array([0] * 30 + [1] * 30)
    files = [f"rec_{i // 2}.wav" for i in range(60)]
    source_mean = scaler.transform(X_tr).mean(axis=0)
    source_cov = np.cov(scaler.transform(X_tr), rowvar=False)

    calibrator = FamilyCalibrator.fit(
        model, scaler, ["a", "b"],
        X_fam, ["a"] * 30 + ["b"] * 30, files=files,
        X_unlabeled=X_fam,
        source_mean=source_mean, source_cov=source_cov,
        prototype_metric="euclidean",
    )
    assert calibrator.method in ("coral", "proto", "coral-proto"), calibrator.method

    proba = calibrator.predict_proba(X_fam)
    pred = np.argmax(proba, axis=1)
    assert balanced_accuracy_score(y_fam, pred) > 0.85

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "adapter.joblib")
        joblib.dump(calibrator, path)
        loaded = joblib.load(path).bind(model, scaler)
        np.testing.assert_allclose(
            loaded.predict_proba(X_fam), proba, rtol=1e-6
        )
    return calibrator.method


def test_real_dbl_holdout():
    cache = os.path.expanduser("~/.cache/cry-agent/loco_features_v1.npz")
    if not os.path.exists(cache):
        print("  (skipped: LOCO feature cache missing)")
        return None

    from test_adaptation import build_global_model
    from test_loco import CLASSES

    data = np.load(cache, allow_pickle=False)
    X, labels, sources, files = (
        data["X"], data["y"], data["source"], data["files"]
    )
    keep = np.isin(labels, CLASSES)
    X, labels, sources, files = X[keep], labels[keep], sources[keep], files[keep]
    index = {c: i for i, c in enumerate(CLASSES)}
    y = np.array([index[str(v)] for v in labels])

    train = np.where(sources != "dbl")[0]
    pool = np.where(sources == "dbl")[0]
    scaler, model = build_global_model(X[train], y[train])

    rng = np.random.default_rng(7)
    cal_idx, test_idx = [], []
    for cls in range(len(CLASSES)):
        rows = pool[y[pool] == cls]
        names = sorted({str(files[i]) for i in rows})
        rng.shuffle(names)
        cut = max(1, int(len(names) * 0.3))
        cal_files = set(names[:cut])
        cal_idx.extend(i for i in rows if str(files[i]) in cal_files)
        test_idx.extend(i for i in rows if str(files[i]) not in cal_files)
    cal_idx, test_idx = np.array(cal_idx), np.array(test_idx)

    # Source moments must come from the training corpora only; using the
    # shipped all-corpus file here would leak the held-out family.
    Xs_train = scaler.transform(X[train])
    mean = Xs_train.mean(axis=0)
    cov = np.cov(Xs_train, rowvar=False)

    calibrator = FamilyCalibrator.fit(
        model, scaler, CLASSES,
        X[cal_idx], y[cal_idx],
        files=[files[i] for i in cal_idx],
        X_unlabeled=X[pool],
        source_mean=mean, source_cov=cov,
    )
    global_bal = balanced_accuracy_score(
        y[test_idx],
        np.argmax(model.predict_proba(scaler.transform(X[test_idx])), axis=1),
    )
    family_bal = balanced_accuracy_score(
        y[test_idx],
        np.argmax(calibrator.predict_proba(X[test_idx]), axis=1),
    )
    print(f"  DBL holdout: global={global_bal:.1%} "
          f"family={family_bal:.1%} method={calibrator.method}")
    assert family_bal > global_bal + 0.15, (global_bal, family_bal)
    return global_bal, family_bal


if __name__ == "__main__":
    print("synthetic shift ...")
    base = test_synthetic_shift()
    print(f"  global balanced={base:.1%}  (coral fixed it)")
    print("calibrator end-to-end ...")
    method = test_calibrator_end_to_end()
    print(f"  selected={method}")
    print("real DBL holdout ...")
    result = test_real_dbl_holdout()
    print("Family calibrator tests passed.")
