"""Integration test: pilot feedback -> family adapter -> inference."""
import glob
import os
import tempfile
from types import SimpleNamespace

import joblib
import numpy as np

from adapt import FamilyCalibrator
from pilot import PilotStore

DONATE = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--Nooon--Donate_a_cry/snapshots/"
    "491487c4258287aa2be7cc179b7928768be7008d"
)
LABELS = {
    "belly_pain": "pain", "burping": "burp", "discomfort": "discomfort",
    "hungry": "hungry", "tired": "sleepy",
}


def fake(cause):
    return SimpleNamespace(
        cause=cause, confidence=0.5, model="fusion",
        fused_cause=cause, fused_confidence=0.5,
        audio_cause=cause, audio_confidence=0.5,
        vision_cause="unknown", vision_confidence=0.0,
        escalated=False, critical_override=False,
        audio_ood=False, session_ood=False, latency_ms=10.0,
        trusted=True, scenario="baby_cry", per_signal={},
    )


def main():
    if not os.path.isdir(DONATE):
        print("skipped: Donate-a-Cry not downloaded")
        return

    clips = []
    for name in sorted(os.listdir(DONATE)):
        label = LABELS.get(name)
        folder = os.path.join(DONATE, name)
        if not label or not os.path.isdir(folder):
            continue
        for path in sorted(glob.glob(os.path.join(folder, "*.wav")))[:2]:
            clips.append((path, label))
    if len(clips) < 10:
        print(f"skipped: only {len(clips)} clips")
        return

    with tempfile.TemporaryDirectory() as tmp:
        store = PilotStore(tmp)
        family = store.add_family(42, "pilot-1.0", True, "calibration test")
        for path, label in clips:
            event = store.log_result(family, fake(label), audio_path=path)
            store.record_feedback(event, actual_cause=label)

        labeled = store.labeled_audio(family)
        assert len(labeled) == len(clips)

        from audio_inference import CryAudioClassifier
        clf = CryAudioClassifier()
        X_labeled, y_labeled, files = [], [], []
        for path, label in labeled:
            for feat in clf.features_from_file(path):
                X_labeled.append(feat)
                y_labeled.append(label)
                files.append(path)
        X_unlabeled = []
        for path in store.family_audio(family):
            X_unlabeled.extend(clf.features_from_file(path))

        stats_path = os.path.join(
            os.path.dirname(clf.model_path), "source_stats.npz"
        )
        mean = cov = None
        if os.path.exists(stats_path):
            stats = np.load(stats_path)
            mean, cov = stats["mean"], stats["cov"]

        calibrator = FamilyCalibrator.fit(
            clf.pipeline, clf.pipeline.named_steps["scaler"], clf.classes,
            X_labeled, y_labeled, files=files, X_unlabeled=X_unlabeled,
            source_mean=mean, source_cov=cov,
        )
        out = os.path.join(tmp, "adapters", f"{family}.joblib")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        joblib.dump(calibrator, out)
        assert os.path.getsize(out) < 2_000_000, os.path.getsize(out)

        clf.load_family_adapter(out)
        result = clf.classify_file(clips[0][0])
        assert result["cause"] in clf.classes
        print(f"clips={len(clips)} segments={len(X_labeled)} "
              f"method={calibrator.method} size={os.path.getsize(out)//1024}KB")
        print("Pilot calibration test passed.")


if __name__ == "__main__":
    main()
