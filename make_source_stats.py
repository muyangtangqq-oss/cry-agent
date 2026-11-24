"""Build compact source statistics for CORAL family calibration.

The production model is shipped as an SVM plus a StandardScaler. CORAL
also needs the mean and covariance of the source features in that scaled
space. ``train_augmented.py`` writes them on the next retrain; this script
backfills them for an existing model from the cached LOCO features.

Usage:
  python make_source_stats.py
  python make_source_stats.py --model-dir ~/.config/cry-agent/models
"""
import argparse
import os

import joblib
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        default=os.path.expanduser("~/.cache/cry-agent/loco_features_v1.npz"),
    )
    parser.add_argument(
        "--model-dir",
        default=os.path.expanduser("~/.config/cry-agent/models"),
    )
    args = parser.parse_args()

    data = np.load(os.path.expanduser(args.cache), allow_pickle=False)
    X = data["X"]

    model = joblib.load(
        os.path.join(os.path.expanduser(args.model_dir),
                     "cry_classifier.joblib")
    )
    scaler = model.named_steps["scaler"]
    Xs = scaler.transform(X)

    out = os.path.join(os.path.expanduser(args.model_dir), "source_stats.npz")
    np.savez_compressed(
        out,
        mean=Xs.mean(axis=0).astype(np.float32),
        cov=np.cov(Xs, rowvar=False).astype(np.float32),
    )
    print(f"source features: {Xs.shape}")
    print(f"saved {out} ({os.path.getsize(out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
