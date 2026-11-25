"""Train v2: combined Donate-a-Cry + InfantCry-DBL + DynamicSuperb.

Expanded to 7 classes: hungry, sleepy, discomfort, pain, burp,
lonely, startle. (temperature excluded — only 2 samples)
"""
import glob, os, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, balanced_accuracy_score,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SVMSMOTE
import joblib

DONATE_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--Nooon--Donate_a_cry/snapshots/"
    "491487c4258287aa2be7cc179b7928768be7008d")
DBL_DIR = "/Users/kl/Documents/Codex/2026-11-23/you-2/work/cry-agent/data/infantcry-dbl-v2/dunstan  337"
DS_DIR = "/Users/kl/Documents/Codex/2026-11-23/you-2/work/cry-agent/data/dynamic-superb"
MODEL_DIR = os.path.expanduser("~/.config/cry-agent/models")
SR = 16000
SEG_DUR = 2.0

DONATE_MAP = {"belly_pain":"pain","burping":"burp","discomfort":"discomfort","hungry":"hungry","tired":"sleepy"}
DUNSTAN_MAP = {"neh":"hungry","owh":"sleepy","heh":"discomfort","eairh":"pain","eh":"burp"}
# DynamicSuperb already mapped by directory name

def extract_features(y, sr=SR, loudness="peak"):
    if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR); sr = SR
    if len(y) < sr * 0.3: return None
    if loudness == "rms":
        rms = np.sqrt(np.mean(y ** 2)) + 1e-8
        y = y / rms * 0.1
    else:
        y = y / (np.max(np.abs(y)) + 1e-8)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    d1 = librosa.feature.delta(mfcc); d2 = librosa.feature.delta(mfcc, order=2)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    ro = librosa.feature.spectral_rolloff(y=y, sr=sr)
    fl = librosa.feature.spectral_flatness(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    f0, voiced, _ = librosa.pyin(y=y, sr=sr, fmin=80, fmax=500)
    f0v = f0[voiced] if voiced.any() else np.array([0])
    return np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        d1.mean(axis=1), d2.mean(axis=1),
        [cent.mean(),cent.std()],[bw.mean(),bw.std()],[ro.mean(),ro.std()],
        [fl.mean(),fl.std()],[zcr.mean(),zcr.std()],[rms.mean(),rms.std()],
        [f0v.mean() if len(f0v)>0 else 0, f0v.std() if len(f0v)>0 else 0,
         voiced.sum()/len(voiced) if len(voiced)>0 else 0],
    ])

def segs(y, sr=SR):
    sl = int(sr*SEG_DUR)
    if len(y)<sl: y=np.pad(y,(0,sl-len(y))); return [y]
    s=[]; hop=sl//2
    for i in range(0,len(y)-sl+1,hop): s.append(y[i:i+sl])
    return s

def load_dir(base, label_map=None):
    samples, labels = [], []
    for cn in os.listdir(base):
        cd = os.path.join(base, cn)
        if not os.path.isdir(cd): continue
        mapped = label_map.get(cn, cn) if label_map else cn
        for wf in glob.glob(os.path.join(cd, "*.wav")):
            try:
                y,sr = librosa.load(wf, sr=SR)
                for s in segs(y,sr):
                    f = extract_features(s,sr)
                    if f is not None: samples.append(f); labels.append(mapped)
            except: pass
    return samples, labels

def train():
    print("="*60)
    print("COMBINED V2: Donate-a-Cry + InfantCry-DBL + DynamicSuperb")
    print("="*60)
    print("\n[1/3] Loading all datasets...")
    t0 = time.time()
    s1,l1 = load_dir(DONATE_DIR, DONATE_MAP)
    s2,l2 = load_dir(DBL_DIR, DUNSTAN_MAP)
    s3,l3 = load_dir(DS_DIR)
    X = np.array(s1+s2+s3)
    y = np.array(l1+l2+l3)
    from collections import Counter
    print(f"  Segments: {len(X)}  Features: {X.shape[1]}  Time: {time.time()-t0:.0f}s")
    print(f"  Sources: donate={len(s1)}, dbl={len(s2)}, dynamic={len(s3)}")
    print(f"  Distribution: {Counter(y)}")

    le = LabelEncoder()
    ye = le.fit_transform(y)
    print(f"  Classes: {list(le.classes_)}")

    print(f"\n[2/3] Cross-validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    yp = np.zeros(len(ye), dtype=int)
    for fold,(tr,te) in enumerate(skf.split(X,ye)):
        pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SVMSMOTE(random_state=42, k_neighbors=min(3,min(np.bincount(ye[tr]))-1))),
            ("clf", SVC(C=1.0,kernel="rbf",gamma="scale",class_weight="balanced",probability=True,random_state=42)),
        ])
        pipe.fit(X[tr],ye[tr]); yp[te]=pipe.predict(X[te])

    acc = accuracy_score(ye,yp); bal = balanced_accuracy_score(ye,yp)
    print(f"  Accuracy:     {acc*100:.1f}%")
    print(f"  Balanced Acc: {bal*100:.1f}%")
    print()
    print(classification_report(ye,yp,target_names=le.classes_,zero_division=0))
    cm = confusion_matrix(ye,yp)
    print("Confusion Matrix:")
    print(f"  {'':12s}", "  ".join(f"{c[:4]:>5s}" for c in le.classes_))
    for i,row in enumerate(cm):
        print(f"  {le.classes_[i]:12s}", "  ".join(f"{v:5d}" for v in row))

    print(f"\n[3/3] Saving...")
    final = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SVMSMOTE(random_state=42, k_neighbors=min(3,min(np.bincount(ye))-1))),
        ("clf", SVC(C=1.0,kernel="rbf",gamma="scale",class_weight="balanced",probability=True,random_state=42)),
    ])
    final.fit(X,ye)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(final, os.path.join(MODEL_DIR, "cry_classifier.joblib"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))
    with open(os.path.join(MODEL_DIR,"model_config.json"),"w") as f:
        json.dump({"model_type":"svm_rbf","training":"donate+dbl+dynamic",
            "n_features":X.shape[1],"classes":list(le.classes_),
            "cv_accuracy":round(acc,3),"cv_balanced_accuracy":round(bal,3),
            "sample_rate":SR}, f, indent=2)
    print(f"\n{'='*60}")
    print(f"DONE — Accuracy: {acc*100:.1f}%, Balanced: {bal*100:.1f}%")
    print(f"{'='*60}")

if __name__ == "__main__":
    train()
