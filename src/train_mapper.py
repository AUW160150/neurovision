import json
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors

EMB = "embeds/gallery_real.npy"
META = "embeds/meta_real.json"
EEG = "data/eeg/features_real.npy"
OUT = "models/mapper.joblib"

def l2norm(a):
    n = np.linalg.norm(a, axis=1, keepdims=True) + 1e-8
    return a / n

def main():
    feats = np.load(EMB).astype("float32")         # (N,512) CLIP (already L2-normalized)
    meta = json.loads(Path(META).read_text())
    X = np.load(EEG).astype("float32")             # (N,F) EEG bandpower features

    idx = np.arange(len(feats))
    Xtr, Xte, Ytr, Yte, Itr, Ite = train_test_split(
        X, feats, idx, test_size=0.2, random_state=0, shuffle=True
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ])
    model.fit(Xtr, Ytr)

    Yhat = l2norm(model.predict(Xte).astype("float32"))
    knn = NearestNeighbors(metric="cosine").fit(feats)
    D, I = knn.kneighbors(Yhat, n_neighbors=5)

    labels = np.array([m["label"] for m in meta])
    exact_at1 = (I[:,0] == Ite).mean()
    samecls_at1 = (labels[I[:,0]] == labels[Ite]).mean()

    print(f"Test exact@1:    {exact_at1:.3f}")
    print(f"Test same-class@1:{samecls_at1:.3f}")

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    dump(model, OUT)
    print("Saved mapper:", OUT)

if __name__ == "__main__":
    main()
