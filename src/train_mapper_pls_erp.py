import json
from pathlib import Path
import numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import NearestNeighbors

EMB = "embeds/gallery_real.npy"
META = "embeds/meta_real.json"
EEG = "data/eeg/features_real_erp.npy"   # ERP features
OUT = "models/mapper.joblib"             # overwrite so app uses it

def l2(a):
    n = np.linalg.norm(a, axis=1, keepdims=True) + 1e-8
    return a / n

def main():
    feats = np.load(EMB).astype("float32")      # (N,512)
    meta = json.loads(Path(META).read_text())
    labels = np.array([m["label"] for m in meta])
    X = np.load(EEG).astype("float32")          # (N,8000)

    idx = np.arange(len(feats))
    Xtr, Xte, Ytr, Yte, Itr, Ite = train_test_split(
        X, feats, idx, test_size=0.2, random_state=0, shuffle=True
    )

    comps_grid = [16, 32, 64, 96, 128, 160]
    best = None
    for c in comps_grid:
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("pls", PLSRegression(n_components=c, scale=False, max_iter=1000))
        ])
        pipe.fit(Xtr, Ytr)
        # eval
        Yhat = l2(pipe.predict(Xte).astype("float32"))
        knn = NearestNeighbors(metric="cosine").fit(feats)
        D, I = knn.kneighbors(Yhat, n_neighbors=5)
        exact = (I[:,0] == Ite).mean()
        same  = (labels[I[:,0]] == labels[Ite]).mean()
        print(f"components={c:3d}  exact@1={exact:.3f}  same-class@1={same:.3f}")
        if best is None or (same, exact) > (best[1], best[2]):
            best = (c, same, exact, pipe)

    print(f"BEST: n_components={best[0]}  same@1={best[1]:.3f}  exact@1={best[2]:.3f}")

    # retrain best on ALL data
    final = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("pls", PLSRegression(n_components=best[0], scale=False, max_iter=1000))
    ])
    final.fit(X, feats)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    dump(final, OUT)
    print("Saved mapper:", OUT)

if __name__ == "__main__":
    main()
