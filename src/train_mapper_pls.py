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
EEG = "data/eeg/features_real.npy"
OUT = "models/mapper.joblib"   # overwrite so the app picks it up

def l2(a):
    n = np.linalg.norm(a, axis=1, keepdims=True) + 1e-8
    return a / n

def eval_model(model, Xte, Yte, feats, labels, true_idx):
    Yhat = l2(model.predict(Xte).astype("float32"))
    knn = NearestNeighbors(metric="cosine").fit(feats)
    D, I = knn.kneighbors(Yhat, n_neighbors=5)
    exact_at1 = (I[:,0] == true_idx).mean()
    samecls_at1 = (labels[I[:,0]] == labels[true_idx]).mean()
    return float(exact_at1), float(samecls_at1)

def main():
    feats = np.load(EMB).astype("float32")      # (N,512)
    meta = json.loads(Path(META).read_text())
    labels = np.array([m["label"] for m in meta])
    X = np.load(EEG).astype("float32")          # (N,F)

    idx = np.arange(len(feats))
    Xtr, Xte, Ytr, Yte, Itr, Ite = train_test_split(
        X, feats, idx, test_size=0.2, random_state=0, shuffle=True
    )

    # small grid for components
    comps_grid = [8, 16, 32, 48, 64]
    results = []
    best = None

    for c in comps_grid:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pls", PLSRegression(n_components=c, scale=False, max_iter=1000))
        ])
        pipe.fit(Xtr, Ytr)
        exact, same = eval_model(pipe, Xte, Yte, feats, labels, Ite)
        results.append((c, exact, same))
        if best is None or same > best[1] or (same == best[1] and exact > best[2]):
            best = (c, same, exact, pipe)

    print("components  exact@1  same-class@1")
    for c, ex, sa in results:
        print(f"{c:10d}  {ex:7.3f}   {sa:12.3f}")
    print(f"BEST n_components={best[0]}  same@1={best[1]:.3f} exact@1={best[2]:.3f}")

    # retrain best on ALL data, save
    final = Pipeline([
        ("scaler", StandardScaler()),
        ("pls", PLSRegression(n_components=best[0], scale=False, max_iter=1000))
    ])
    final.fit(X, feats)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    dump(final, OUT)
    print("Saved mapper:", OUT)

if __name__ == "__main__":
    main()
