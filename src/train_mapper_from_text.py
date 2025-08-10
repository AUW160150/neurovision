import numpy as np, json
from pathlib import Path
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import NearestNeighbors

X_TRAIN = "data/eeg/train_features_erp.npy"      # (16540, 1700)
Y_TRAIN = "embeds/train_textclip.npy"            # (16540, 512)
X_TEST  = "data/eeg/test_features_erp17.npy"     # (200, 1700)
G_TEST  = "embeds/gallery_real.npy"              # (200, 512)
OUT     = "models/mapper.joblib"

def l2(a):
    n = np.linalg.norm(a, axis=1, keepdims=True) + 1e-8
    return a / n

def main():
    X = np.load(X_TRAIN).astype("float32")
    Y = np.load(Y_TRAIN).astype("float32")
    Xtr, Xval, Ytr, Yval = train_test_split(X, Y, test_size=0.1, random_state=0, shuffle=True)

    comps_grid = [16, 32, 64, 96, 128]
    best = None
    for c in comps_grid:
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("pls", PLSRegression(n_components=c, scale=False, max_iter=1000))
        ])
        pipe.fit(Xtr, Ytr)
        Yhat = l2(pipe.predict(Xval).astype("float32"))
        Yval_n = l2(Yval.astype("float32"))
        # cosine sim against ground-truth (diagonal)
        diag = (Yhat * Yval_n).sum(axis=1).mean()
        print(f"components={c:3d}  val_cos={diag:.3f}")
        if best is None or diag > best[0]:
            best = (diag, c, pipe)

    print(f"BEST n_components={best[1]}  val_cos={best[0]:.3f}")

    # retrain on all training data with best c
    final = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("pls", PLSRegression(n_components=best[1], scale=False, max_iter=1000))
    ])
    final.fit(X, Y)

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    dump(final, OUT)
    print("Saved mapper:", OUT)

    # --- Evaluate on TEST EEG against TEST image gallery ---
    Xt = np.load(X_TEST).astype("float32")        # (200, 1700)
    G  = np.load(G_TEST).astype("float32")        # (200, 512) L2-normalized already
    Ypred = l2(final.predict(Xt).astype("float32"))
    knn = NearestNeighbors(metric="cosine").fit(G)
    D, I = knn.kneighbors(Ypred, n_neighbors=1)
    exact_at1 = (I[:,0] == np.arange(G.shape[0])).mean()
    print(f"TEST exact@1 vs gallery: {exact_at1:.3f}")

if __name__ == "__main__":
    main()
