import numpy as np
from pathlib import Path
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import NearestNeighbors

X_TRAIN = "data/eeg/train_features_erp_raw.npy"   # (16540, 1700)
Y_TRAIN = "embeds/train_textclip.npy"             # (16540, 512)
X_TEST  = "data/eeg/test_features_erp17_raw.npy"  # (200, 1700)
G_IMG   = "embeds/gallery_real.npy"               # (200, 512)  image CLIP
G_TXT   = "embeds/gallery_test_text.npy"          # (200, 512)  text CLIP

def l2(a):
    n = np.linalg.norm(a, axis=1, keepdims=True) + 1e-8
    return a / n

def main():
    X = np.load(X_TRAIN).astype("float32")
    Y = np.load(Y_TRAIN).astype("float32")
    Xtr, Xval, Ytr, Yval = train_test_split(X, Y, test_size=0.1, random_state=0, shuffle=True)

    comps_grid = [32, 64, 96, 128]
    best = None
    for c in comps_grid:
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("pls", PLSRegression(n_components=c, scale=False, max_iter=1000))
        ])
        pipe.fit(Xtr, Ytr)
        Yhat = l2(pipe.predict(Xval).astype("float32"))
        Yval_n = l2(Yval.astype("float32"))
        diag = (Yhat * Yval_n).sum(axis=1).mean()  # mean cosine with ground-truth (diagonal)
        print(f"components={c:3d}  val_cos={diag:.3f}")
        if best is None or diag > best[0]:
            best = (diag, c, pipe)

    print(f"BEST n_components={best[1]}  val_cos={best[0]:.3f}")

    # retrain on ALL train data with best components
    final = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("pls", PLSRegression(n_components=best[1], scale=False, max_iter=1000))
    ])
    final.fit(X, Y)

    Path("models").mkdir(parents=True, exist_ok=True)
    dump(final, "models/mapper.joblib")
    print("Saved mapper:", "models/mapper.joblib")

    # --- Evaluate on TEST with IMAGE gallery ---
    Xt = np.load(X_TEST).astype("float32")
    pred = l2(final.predict(Xt).astype("float32"))
    Gimg = np.load(G_IMG).astype("float32")
    I_img = np.argmax(pred @ Gimg.T, axis=1)
    exact_img = (I_img == np.arange(Gimg.shape[0])).mean()
    print(f"TEST exact@1 vs IMAGE gallery: {exact_img:.3f}")

    # --- Evaluate on TEST with TEXT gallery ---
    Gtxt = np.load(G_TXT).astype("float32")
    I_txt = np.argmax(pred @ Gtxt.T, axis=1)
    exact_txt = (I_txt == np.arange(Gtxt.shape[0])).mean()
    print(f"TEST exact@1 vs TEXT gallery:  {exact_txt:.3f}")

if __name__ == "__main__":
    main()
