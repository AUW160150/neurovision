import numpy as np
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

Xt = np.load("data/eeg/test_features_erp17_raw.npy").astype("float32")   # (200,1700)
G  = np.load("embeds/gallery_real_pca64.npy").astype("float32")          # (200,64)

def l2(a):
    n = np.linalg.norm(a, axis=1, keepdims=True) + 1e-8
    return a / n

def eval_topk(S, k):
    I = np.argpartition(-S, k, axis=1)[:, :k]
    return np.mean([i in row for i, row in enumerate(I)])

for c in [16, 32, 48, 64]:
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    acc1s, acc5s = [], []
    for tr, te in kf.split(Xt):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pls", PLSRegression(n_components=c, scale=False, max_iter=1000))
        ])
        pipe.fit(Xt[tr], G[tr])
        Yp = l2(pipe.predict(Xt[te]).astype("float32"))
        S = Yp @ G.T
        acc1s.append(np.mean(np.argmax(S, axis=1) == np.array(te)))
        acc5s.append(eval_topk(S, 5))
    print(f"PLS c={c:2d} | top1={np.mean(acc1s):.3f}  top5={np.mean(acc5s):.3f}")
