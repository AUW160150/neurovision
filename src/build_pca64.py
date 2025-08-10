import numpy as np
from pathlib import Path
from joblib import dump
from sklearn.decomposition import PCA

TRAIN_EMB = "embeds/train_imgclip.npy"        # (Ntrain, 512) — currently your 1,000 subset
TEST_EMB  = "embeds/gallery_real.npy"         # (200, 512)
OUT_TRAIN = "embeds/train_imgclip_pca64.npy"  # (Ntrain, 64)
OUT_TEST  = "embeds/gallery_real_pca64.npy"   # (200, 64)
OUT_PCA   = "models/pca64.joblib"

def main():
    Xt = np.load(TRAIN_EMB).astype("float32")
    Xg = np.load(TEST_EMB).astype("float32")

    pca = PCA(n_components=64, random_state=0)
    Xt64 = pca.fit_transform(Xt)
    Xg64 = pca.transform(Xg)

    Path("embeds").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    np.save(OUT_TRAIN, Xt64.astype("float32"))
    np.save(OUT_TEST,  Xg64.astype("float32"))
    dump(pca, OUT_PCA)

    print("Saved:", OUT_TRAIN, Xt64.shape)
    print("Saved:", OUT_TEST,  Xg64.shape)
    print("Saved:", OUT_PCA)

if __name__ == "__main__":
    main()
