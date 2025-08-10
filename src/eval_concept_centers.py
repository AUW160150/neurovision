import json
from collections import defaultdict
import numpy as np
from pathlib import Path
from joblib import load

TRAIN_IMG_EMB = "embeds/train_imgclip.npy"          # (16540, 512)
TRAIN_CONCEPTS = "embeds/train_concepts.json"       # 16540 strings (we created earlier)
PCA64 = "models/pca64.joblib"                       # fitted on full train
TEST_EEG = "data/eeg/test_features_erp17_raw.npy"   # (200, 1700)
TEST_CONCEPTS = "embeds/test_concepts.json"         # 200 strings (we created earlier)
MAPPER = "models/mapper.joblib"                     # 1700 -> 64 (PLS)

def l2(x):
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / n

def main():
    # load train embeddings + concepts
    F = np.load(TRAIN_IMG_EMB).astype("float32")         # (16540, 512)
    concepts = json.loads(Path(TRAIN_CONCEPTS).read_text())
    assert len(concepts) == F.shape[0]

    # group by concept and average to get a center in 512D (then PCA->64D)
    buckets = defaultdict(list)
    for i, c in enumerate(concepts):
        buckets[c].append(F[i])
    names = sorted(buckets.keys())
    centers_512 = np.stack([np.mean(buckets[c], axis=0) for c in names]).astype("float32")
    centers_512 = l2(centers_512)

    # PCA -> 64D and L2-normalize
    pca = load(PCA64)
    centers_64 = pca.transform(centers_512).astype("float32")
    centers_64 = l2(centers_64)

    # map TEST EEG -> 64D with our mapper
    Xt = np.load(TEST_EEG).astype("float32")
    mapper = load(MAPPER)
    Yp = mapper.predict(Xt).astype("float32")
    Yp = l2(Yp)

    # ground-truth test concepts (200)
    test_concepts = json.loads(Path(TEST_CONCEPTS).read_text())
    name_to_idx = {n:i for i,n in enumerate(names)}
    gt = np.array([name_to_idx.get(c, -1) for c in test_concepts], dtype=int)
    mask = gt >= 0
    if mask.sum() < len(gt):
        missing = [(i,c) for i,c in enumerate(test_concepts) if name_to_idx.get(c, -1) < 0]
        print(f"WARNING: {len(missing)} / {len(gt)} test concepts not in train; excluding from eval")

    # cosine scores vs concept centers
    S = Yp @ centers_64.T
    I1 = np.argmax(S, axis=1)
    acc1 = (I1[mask] == gt[mask]).mean()

    # top-5
    I5 = np.argpartition(-S, 5, axis=1)[:, :5]
    acc5 = np.mean([gt[i] in set(I5[i]) for i in range(len(gt)) if mask[i]])

    print("Concept-centroid eval")
    print(f" concepts in train: {len(names)}")
    print(f" test usable: {int(mask.sum())}/{len(gt)}")
    print(f" top1={acc1:.3f}, top5={acc5:.3f}")
    # Save artifacts for app use later
    np.save("embeds/concept_centers_pca64.npy", centers_64.astype('float32'))
    Path("embeds/concept_names.json").write_text(json.dumps(names))
    print("Saved centers:", "embeds/concept_centers_pca64.npy", centers_64.shape)
    print("Saved names:", "embeds/concept_names.json")
if __name__ == "__main__":
    main()
