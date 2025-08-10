import json, random
from pathlib import Path
import numpy as np
from sklearn.neighbors import NearestNeighbors

EMB = "embeds/gallery.npy"
META = "embeds/meta.json"

feats = np.load(EMB).astype("float32")
meta = json.loads(Path(META).read_text())

knn = NearestNeighbors(metric="cosine")
knn.fit(feats)

q = random.randrange(feats.shape[0])
D, I = knn.kneighbors(feats[q:q+1], n_neighbors=6)

print("Query:", meta[q])
for rank, (idx, dist) in enumerate(zip(I[0], D[0]), 1):
    cos_sim = 1.0 - float(dist)
    print(f"{rank}. cos_sim={cos_sim:.3f}  -> {meta[idx]}")
