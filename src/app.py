import json, random
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from joblib import load as joblib_load

EMB   = "embeds/gallery_real_pca64.npy"
META  = "embeds/meta_real.json"
EEG_X = "data/eeg/test_features_erp17_raw.npy"
MAPR  = "models/mapper.joblib"

st.set_page_config(page_title="NeuroVision — EEG→Image", page_icon="🧠", layout="wide")
st.title("NeuroVision — EEG → Image (real EEG, CPU-only)")
st.caption("Predict a CLIP embedding from EEG features, then retrieve nearest images.")

@st.cache_resource
def load_gallery():
    feats = np.load(EMB).astype("float32")           # [N,512], L2-normalized
    meta = json.loads(Path(META).read_text())        # list of {path,label}
    knn  = NearestNeighbors(metric="cosine").fit(feats)
    return feats, knn, meta

@st.cache_resource
def load_eeg_and_mapper():
    X = np.load(EEG_X).astype("float32")             # [N,F]
    mapper = joblib_load(MAPR)                       # sklearn pipeline
    return X, mapper

@st.cache_data
def load_image(p: str):
    return Image.open(p).convert("RGB")

feats, knn, meta = load_gallery()
N = len(meta)
labels = sorted({m["label"] for m in meta})
X_eeg, mapper = load_eeg_and_mapper()

with st.sidebar:
    st.subheader("Query source")
    source = st.radio("Pick source", ["Image embedding", "EEG (real)"], index=1)
    k = st.slider("Top-K", 1, 12, 6)

    if source == "Image embedding":
        mode = st.radio("Pick query", ["Random", "By index"], index=0)
        if mode == "By index":
            q_idx = st.number_input("Image index", 0, N-1, 0)
        else:
            if "q_idx" not in st.session_state:
                st.session_state.q_idx = random.randrange(N)
            if st.button("Randomize"):
                st.session_state.q_idx = random.randrange(N)
            q_idx = st.session_state.q_idx
    else:
        eeg_idx = st.number_input("EEG trial index", 0, N-1, 0)

# Build query vector
if source == "Image embedding":
    q_vec = feats[int(q_idx):int(q_idx)+1]
    q_meta = meta[int(q_idx)]
else:
    x = X_eeg[int(eeg_idx):int(eeg_idx)+1]
    pred = mapper.predict(x).astype("float32")
    pred /= (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
    q_vec = pred
    q_meta = meta[int(eeg_idx)]  # same order as features/meta

# Search
D, I = knn.kneighbors(q_vec, n_neighbors=int(k))
sims = 1.0 - D[0]

# Show query
st.markdown("### Query")
colq1, colq2 = st.columns([1,2])
with colq1:
    st.image(load_image(q_meta["path"]), caption=f"{q_meta['label']}", use_container_width=True)
with colq2:
    st.write({"source": source, "path": q_meta["path"], "label": q_meta["label"]})

# Show results
st.markdown("### Nearest images")
cols = st.columns(int(k))
for rank, (idx, sim) in enumerate(zip(I[0], sims)):
    with cols[rank]:
        m = meta[int(idx)]
        st.image(load_image(m["path"]), caption=f"{rank+1}. {m['label']} (cos={sim:.3f})", use_container_width=True)
