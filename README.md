# neurovision
Brain decoder
NeuroVision — EEG → Image Retrieval (CPU–only)
Goal: Given real EEG from the public THINGS-EEG2 dataset, map each trial to a semantic embedding and retrieve the closest image from the held-out test gallery.
Stack: Python, NumPy, scikit-learn, Streamlit. No local GPU required. We precompute CLIP embeddings once, then run the web app on CPU.
  >> This is retrieval, not pixel-level reconstruction. We learn an EEG→CLIP projector and rank the test gallery by cosine similarity.
>  
> What this repo contains
A small Streamlit app (src/app.py) that:
loads precomputed EEG features for the 200 test trials,
loads precomputed (PCA-reduced) CLIP embeddings for the 200 test images,
applies a trained EEG→PCA(64) embedding mapper,
shows the ground-truth image and the top-K nearest matches.

> Lightweight scripts to:

fetch THINGS-EEG2 data from OSF,
build ERP features (channels × time) for train/test EEG,
embed train/test images with OpenCLIP (ViT-B/32),
fit PCA→64D on CLIP,
train simple PLS/MLP mappers on CPU.
Precomputed, small artifacts checked in for deployment:
embeds/gallery_real_pca64.npy — test gallery embeddings (200×64)
embeds/meta_real.json — paths + labels for the 200 test images
data/eeg/test_features_erp17_raw.npy — ERP features for the 200 test trials (17 channels × 100 time → 1700)
models/mapper.joblib — the trained EEG→PCA64 projector

Data sources (public, free)
Dataset: THINGS-EEG2 (OSF)
Preprocessed EEG (subject 01): component anp5v
Behavioral (not needed for filenames): component b56ha
Images + metadata: component y63gw

src/
  app.py                        # Streamlit UI
  build_train_features_erp.py   # train ERP features (17×100 → 1700)
  build_test_features_erp17.py  # test ERP features aligned to same 17 channels
  embed_train_images.py         # embed train images with OpenCLIP (CPU OK, slower)
  embed_from_meta.py            # embed the 200 test images
  build_pca64.py                # PCA(64) on train CLIP; transform test CLIP
  train_mapper_from_text_raw.py # (exploration) EEG->text CLIP path (kept for reference)
  train_mapper_pls_erp.py       # (exploration) PLS on ERP (kept for reference)
  ... (other exploration scripts kept for transparency)
data/
  things-eeg2/
    sub-01/eeg/                 # preprocessed_eeg_training.npy, preprocessed_eeg_test.npy
    images/                     # training_images/, test/test_images/, image_metadata.npy
  eeg/
    test_features_erp17_raw.npy # used by the app
embeds/
  gallery_real.npy              # 200×512 (image CLIP)  — built offline
  gallery_real_pca64.npy        # 200×64   (after PCA)  — used by the app
  meta_real.json                # [{"path": "...", "label": "..."}] in the 200-test order
models/
  mapper.joblib                 # EEG(1700) → PCA64(64) projector (PLS or MLP)
  pca64.joblib                  # (optional) PCA model if you need to re-transform server-side

We use image_metadata.npy to get ordered lists: train_img_files (16,540) and test_img_files (200).
Credits
THINGS-EEG2 dataset (OSF): authors and maintainers of the THINGS/THINGS-EEG2 projects.
OpenCLIP (ViT-B/32, “openai” weights) for image/text embeddings.
Thanks to the open-source communities behind NumPy, scikit-learn, Streamlit, and osfclient.
