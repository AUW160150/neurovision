# src/build_real_features.py
import os, glob, json
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, welch

EEG_NPY = "data/things-eeg2/sub-01/eeg/preprocessed_eeg_test.npy"
META_NPY = "data/things-eeg2/images/image_metadata.npy"
IMG_ROOT = "data/things-eeg2/images/test/test_images"
OUT_FEATS = "data/eeg/features_real.npy"
OUT_META  = "embeds/meta_real.json"
BANDS = [(4,8),(8,12),(12,30),(30,40)]

def main():
    # --- load EEG object ---
    obj = np.load(EEG_NPY, allow_pickle=True).item()
    X = obj["preprocessed_eeg_data"]        # (200, 80, 17, 100)
    times = obj.get("times", None)
    fs = float(round(1.0/np.mean(np.diff(times)), 3)) if times is not None else 200.0

    # average repeats -> (trials, channels, time)
    X = X.mean(axis=2).astype("float32")    # -> (200, 80, 100)

    # 1–40 Hz bandpass
    b, a = butter(4, [1/(fs/2), 40/(fs/2)], btype="band")
    X = filtfilt(b, a, X, axis=-1)

    # bandpower features per channel
    feats=[]
    for tr in range(X.shape[0]):
        trial = X[tr]                       # (80, T)
        f, Pxx = welch(trial, fs=fs, nperseg=min(128, trial.shape[-1]))
        vec=[]
        for lo,hi in BANDS:
            m = (f>=lo) & (f<=hi)
            bp = Pxx[:, m].mean(axis=1)     # mean power per channel
            vec.append(bp.astype("float32"))
        vec = np.stack(vec, axis=1).reshape(-1)  # (80*4,)
        feats.append(vec)
    feats = np.stack(feats).astype("float32")    # (200, 320)
    # z-score columns
    feats = (feats - feats.mean(0, keepdims=True)) / (feats.std(0, keepdims=True) + 1e-6)

    # --- build meta from image_metadata test list ---
    imeta = np.load(META_NPY, allow_pickle=True).item()
    files = imeta["test_img_files"]        # 200 names
    meta=[]
    for fname in files:
        hits = glob.glob(os.path.join(IMG_ROOT, "**", fname), recursive=True)
        if not hits:
            raise FileNotFoundError(f"Image not found for {fname}")
        p = hits[0]
        label = Path(p).parent.name.split("_", 1)[-1]  # e.g., 00005_banana -> banana
        meta.append({"path": p, "label": label})

    # save
    Path(OUT_FEATS).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_META).parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_FEATS, feats)
    with open(OUT_META, "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved features:", OUT_FEATS, feats.shape)
    print("Saved meta:", OUT_META, len(meta))

if __name__ == "__main__":
    main()
