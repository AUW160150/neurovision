# src/build_real_features_erp.py
import numpy as np
from pathlib import Path

EEG_NPY = "data/things-eeg2/sub-01/eeg/preprocessed_eeg_test.npy"
OUT = "data/eeg/features_real_erp.npy"

def main():
    obj = np.load(EEG_NPY, allow_pickle=True).item()
    X = obj["preprocessed_eeg_data"]          # (200, 80, 17, 100)
    X = X.mean(axis=2).astype("float32")      # average repeats -> (200, 80, 100)
    # z-score per feature (channel×time)
    X = X.reshape(X.shape[0], -1)             # (200, 8000)
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)
    Path(Path(OUT).parent).mkdir(parents=True, exist_ok=True)
    np.save(OUT, X)
    print("Saved ERP features:", OUT, X.shape)

if __name__ == "__main__":
    main()
