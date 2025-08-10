import numpy as np
from pathlib import Path

TRAIN_EEG = "data/things-eeg2/sub-01/eeg/preprocessed_eeg_training.npy"
TEST_EEG  = "data/things-eeg2/sub-01/eeg/preprocessed_eeg_test.npy"
OUT       = "data/eeg/test_features_erp17.npy"

def main():
    tr = np.load(TRAIN_EEG, allow_pickle=True).item()
    te = np.load(TEST_EEG,  allow_pickle=True).item()

    ch_keep = list(tr["ch_names"])           # 17 train channels
    X = te["preprocessed_eeg_data"]          # expected (N, 80, 17, 100) for test
    ch_names_te = list(te["ch_names"])

    # sanity: find indices of the 17 train channels in test set
    name_to_idx = {n:i for i,n in enumerate(ch_names_te)}
    idx = [name_to_idx[c] for c in ch_keep if c in name_to_idx]
    if len(idx) != len(ch_keep):
        missing = [c for c in ch_keep if c not in name_to_idx]
        raise SystemExit(f"Missing channels in TEST: {missing}")

    # average the 17 repeats (axis=2), keep those 17 channels (axis=1)
    # X shape should be (N, C=80, R=17, T=100)
    if X.ndim != 4:
        raise SystemExit(f"Unexpected TEST shape: {X.shape}")
    X = X.mean(axis=2)              # -> (N, 80, 100)
    X = X[:, idx, :]                # -> (N, 17, 100)

    # flatten to ERP features and z-score per column
    X = X.reshape(X.shape[0], -1).astype("float32")   # (N, 1700)
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)

    Path(Path(OUT).parent).mkdir(parents=True, exist_ok=True)
    np.save(OUT, X)
    print("Saved test ERP(17ch):", OUT, X.shape)

if __name__ == "__main__":
    main()
