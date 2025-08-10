import numpy as np
from pathlib import Path

IN  = "data/things-eeg2/sub-01/eeg/preprocessed_eeg_training.npy"
OUT = "data/eeg/train_features_erp.npy"

def main():
    obj = np.load(IN, allow_pickle=True).item()
    X = obj["preprocessed_eeg_data"]      # expected (16540, 4, 17, 100) = (trials, repeats, channels, time)
    X = X.mean(axis=1).astype("float32")  # average repeats -> (16540, 17, 100)
    X = X.reshape(X.shape[0], -1)         # -> (16540, 1700)
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)
    Path(Path(OUT).parent).mkdir(parents=True, exist_ok=True)
    np.save(OUT, X)
    print("Saved train ERP:", OUT, X.shape)

if __name__ == "__main__":
    main()
