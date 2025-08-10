import os, glob, json, argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import open_clip

ROOT = "data/things-eeg2/images/train/training_images"
META_NPY = "data/things-eeg2/images/image_metadata.npy"
OUT_EMB = "embeds/train_imgclip.npy"
OUT_META = "embeds/train_meta.json"

def build_index(root):
    idx = {}
    for p in glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True):
        idx[os.path.basename(p)] = p
    return idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="optionally embed only the first N files")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    m = np.load(META_NPY, allow_pickle=True).item()
    files = list(m["train_img_files"])
    if args.limit:
        files = files[:args.limit]
    print(f"train files to embed: {len(files)}")

    name2path = build_index(ROOT)
    paths = []
    meta = []
    missing = 0
    for fname in files:
        p = name2path.get(os.path.basename(fname))
        if p is None:
            missing += 1
            paths.append(None)
            meta.append({"path": "", "label": "missing"})
        else:
            paths.append(p)
            label = Path(p).parent.name.split("_", 1)[-1]
            meta.append({"path": p, "label": label})
    if missing:
        print(f"WARNING: {missing} files missing from disk")

    device = "cpu"
    torch.set_num_threads(2)
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=device)
    model.eval()

    feats = []
    batch = []
    for i, p in enumerate(paths, 1):
        if not p:
            # placeholder zero vector for missing, will be L2-normed later
            feats.append(np.zeros((1,512), dtype="float32"))
            continue
        im = Image.open(p).convert("RGB")
        x = preprocess(im).unsqueeze(0)
        batch.append(x)
        if len(batch) == args.batch or i == len(paths):
            X = torch.cat(batch, dim=0)
            with torch.no_grad():
                f = model.encode_image(X.to(device))
                f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy().astype("float32"))
            batch = []
            if i % 512 == 0:
                print(f"embedded {i}/{len(paths)}")

    F = np.concatenate(feats, axis=0).astype("float32")
    Path("embeds").mkdir(parents=True, exist_ok=True)
    np.save(OUT_EMB, F)
    Path(OUT_META).write_text(json.dumps(meta, indent=2))
    print("Saved:", OUT_EMB, F.shape)
    print("Saved:", OUT_META, len(meta))

if __name__ == "__main__":
    main()
