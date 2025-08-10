# src/embed_gallery.py  (no faiss version)
import argparse, os, json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision import datasets
import open_clip

STL10_CLASSES = ["airplane","bird","car","cat","deer","dog","horse","monkey","ship","truck"]

def load_stl10(root):
    return datasets.STL10(root=root, split="train", download=True)

def pick_indices_by_class(ds, want_classes, per_class):
    want_ids = [STL10_CLASSES.index(c) for c in want_classes]
    buckets = {cid: [] for cid in want_ids}
    for i, y in enumerate(ds.labels):
        if y in want_ids and len(buckets[y]) < per_class:
            buckets[y].append(i)
        if all(len(v) >= per_class for v in buckets.values()):
            break
    picks = []
    for cid, idxs in buckets.items():
        picks += [(i, cid) for i in idxs]
    return picks

def save_gallery_images(ds, picks, outdir):
    paths, labels = [], []
    for i, cid in picks:
        img = ds[i][0]
        cls = STL10_CLASSES[cid]
        d = Path(outdir) / cls
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{cls}_{i}.png"
        img.save(p)
        paths.append(str(p))
        labels.append(cls)
    return paths, labels

def encode_clip(img_paths, model_name="ViT-B-32", pretrained="openai"):
    device = "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model.eval()
    feats = []
    for p in img_paths:
        im = Image.open(p).convert("RGB")
        x = preprocess(im).unsqueeze(0)
        with torch.no_grad():
            f = model.encode_image(x.to(device))
            f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().numpy())
    feats = np.concatenate(feats, axis=0).astype("float32")  # [N, 512]
    return feats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="bird,car,dog,ship")
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--gallery-dir", default="data/gallery")
    ap.add_argument("--embeds-npy", default="embeds/gallery.npy")
    ap.add_argument("--meta-json", default="embeds/meta.json")
    args = ap.parse_args()

    want_classes = [c.strip() for c in args.classes.split(",")]

    print("↓ Download/load STL10…")
    ds = load_stl10(args.root)

    print("↓ Pick subset…")
    picks = pick_indices_by_class(ds, want_classes, args.per_class)
    if not picks:
        raise SystemExit("No images picked; check class names.")
    print(f"Picked {len(picks)} images.")

    print("↓ Save gallery images…")
    img_paths, labels = save_gallery_images(ds, picks, args.gallery_dir)

    print("↓ Build CLIP embeddings (CPU)…")
    feats = encode_clip(img_paths)

    print("↓ Save artifacts…")
    os.makedirs(Path(args.embeds_npy).parent, exist_ok=True)
    np.save(args.embeds_npy, feats)
    meta = [{"path": p, "label": lbl} for p, lbl in zip(img_paths, labels)]
    Path(args.meta_json).write_text(json.dumps(meta, indent=2))

    print("Done.")
    print(f"Gallery dir: {args.gallery_dir}")
    print(f"Embeds:      {args.embeds_npy}  shape={feats.shape}")
    print(f"Meta:        {args.meta_json}")

if __name__ == "__main__":
    main()
