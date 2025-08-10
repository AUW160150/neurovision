import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import open_clip

META = "embeds/meta_real.json"
OUT  = "embeds/gallery_real.npy"

def main():
    meta = json.loads(Path(META).read_text())
    device = "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device
    )
    model.eval()

    feats = []
    for m in meta:
        p = m["path"]
        im = Image.open(p).convert("RGB")
        x = preprocess(im).unsqueeze(0)
        with torch.no_grad():
            f = model.encode_image(x.to(device))
            f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().numpy())

    feats = np.concatenate(feats, axis=0).astype("float32")
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT, feats)
    print(f"Saved: {OUT} {feats.shape}")

if __name__ == "__main__":
    main()
