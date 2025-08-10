import json
from pathlib import Path
import numpy as np
import torch
import open_clip

META = "data/things-eeg2/images/image_metadata.npy"  # has train_img_files (16540)
OUT_EMB = "embeds/train_textclip.npy"                # (16540, 512)
OUT_LBL = "embeds/train_concepts.json"               # concept per trial

def concept_from_fname(fname: str) -> str:
    base = fname.rsplit("/", 1)[-1]
    stem = base.rsplit(".", 1)[0]
    return stem.rsplit("_", 1)[0]  # drop the trailing _01b/_05s/etc

def encode_text_batched(model, tokenizer, texts, device="cpu", bs=64):
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            toks = tokenizer(texts[i:i+bs])
            feats = model.encode_text(toks.to(device))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embs.append(feats.cpu().numpy().astype("float32"))
    return np.concatenate(embs, axis=0)

def main():
    m = np.load(META, allow_pickle=True).item()
    files = list(m["train_img_files"])  # len=16540
    concepts = [concept_from_fname(f) for f in files]
    uniq = sorted(set(concepts))
    print("train trials:", len(files), "unique concepts:", len(uniq))

    device = "cpu"
    torch.set_num_threads(2)  # keep CPU usage modest
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    texts = [f"a photo of a {c.replace('_',' ')}" for c in uniq]
    feats_uniq = encode_text_batched(model, tokenizer, texts, device=device, bs=64)  # (n_uniq, 512)
    idx = {c:i for i,c in enumerate(uniq)}
    arr = np.stack([feats_uniq[idx[c]] for c in concepts]).astype("float32")  # (16540, 512)

    Path(Path(OUT_EMB).parent).mkdir(parents=True, exist_ok=True)
    np.save(OUT_EMB, arr)
    Path(OUT_LBL).write_text(json.dumps(concepts))

    print("Saved:", OUT_EMB, arr.shape)
    print("Saved:", OUT_LBL, len(concepts))

if __name__ == "__main__":
    main()
