"""
Pre-compute DINOv2 embeddings for all product reference images.
Output: gallery.npz with embeddings + product_code-to-category_id mapping.
"""
import json
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


def main():
    annotations_path = "data/train/annotations.json"
    metadata_path = "data/metadata.json"
    ref_dir = Path("data")
    output_path = "gallery.npz"

    # Load mappings
    with open(annotations_path) as f:
        coco = json.load(f)
    with open(metadata_path) as f:
        meta = json.load(f)

    cat_by_name = {c["name"].upper().strip(): c["id"] for c in coco["categories"]}
    code_to_catid = {}
    for p in meta["products"]:
        name = p.get("product_name", "").upper().strip()
        code = p.get("product_code", "")
        if name in cat_by_name and p.get("has_images"):
            code_to_catid[code] = cat_by_name[name]

    print(f"Mapped {len(code_to_catid)} product codes to categories")

    # Load DINOv2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m",
                               pretrained=True, num_classes=0)
    model = model.eval().to(device)
    config = resolve_data_config(model.pretrained_cfg)
    transform = create_transform(**config)

    # Compute embeddings for each product
    all_embeddings = []
    all_catids = []
    all_codes = []

    for product_code, cat_id in sorted(code_to_catid.items()):
        prod_dir = ref_dir / product_code
        if not prod_dir.exists():
            continue

        # Use main + front views (most representative)
        views = ["main.jpg", "front.jpg"]
        prod_embeddings = []
        for view in views:
            img_path = prod_dir / view
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            inp = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model(inp).cpu().numpy()[0]
            prod_embeddings.append(emb)

        if not prod_embeddings:
            # Fallback: use any available image
            for img_path in prod_dir.glob("*.jpg"):
                img = Image.open(img_path).convert("RGB")
                inp = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model(inp).cpu().numpy()[0]
                prod_embeddings.append(emb)
                break

        if prod_embeddings:
            # Average embeddings from multiple views
            avg_emb = np.mean(prod_embeddings, axis=0)
            avg_emb = avg_emb / np.linalg.norm(avg_emb)  # L2 normalize
            all_embeddings.append(avg_emb)
            all_catids.append(cat_id)
            all_codes.append(product_code)

    embeddings = np.array(all_embeddings, dtype=np.float32)
    catids = np.array(all_catids, dtype=np.int32)

    np.savez_compressed(output_path,
                        embeddings=embeddings,
                        catids=catids)

    print(f"Gallery: {len(embeddings)} products, embed dim={embeddings.shape[1]}")
    print(f"Saved to {output_path} ({Path(output_path).stat().st_size / 1024:.0f} KB)")

    # Also save the code-to-catid mapping as JSON for reference
    with open("code_to_catid.json", "w") as f:
        json.dump(code_to_catid, f)


if __name__ == "__main__":
    main()
