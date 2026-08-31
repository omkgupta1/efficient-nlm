import time
import numpy as np
from pathlib import Path
from PIL import Image

from nlm import nlm_efficient


INPUT_DIR = Path("data/original")
OUTPUT_DIR = Path("results/efficient")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 3
SEARCH_SIZE = 7
H = 100.0


for image_path in sorted(INPUT_DIR.glob("*.tiff")):
    print(f"\nProcessing {image_path.name}...")

    image = np.array(
        Image.open(image_path).convert("L"),
        dtype=np.uint8,
    )

    start = time.perf_counter()

    result = nlm_efficient(
        image,
        patch_size=PATCH_SIZE,
        search_size=SEARCH_SIZE,
        h=H,
    )

    runtime = time.perf_counter() - start

    output_path = OUTPUT_DIR / image_path.name
    Image.fromarray(result).save(output_path)

    print(f"Runtime: {runtime:.4f} seconds")
    print(f"Saved:   {output_path}")
