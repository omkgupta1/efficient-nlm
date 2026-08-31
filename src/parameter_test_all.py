import time
import numpy as np
from pathlib import Path
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from nlm import nlm_efficient


ORIGINAL_DIR = Path("data/original")
NOISY_DIR = Path("data/noisy")

CONFIGS = [
    (3, 7, 100),
    (5, 11, 100),
]

print(
    f"{'Image':<15}"
    f"{'Patch':>8}"
    f"{'Search':>8}"
    f"{'PSNR':>10}"
    f"{'SSIM':>10}"
    f"{'Time(s)':>10}"
)

print("-" * 70)

for noisy_path in sorted(NOISY_DIR.glob("*.tiff")):

    name = noisy_path.name

    noisy = np.array(
        Image.open(noisy_path).convert("L"),
        dtype=np.uint8,
    )

    original = np.array(
        Image.open(ORIGINAL_DIR / name).convert("L"),
        dtype=np.uint8,
    )

    for patch_size, search_size, h in CONFIGS:

        start = time.perf_counter()

        result = nlm_efficient(
            noisy,
            patch_size=patch_size,
            search_size=search_size,
            h=h,
        )

        runtime = time.perf_counter() - start

        psnr = peak_signal_noise_ratio(
            original,
            result,
            data_range=255,
        )

        ssim = structural_similarity(
            original,
            result,
            data_range=255,
        )

        print(
            f"{name:<15}"
            f"{patch_size:>8}"
            f"{search_size:>8}"
            f"{psnr:>10.4f}"
            f"{ssim:>10.4f}"
            f"{runtime:>10.4f}"
        )
