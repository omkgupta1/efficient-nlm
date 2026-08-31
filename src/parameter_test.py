import time
import numpy as np
from pathlib import Path
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from nlm import nlm_efficient


IMAGE_PATH = Path("data/noisy/image_01.tiff")

image = np.array(
    Image.open(IMAGE_PATH).convert("L"),
    dtype=np.uint8,
)

original = np.array(
    Image.open("data/original/image_01.tiff").convert("L"),
    dtype=np.uint8,
)

configs = [
    (3, 7, 100),
    (3, 11, 100),
    (3, 21, 100),
    (5, 11, 100),
    (5, 21, 100),
    (7, 21, 100),
]

print(
    f"{'Patch':>8} {'Search':>8} {'h':>8} "
    f"{'PSNR':>10} {'SSIM':>10} {'Time(s)':>10}"
)

print("-" * 60)

for patch_size, search_size, h in configs:

    start = time.perf_counter()

    result = nlm_efficient(
        image,
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
        f"{patch_size:>8} "
        f"{search_size:>8} "
        f"{h:>8} "
        f"{psnr:>10.4f} "
        f"{ssim:>10.4f} "
        f"{runtime:>10.4f}"
    )
