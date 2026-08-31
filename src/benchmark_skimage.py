import time
import numpy as np
from pathlib import Path
from PIL import Image
from skimage.restoration import denoise_nl_means
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


ORIGINAL_DIR = Path("data/original")
NOISY_DIR = Path("data/noisy")

PATCH_SIZE = 5
PATCH_DISTANCE = 5

H_VALUES = [
    0.05,
    0.06,
    0.07,
    0.08,
    0.09,
    0.10,
]


for h in H_VALUES:

    psnr_values = []
    ssim_values = []
    runtime_values = []

    print(f"\n{'=' * 65}")
    print(f"h = {h}")
    print(f"{'=' * 65}")

    for noisy_path in sorted(NOISY_DIR.glob("*.tiff")):

        name = noisy_path.name

        noisy = np.array(
            Image.open(noisy_path).convert("L"),
            dtype=np.float64,
        ) / 255.0

        original = np.array(
            Image.open(
                ORIGINAL_DIR / name
            ).convert("L"),
            dtype=np.float64,
        ) / 255.0

        start = time.perf_counter()

        result = denoise_nl_means(
            noisy,
            patch_size=PATCH_SIZE,
            patch_distance=PATCH_DISTANCE,
            h=h,
            fast_mode=True,
            preserve_range=True,
            channel_axis=None,
        )

        runtime = time.perf_counter() - start

        psnr = peak_signal_noise_ratio(
            original,
            result,
            data_range=1.0,
        )

        ssim = structural_similarity(
            original,
            result,
            data_range=1.0,
        )

        psnr_values.append(psnr)
        ssim_values.append(ssim)
        runtime_values.append(runtime)

        print(
            f"{name}: "
            f"PSNR={psnr:.4f}, "
            f"SSIM={ssim:.4f}, "
            f"Runtime={runtime:.4f}s"
        )

    print(
        f"\nAVERAGE: "
        f"PSNR={np.mean(psnr_values):.4f}, "
        f"SSIM={np.mean(ssim_values):.4f}, "
        f"Runtime={np.mean(runtime_values):.4f}s"
    )
