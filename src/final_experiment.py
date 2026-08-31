import csv
import time

import numpy as np
from pathlib import Path
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from skimage.restoration import denoise_nl_means

from nlm import nlm_basic, nlm_efficient


ORIGINAL_DIR = Path("data/original")
NOISY_DIR = Path("data/noisy")

BASIC_DIR = Path("data/results/basic")
EFFICIENT_DIR = Path("data/results/efficient")
SKIMAGE_DIR = Path("data/results/skimage")

RESULTS_DIR = Path("results")

BASIC_DIR.mkdir(parents=True, exist_ok=True)
EFFICIENT_DIR.mkdir(parents=True, exist_ok=True)
SKIMAGE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


PATCH_SIZE = 5
SEARCH_SIZE = 11
PATCH_DISTANCE = 5

OUR_H = 100.0
SKIMAGE_H = 0.07

NOISE_SIGMA = 20.0
RANDOM_SEED = 2026


def add_gaussian_noise(image, sigma, rng):
    noise = rng.normal(
        0.0,
        sigma,
        image.shape,
    )

    noisy = image.astype(np.float64) + noise

    return np.clip(
        noisy,
        0,
        255,
    ).astype(np.uint8)


def metrics(reference, result):
    psnr = peak_signal_noise_ratio(
        reference,
        result,
        data_range=255,
    )

    ssim = structural_similarity(
        reference,
        result,
        data_range=255,
    )

    return psnr, ssim


def main():

    rng = np.random.default_rng(RANDOM_SEED)

    rows = []

    for original_path in sorted(
        ORIGINAL_DIR.glob("*.tiff")
    ):

        name = original_path.name

        print("\n" + "=" * 70)
        print(f"Processing {name}")
        print("=" * 70)

        original = np.array(
            Image.open(original_path).convert("L"),
            dtype=np.uint8,
        )

        # --------------------------------------------------
        # Generate deterministic noisy image
        # --------------------------------------------------

        noisy = add_gaussian_noise(
            original,
            NOISE_SIGMA,
            rng,
        )

        noisy_path = NOISY_DIR / name
        noisy_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        Image.fromarray(noisy).save(noisy_path)

        noisy_psnr, noisy_ssim = metrics(
            original,
            noisy,
        )

        print(
            f"Noisy: PSNR={noisy_psnr:.4f}, "
            f"SSIM={noisy_ssim:.4f}"
        )

        # --------------------------------------------------
        # Basic NLM
        # --------------------------------------------------

        start = time.perf_counter()

        basic = nlm_basic(
            noisy,
            patch_size=PATCH_SIZE,
            search_size=SEARCH_SIZE,
            h=OUR_H,
        )

        basic_time = time.perf_counter() - start

        Image.fromarray(basic).save(
            BASIC_DIR / name
        )

        basic_psnr, basic_ssim = metrics(
            original,
            basic,
        )

        print(
            f"Basic NLM: "
            f"PSNR={basic_psnr:.4f}, "
            f"SSIM={basic_ssim:.4f}, "
            f"Time={basic_time:.4f}s"
        )

        # --------------------------------------------------
        # Our Efficient NLM
        # --------------------------------------------------

        start = time.perf_counter()

        efficient = nlm_efficient(
            noisy,
            patch_size=PATCH_SIZE,
            search_size=SEARCH_SIZE,
            h=OUR_H,
        )

        efficient_time = time.perf_counter() - start

        Image.fromarray(efficient).save(
            EFFICIENT_DIR / name
        )

        efficient_psnr, efficient_ssim = metrics(
            original,
            efficient,
        )

        print(
            f"Our Efficient NLM: "
            f"PSNR={efficient_psnr:.4f}, "
            f"SSIM={efficient_ssim:.4f}, "
            f"Time={efficient_time:.4f}s"
        )

        # --------------------------------------------------
        # Existing scikit-image NLM
        # --------------------------------------------------

        noisy_float = noisy.astype(np.float64) / 255.0
        original_float = original.astype(np.float64) / 255.0

        start = time.perf_counter()

        skimage_result = denoise_nl_means(
            noisy_float,
            patch_size=PATCH_SIZE,
            patch_distance=PATCH_DISTANCE,
            h=SKIMAGE_H,
            fast_mode=True,
            preserve_range=True,
            channel_axis=None,
        )

        skimage_time = time.perf_counter() - start

        skimage_uint8 = np.clip(
            skimage_result * 255.0,
            0,
            255,
        ).astype(np.uint8)

        Image.fromarray(skimage_uint8).save(
            SKIMAGE_DIR / name
        )

        skimage_psnr = peak_signal_noise_ratio(
            original_float,
            skimage_result,
            data_range=1.0,
        )

        skimage_ssim = structural_similarity(
            original_float,
            skimage_result,
            data_range=1.0,
        )

        print(
            f"scikit-image NLM: "
            f"PSNR={skimage_psnr:.4f}, "
            f"SSIM={skimage_ssim:.4f}, "
            f"Time={skimage_time:.4f}s"
        )

        print(
            f"Our speedup vs Basic: "
            f"{basic_time / efficient_time:.2f}x"
        )

        print(
            f"scikit-image speedup vs Basic: "
            f"{basic_time / skimage_time:.2f}x"
        )

        rows.append({
            "image": name,
            "noise_sigma": NOISE_SIGMA,
            "patch_size": PATCH_SIZE,
            "search_size": SEARCH_SIZE,
            "our_h": OUR_H,
            "skimage_h": SKIMAGE_H,

            "noisy_psnr": noisy_psnr,
            "noisy_ssim": noisy_ssim,

            "basic_psnr": basic_psnr,
            "basic_ssim": basic_ssim,
            "basic_runtime": basic_time,

            "efficient_psnr": efficient_psnr,
            "efficient_ssim": efficient_ssim,
            "efficient_runtime": efficient_time,

            "skimage_psnr": skimage_psnr,
            "skimage_ssim": skimage_ssim,
            "skimage_runtime": skimage_time,

            "our_speedup": basic_time / efficient_time,
            "skimage_speedup": basic_time / skimage_time,
        })

    # ------------------------------------------------------
    # Save final CSV
    # ------------------------------------------------------

    csv_path = RESULTS_DIR / "final_results.csv"

    fieldnames = list(rows[0].keys())

    with open(
        csv_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 70)
    print(f"Final results saved to: {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
