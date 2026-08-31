import time
import csv

import numpy as np
from pathlib import Path
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from nlm import nlm_basic, nlm_efficient


INPUT_DIR = Path("data/original")

NOISY_DIR = Path("data/noisy")
BASIC_DIR = Path("data/results/basic")
EFFICIENT_DIR = Path("data/results/efficient")

RESULTS_DIR = Path("results")

NOISY_DIR.mkdir(parents=True, exist_ok=True)
BASIC_DIR.mkdir(parents=True, exist_ok=True)
EFFICIENT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# NLM parameters
PATCH_SIZE = 5
SEARCH_SIZE = 11
H = 100.0

# Gaussian noise
NOISE_SIGMA = 20.0

# Reproducibility
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


def calculate_metrics(reference, result):
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

    for image_path in sorted(INPUT_DIR.glob("*.tiff")):

        print(f"\n{'=' * 60}")
        print(f"Processing {image_path.name}")
        print(f"{'=' * 60}")

        original = np.array(
            Image.open(image_path).convert("L"),
            dtype=np.uint8,
        )

        # --------------------------------------------------
        # Generate noisy image
        # --------------------------------------------------

        noisy = add_gaussian_noise(
            original,
            NOISE_SIGMA,
            rng,
        )

        noisy_path = NOISY_DIR / image_path.name

        Image.fromarray(noisy).save(
            noisy_path
        )

        noisy_psnr, noisy_ssim = calculate_metrics(
            original,
            noisy,
        )

        print(
            f"Noisy:     PSNR={noisy_psnr:.4f}, "
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
            h=H,
        )

        basic_runtime = time.perf_counter() - start

        basic_path = BASIC_DIR / image_path.name

        Image.fromarray(basic).save(
            basic_path
        )

        basic_psnr, basic_ssim = calculate_metrics(
            original,
            basic,
        )

        print(
            f"Basic NLM: PSNR={basic_psnr:.4f}, "
            f"SSIM={basic_ssim:.4f}, "
            f"Runtime={basic_runtime:.4f}s"
        )

        # --------------------------------------------------
        # Efficient NLM
        # --------------------------------------------------

        start = time.perf_counter()

        efficient = nlm_efficient(
            noisy,
            patch_size=PATCH_SIZE,
            search_size=SEARCH_SIZE,
            h=H,
        )

        efficient_runtime = time.perf_counter() - start

        efficient_path = EFFICIENT_DIR / image_path.name

        Image.fromarray(efficient).save(
            efficient_path
        )

        efficient_psnr, efficient_ssim = calculate_metrics(
            original,
            efficient,
        )

        print(
            f"Efficient NLM: PSNR={efficient_psnr:.4f}, "
            f"SSIM={efficient_ssim:.4f}, "
            f"Runtime={efficient_runtime:.4f}s"
        )

        speedup = basic_runtime / efficient_runtime

        print(
            f"Speedup: {speedup:.2f}x"
        )

        rows.append({
            "image": image_path.name,
            "noise_sigma": NOISE_SIGMA,
            "patch_size": PATCH_SIZE,
            "search_size": SEARCH_SIZE,
            "h": H,

            "noisy_psnr": noisy_psnr,
            "noisy_ssim": noisy_ssim,

            "basic_psnr": basic_psnr,
            "basic_ssim": basic_ssim,
            "basic_runtime": basic_runtime,

            "efficient_psnr": efficient_psnr,
            "efficient_ssim": efficient_ssim,
            "efficient_runtime": efficient_runtime,

            "speedup": speedup,
        })

    # ------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------

    csv_path = RESULTS_DIR / "experiment_results.csv"

    fieldnames = rows[0].keys()

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

    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
