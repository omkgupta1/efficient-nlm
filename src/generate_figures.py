import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


ORIGINAL_DIR = Path("data/original")
NOISY_DIR = Path("data/noisy")
BASIC_DIR = Path("data/results/basic")
EFFICIENT_DIR = Path("data/results/efficient")
SKIMAGE_DIR = Path("data/results/skimage")

RESULTS_FILE = Path("results/final_results.csv")
FIGURE_DIR = Path("figures")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# Load final metrics
metrics = {}

with open(RESULTS_FILE, newline="") as f:
    for row in csv.DictReader(f):
        metrics[row["image"]] = row


for path in sorted(ORIGINAL_DIR.glob("*.tiff")):

    name = path.name
    row = metrics[name]

    original = np.array(
        Image.open(path).convert("L")
    )

    noisy = np.array(
        Image.open(NOISY_DIR / name).convert("L")
    )

    basic = np.array(
        Image.open(BASIC_DIR / name).convert("L")
    )

    efficient = np.array(
        Image.open(EFFICIENT_DIR / name).convert("L")
    )

    skimage = np.array(
        Image.open(SKIMAGE_DIR / name).convert("L")
    )

    images = [
        original,
        noisy,
        basic,
        efficient,
        skimage,
    ]

    titles = [
        "Original",
        (
            f"Noisy\n"
            f"PSNR: {float(row['noisy_psnr']):.2f} dB\n"
            f"SSIM: {float(row['noisy_ssim']):.4f}"
        ),
        (
            f"Basic NLM\n"
            f"PSNR: {float(row['basic_psnr']):.2f} dB\n"
            f"SSIM: {float(row['basic_ssim']):.4f}"
        ),
        (
            f"Our Efficient NLM\n"
            f"PSNR: {float(row['efficient_psnr']):.2f} dB\n"
            f"SSIM: {float(row['efficient_ssim']):.4f}"
        ),
        (
            f"scikit-image Fast NLM\n"
            f"PSNR: {float(row['skimage_psnr']):.2f} dB\n"
            f"SSIM: {float(row['skimage_ssim']):.4f}"
        ),
    ]

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(20, 4.8),
    )

    for ax, image, title in zip(
        axes,
        images,
        titles,
    ):
        ax.imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=255,
        )

        ax.set_title(
            title,
            fontsize=10,
        )

        ax.axis("off")

    fig.suptitle(
        f"NLM Denoising Comparison — {name}",
        fontsize=15,
    )

    fig.text(
        0.5,
        0.01,
        (
            "Gaussian noise: σ=20 | "
            "Patch: 5×5 | "
            "Search window: 11×11 | "
            "Our h=100 | "
            "scikit-image h=0.07"
        ),
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(
        rect=(0, 0.04, 1, 0.94)
    )

    output_path = (
        FIGURE_DIR /
        f"{path.stem}_comparison.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")
