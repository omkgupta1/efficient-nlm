import numpy as np


def nlm_basic(
    image: np.ndarray,
    patch_size: int = 3,
    search_size: int = 7,
    h: float = 100.0,
) -> np.ndarray:
    """
    Basic Non-Local Means denoising.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image with values in [0, 255].
    patch_size : int
        Size of the square patch.
    search_size : int
        Size of the square search window.
    h : float
        Filtering parameter controlling patch similarity.

    Returns
    -------
    np.ndarray
        Denoised image with values in [0, 255].
    """

    if image.ndim != 2:
        raise ValueError("Input image must be grayscale.")

    if patch_size % 2 == 0 or search_size % 2 == 0:
        raise ValueError("patch_size and search_size must be odd.")

    if search_size < patch_size:
        raise ValueError("search_size must be >= patch_size.")

    image = image.astype(np.float64)

    height, width = image.shape

    patch_radius = patch_size // 2
    search_radius = search_size // 2

    # Replicate padding, consistent with the border-padding discussion
    # in the lecture.
    padded = np.pad(
        image,
        patch_radius + search_radius,
        mode="reflect",
    )

    output = np.zeros_like(image, dtype=np.float64)

    for y in range(height):
        for x in range(width):

            # Position of the reference pixel in the padded image.
            py = y + patch_radius + search_radius
            px = x + patch_radius + search_radius

            reference_patch = padded[
                py - patch_radius : py + patch_radius + 1,
                px - patch_radius : px + patch_radius + 1,
            ]

            weighted_sum = 0.0
            weight_sum = 0.0

            for dy in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):

                    # Candidate patch centre.
                    cy = py + dy
                    cx = px + dx

                    candidate_patch = padded[
                        cy - patch_radius : cy + patch_radius + 1,
                        cx - patch_radius : cx + patch_radius + 1,
                    ]

                    # Squared Euclidean distance between patches.
                    distance = np.sum(
                        (reference_patch - candidate_patch) ** 2
                    )

                    # NLM weight.
                    weight = np.exp(-distance / (h * h))

                    weighted_sum += weight * padded[cy, cx]
                    weight_sum += weight

            output[y, x] = weighted_sum / weight_sum

    return np.clip(output, 0, 255).astype(np.uint8)
