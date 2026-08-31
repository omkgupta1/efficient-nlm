import numpy as np


def _validate_parameters(image, patch_size, search_size, h):
    if image.ndim != 2:
        raise ValueError("Input image must be grayscale.")

    if patch_size % 2 == 0 or search_size % 2 == 0:
        raise ValueError("patch_size and search_size must be odd.")

    if search_size < patch_size:
        raise ValueError("search_size must be >= patch_size.")

    if h <= 0:
        raise ValueError("h must be positive.")


def nlm_basic(
    image: np.ndarray,
    patch_size: int = 3,
    search_size: int = 7,
    h: float = 100.0,
) -> np.ndarray:

    _validate_parameters(image, patch_size, search_size, h)

    image = image.astype(np.float64)

    height, width = image.shape

    pr = patch_size // 2
    sr = search_size // 2
    padding = pr + sr

    padded = np.pad(
        image,
        padding,
        mode="reflect",
    )

    output = np.zeros_like(image)

    for y in range(height):
        for x in range(width):

            py = y + padding
            px = x + padding

            reference_patch = padded[
                py - pr:py + pr + 1,
                px - pr:px + pr + 1,
            ]

            weighted_sum = 0.0
            weight_sum = 0.0

            for dy in range(-sr, sr + 1):
                for dx in range(-sr, sr + 1):

                    cy = py + dy
                    cx = px + dx

                    candidate_patch = padded[
                        cy - pr:cy + pr + 1,
                        cx - pr:cx + pr + 1,
                    ]

                    distance = np.sum(
                        (reference_patch - candidate_patch) ** 2
                    )

                    weight = np.exp(
                        -distance / (h * h)
                    )

                    weighted_sum += (
                        weight * padded[cy, cx]
                    )

                    weight_sum += weight

            output[y, x] = (
                weighted_sum / weight_sum
            )

    return np.clip(
        output,
        0,
        255,
    ).astype(np.uint8)


def _reflect_indices(indices, size):
    """
    Reflect arbitrary integer indices into [0, size-1].
    """

    if size == 1:
        return np.zeros_like(indices)

    period = 2 * size - 2

    indices = np.mod(indices, period)

    return np.where(
        indices < size,
        indices,
        period - indices,
    )


def _shift_reflect(image, dy, dx):
    """
    Return image shifted by (dy, dx) using the same
    reflection convention used by the basic implementation.

    Output[y, x] = image[y + dy, x + dx]
    with reflected boundary handling.
    """

    height, width = image.shape

    y = _reflect_indices(
        np.arange(height) + dy,
        height,
    )

    x = _reflect_indices(
        np.arange(width) + dx,
        width,
    )

    return image[np.ix_(y, x)]


def _box_sum(image, radius):
    """
    Compute the sum of every (2*radius+1) x
    (2*radius+1) neighbourhood using an integral image.

    Reflection padding is used at the boundary.
    """

    size = 2 * radius + 1

    padded = np.pad(
        image,
        radius,
        mode="reflect",
    )

    # Integral image.
    integral = np.pad(
        padded,
        ((1, 0), (1, 0)),
        mode="constant",
        constant_values=0,
    )

    integral = np.cumsum(
        np.cumsum(integral, axis=0),
        axis=1,
    )

    result = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )

    return result


def nlm_efficient(
    image: np.ndarray,
    patch_size: int = 3,
    search_size: int = 7,
    h: float = 100.0,
) -> np.ndarray:
    """
    Efficient Non-Local Means using an integral image.

    For each search displacement (dy, dx):

        1. Construct the pixel-wise squared difference between
           the reference image and the shifted candidate image.
        2. Use an integral image to obtain the SSD of every
           patch simultaneously.
        3. Convert SSD values into NLM weights.
        4. Accumulate the weighted candidate pixels.

    The result is mathematically equivalent to nlm_basic().
    """

    _validate_parameters(
        image,
        patch_size,
        search_size,
        h,
    )

    image = image.astype(np.float64)

    height, width = image.shape

    pr = patch_size // 2
    sr = search_size // 2

    # IMPORTANT:
    # Use exactly the same padding convention as nlm_basic().
    padding = pr + sr

    padded = np.pad(
        image,
        padding,
        mode="reflect",
    )

    output = np.zeros(
        (height, width),
        dtype=np.float64,
    )

    weight_sum = np.zeros(
        (height, width),
        dtype=np.float64,
    )

    h_squared = h * h

    # Coordinates corresponding to the complete region required
    # for all K x K patches centered on the output pixels.
    #
    # For every output pixel, the reference and candidate patches
    # are both read directly from the SAME padded image.
    y0 = padding - pr
    y1 = padding + height + pr

    x0 = padding - pr
    x1 = padding + width + pr

    reference_region = padded[
        y0:y1,
        x0:x1,
    ]

    for dy in range(-sr, sr + 1):
        for dx in range(-sr, sr + 1):

            # Candidate region is shifted inside the SAME padded
            # image. No second padding operation is performed.
            candidate_region = padded[
                y0 + dy:y1 + dy,
                x0 + dx:x1 + dx,
            ]

            # Pixel-wise squared differences.
            squared_difference = (
                reference_region - candidate_region
            ) ** 2

            # Integral image.
            integral = np.pad(
                squared_difference,
                ((1, 0), (1, 0)),
                mode="constant",
            )

            integral = np.cumsum(
                np.cumsum(integral, axis=0),
                axis=1,
            )

            # Sum every patch_size x patch_size neighbourhood.
            k = patch_size

            distance = (
                integral[k:, k:]
                - integral[:-k, k:]
                - integral[k:, :-k]
                + integral[:-k, :-k]
            )

            # distance has exactly H x W entries.
            #
            # Candidate centre pixels correspond to the center
            # portion of candidate_region.
            candidate_center = padded[
                padding + dy:padding + dy + height,
                padding + dx:padding + dx + width,
            ]

            # NLM weight.
            weight = np.exp(
                -distance / h_squared
            )

            output += (
                weight * candidate_center
            )

            weight_sum += weight

    result = output / np.maximum(
        weight_sum,
        1e-12,
    )

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)
