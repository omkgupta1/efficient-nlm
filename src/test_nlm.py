import numpy as np
from PIL import Image

from nlm import nlm_basic


image = np.array(
    Image.open("data/original/image_01.tiff").convert("L")
)

result = nlm_basic(
    image,
    patch_size=3,
    search_size=7,
    h=100,
)

Image.fromarray(result).save(
    "results/test_basic_nlm.tiff"
)

print("Basic NLM test completed.")
