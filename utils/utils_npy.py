import numpy as np


def write_npy(image, output_path, dtype=None, ret=False):
    if dtype is not None:
        image = image.astype(dtype)

    np.save(output_path, image)
    print(f"Saved npy to: {output_path}")

    if ret:
        return image