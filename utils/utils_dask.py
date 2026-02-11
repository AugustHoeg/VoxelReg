import tifffile
import dask.array as da
import numpy as np
import h5py
from scipy.ndimage import zoom
import dask_image.ndfilters
from dask.diagnostics import ProgressBar


def load_da_from_tiff(input_path, chunks=(512, 512, 512), dtype=np.float32):
    """
    Very simple function to load a tiff file into a dask array. Assumes that the entire tiff fits into memory.
    :param input_path:
    :param chunks:
    :param dtype:
    :return:
    """

    print(f"Reading input file: {input_path}")
    image = tifffile.imread(input_path, dtype=dtype)
    print(f"tiff shape: {image.shape}")

    dask_image = da.from_array(image, chunks=chunks)

    return dask_image


def downsample_dask(image, sigma_list, factor, output_path=None):

    assert isinstance(image, da.Array), "Input must be a dask array"

    block_size = image.chunksize
    block_size_ds = tuple(int(block / factor) for block in block_size)

    image = dask_image.ndfilters.gaussian_filter(image, sigma=[2, 2, 2])

    image_ds = da.map_blocks(lambda x: zoom(x,1/factor), image, dtype=np.uint16, chunks=(block_size_ds,block_size_ds, block_size_ds))

    return image_ds

def threshold_dask(image, threshold, high=1, low=0, dtype=np.uint8):
    mask = da.where(image > threshold, high, low).astype(dtype)
    return mask

def otsu_threshold_dask(arr, bins=65535, value_range=(0, 65535), remove_zero_bin=True, show_progress=True,
):
    """
    Compute Otsu threshold for a Dask array.

    Parameters
    ----------
    arr : dask.array.Array
        Input image/volume.
    bins : int
        Number of histogram bins.
    value_range : tuple
        (min, max) range for histogram.
    remove_zero_bin : bool
        Whether to drop the zero-intensity bin.
    show_progress : bool
        Whether to show Dask progress bar.

    Returns
    -------
    threshold : float
        Otsu threshold value in the same scale as arr.
    """

    # Dask histogram
    hist, bin_edges = da.histogram(arr, bins=bins, range=value_range)

    if show_progress:
        with ProgressBar(dt=1):
            hist = hist.compute()
    else:
        hist = hist.compute()

    # Convert to NumPy and optionally remove zero bin
    hist = hist.astype(np.float64)
    if remove_zero_bin:
        hist = hist[1:]
        bin_edges = bin_edges[1:]

    # Bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Probabilities
    prob = hist / hist.sum()

    # Cumulative sums
    omega = np.cumsum(prob)                      # class probabilities
    mu = np.cumsum(prob * bin_centers)           # class means
    mu_t = mu[-1]                                 # total mean

    # Between-class variance
    eps = 1e-12
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + eps)

    # Otsu threshold
    idx = np.nanargmax(sigma_b2)
    threshold = bin_centers[idx]

    return threshold


def crop_pad_vol(
    volume,
    d_range,
    h_range,
    w_range,
    pad_value=0,
):
    """
    Crop and/or pad a 3D volume using explicit start/stop indices.

    Parameters
    ----------
    volume : np.ndarray or da.Array
        Input array of shape (D, H, W)
    d_range : (int, int)
    h_range : (int, int)
    w_range : (int, int)
        Desired output region in index space.
    pad_value : scalar
        Value used for padding outside bounds.

    Returns
    -------
    Cropped/padded array of shape:
        (z_stop - z_start, y_stop - y_start, x_stop - x_start)
    """

    is_dask = isinstance(volume, da.Array)

    D, H, W = volume.shape

    z0, z1 = d_range
    y0, y1 = h_range
    x0, x1 = w_range

    # --- Compute intersection with valid region ---
    z0_valid = max(z0, 0)
    z1_valid = min(z1, D)

    y0_valid = max(y0, 0)
    y1_valid = min(y1, H)

    x0_valid = max(x0, 0)
    x1_valid = min(x1, W)

    # --- Crop valid region ---
    cropped = volume[
        z0_valid:z1_valid,
        y0_valid:y1_valid,
        x0_valid:x1_valid,
    ]

    # --- Compute padding widths ---
    pad_before = (
        z0_valid - z0,
        y0_valid - y0,
        x0_valid - x0,
    )

    pad_after = (
        z1 - z1_valid,
        y1 - y1_valid,
        x1 - x1_valid,
    )

    pad_width = tuple(
        (before, after)
        for before, after in zip(pad_before, pad_after)
    )

    # --- Apply padding if needed ---
    if any(b > 0 or a > 0 for b, a in pad_width):
        if is_dask:
            cropped = da.pad(
                cropped,
                pad_width=pad_width,
                mode="constant",
                constant_values=pad_value,
            )
        else:
            cropped = np.pad(
                cropped,
                pad_width=pad_width,
                mode="constant",
                constant_values=pad_value,
            )

    return cropped