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