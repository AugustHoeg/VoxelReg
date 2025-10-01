import tifffile
import dask.array as da
import numpy as np
import h5py
from scipy.ndimage import zoom
import dask_image.ndfilters


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
    mask = da.where(image < threshold, low, high).astype(dtype)
    return mask