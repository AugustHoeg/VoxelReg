import os
import numpy as np
import nibabel as nib
from utils.utils_tiff import load_tiff
import zarr
import h5py
import dask.array as da

def load_image(image_path, dtype=np.float32, dataset_name='/exchange/data'):

    filename, file_extension = os.path.basename(image_path).split('.', 1)

    if file_extension == "nii" or file_extension == "nii.gz":
        image = nib.load(image_path).get_fdata().astype(dtype)

    elif file_extension == "tiff" or file_extension == "tif":
        image = load_tiff(image_path, dtype=dtype)

    elif file_extension == "npy":
        image = np.load(image_path).astype(dtype)

    elif file_extension == "zarr":
        image = zarr.open(image_path, mode='r').astype(dtype)

    elif file_extension == "h5":
        data = h5py.File(image_path, 'r')[dataset_name]
        d, h, w = data[dataset_name].shape
        print(f"HDF5 shape: (D={d}, H={h}, W={w})")
        image = da.from_array(data, chunks=(1, h, w))

    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

    return image

def normalize(volume, global_min=0.0, global_max=1.0, dtype=np.float16):

    normalized = (volume - global_min) / (global_max - global_min)
    normalized = normalized.astype(dtype)

    return normalized