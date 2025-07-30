import os
import glob
import numpy as np
import nibabel as nib
from utils.utils_tiff import load_tiff
import zarr
import h5py
import dask.array as da
from monai.transforms import LoadImage


def load_image(image_path,
               dtype=np.float32,
               dataset_name='/exchange/data',
               return_metadata=False,
               as_contiguous=False,
               as_dask_array=False):
    """
    Load an image from various formats, optionally as a Dask array.
    """

    # TODO: add functionality for as_contiguous for other file formats than nifti

    image = None
    metadata = None

    # DICOM folder (no file extension)
    if '.' not in os.path.basename(image_path):
        if glob.glob(os.path.join(image_path, '*.dcm')):
            reader = LoadImage(dtype=dtype, image_only=True)
            image = reader(image_path).numpy()  # MONAI returns torch tensor
            if as_dask_array:
                image = da.from_array(image, chunks="auto")

    else:
        filename, file_extension = os.path.basename(image_path).split('.', 1)

        # NIfTI
        if file_extension in ("nii", "nii.gz"):
            nifti_data = nib.load(image_path)
            image = nifti_data.get_fdata(dtype=dtype)

            if as_contiguous:
                image = np.ascontiguousarray(image)

            if as_dask_array:
                image = da.from_array(image, chunks="auto")

            metadata = nifti_data

        # TIFF
        elif file_extension in ("tiff", "tif"):
            image = load_tiff(image_path, dtype=dtype)
            if as_dask_array:
                image = da.from_array(image, chunks="auto")

        # NPY
        elif file_extension == "npy":
            image = np.load(image_path).astype(dtype)
            if as_dask_array:
                image = da.from_array(image, chunks="auto")

        # Zarr
        elif file_extension == "zarr":
            zarr_data = zarr.open(image_path, mode='r')
            if as_dask_array:
                image = da.from_zarr(zarr_data).astype(dtype)
            else:
                image = np.array(zarr_data, dtype=dtype)

        # HDF5
        elif file_extension == "h5":
            data = h5py.File(image_path, 'r')[dataset_name]
            d, h, w = data.shape
            print(f"HDF5 shape: (D={d}, H={h}, W={w})")

            if as_dask_array:
                image = da.from_array(data, chunks=(1, h, w)).astype(dtype)
            else:
                image = np.array(data, dtype=dtype)

        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")

    if return_metadata:
        return image, metadata
    else:
        return image


def normalize(volume, global_min=0.0, global_max=1.0, dtype=np.float16):

    normalized = (volume - global_min) / (global_max - global_min)
    normalized = normalized.astype(dtype)

    return normalized

def normalize_std(img, standard_deviations=3, mode='rescale'):

    """
    Normalize image using N standard deviations and clip to range [0; 1]
    :param img:
    :param standard_deviations:
    :return:
    """

    mean = np.mean(img)
    std = np.std(img)
    vmin = mean - standard_deviations * std
    vmax = mean + standard_deviations * std
    norm_img = (img - vmin) / (vmax - vmin)
    if mode == 'clip':
        norm_img = np.clip(norm_img, 0, 1)
    elif mode == 'rescale':
        norm_img = (norm_img - np.min(norm_img)) / (np.max(norm_img) - np.min(norm_img))
    return norm_img

def normalize_std_dask(img, standard_deviations=3, mode='rescale'):
    # Compute mean and std (works lazily for dask)
    mean = img.mean()
    std = img.std()

    vmin = mean - standard_deviations * std
    vmax = mean + standard_deviations * std
    norm_img = (img - vmin) / (vmax - vmin)

    if mode == 'clip':
        norm_img = da.clip(norm_img, 0, 1)  # Use dask's clip for lazy evaluation
    elif mode == 'rescale':
        norm_img = (norm_img - norm_img.min()) / (norm_img.max() - norm_img.min())

    return norm_img


def mask_cylinder(img, cylinder_radius):

    D, H, W = img.shape

    # For every slice, any voxels outside the pixel radius will be set to 0

    slice_center = (H / 2, W / 2)

    for slice_idx in range(D):
        Y, X = np.ogrid[:H, :W]
        distance_from_center = np.sqrt((Y - slice_center[0])**2 + (X - slice_center[1])**2)
        mask = distance_from_center <= cylinder_radius

        img[slice_idx, :, :] *= mask

    return img


def create_cylinder_mask(shape, cylinder_radius, cylinder_offset):

    """

    cylinder_offset[0] is down
    cylinder_offset[1] is right

    :param shape:
    :param cylinder_radius:
    :param cylinder_offset:
    :return:
    """

    D, H, W = shape

    # For every slice, any voxels outside the pixel radius will be set to 0
    slice_center = (H / 2 + cylinder_offset[0], W / 2 + cylinder_offset[1])  # Center of slice in voxels with added offset
    mask = np.zeros((D, H, W), dtype=np.uint8)

    for slice_idx in range(D):
        Y, X = np.ogrid[:H, :W]
        distance_from_center = np.sqrt((Y - slice_center[0])**2 + (X - slice_center[1])**2)
        mask[slice_idx, :, :] = distance_from_center <= cylinder_radius

    return mask


if __name__ == "__main__":
    pass