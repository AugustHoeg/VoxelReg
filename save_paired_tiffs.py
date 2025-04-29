import math
import itk
import SimpleITK as sitk
import numpy as np
import dask.array as da
import dask_image.ndfilters
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom

from utils.utils_plot import viz_slices, viz_multiple_images
from utils.utils_tiff import load_tiff, write_tiff, center_crop
from utils.utils_dask import load_da_from_tiff
from utils.utils_nifti import write_nifti


def crop_to_roi(image, roi_factor, margin_percent=0.50, divis_factor=2, minimum_size=(2000, 2000, 2000), maximum_size=(2000, 2000, 2000)):

    # Define roi
    roi = np.array(image.shape)
    roi[1:] = ((roi[1:] // roi_factor) * (1 + margin_percent)).astype(int)  # Reduce size
    roi = (roi // divis_factor * divis_factor).astype(int)  # Ensure shape is divisible by d
    roi = np.maximum(roi, minimum_size)  # Ensure minimum size
    roi = np.minimum(roi, maximum_size)  # Ensure maximum size

    # Center crop to roi
    crop_image = center_crop(image, roi)

    return crop_image


def my_zoom(block, factor=2, sigma=1, truncate=4.0):

    smooth = gaussian_filter(block, sigma=sigma, truncate=truncate)  # type: ignore
    return zoom(smooth, 1 / factor, order=1)  # linear interpolation


def dask_zoom(image, factor=2, sigma=1, truncate=4.0):

    kernel_radius = 2 * int(sigma * truncate + 0.5)

    depth = {0: kernel_radius, 1: kernel_radius, 2: kernel_radius}
    boundary = {0: 'none', 1: 'none', 2: 'none'}
    data = da.overlap.overlap(image, depth, boundary)
    chunksize_ds = (258, 258, 258)  # TODO fix hardcoded rechunk size # [math.ceil(size / factor) for size in data.chunksize]
    data = data.rechunk(chunksize_ds)

    result = data.map_blocks(my_zoom, factor=factor, sigma=sigma, truncate=truncate, dtype=np.uint16, chunks=chunksize_ds)
    kernel_radius_ds = math.ceil(kernel_radius/factor)
    trimmed = da.overlap.trim_internal(result, {0: kernel_radius_ds, 1: kernel_radius_ds, 2: kernel_radius_ds})

    return trimmed.compute()


def get_moving_image(moving_path, chunks=(512,512,512), f=4, d=2, margin_LR=0.50, min_size=(2000, 800, 800), max_size=(2000, 800, 800)):

    # Load moving image tiff as dask array, assuming everything fits in RAM
    moving = load_tiff(moving_path)
    moving = crop_to_roi(moving, roi_factor=f, margin_percent=0.5, divis_factor=d, minimum_size=min_size, maximum_size=max_size)  # Reduce size

    # Create dask array after cropping, otherwise the cropping messes with the chunk sizes.
    moving = da.from_array(moving, chunks=chunks)

    if d == 1:
        return np.array(moving)

    # Smoothing and downsampling
    sigma = d / 2  # Rule of thumb for sigma
    moving = dask_zoom(moving, factor=d, sigma=sigma, truncate=4.0)

    return moving


def get_fixed_image(fixed_path, chunks=(512,512,512), f=4, d=2, min_size=(2000, 800, 800), max_size=(2000, 800, 800)):

    # Load moving image tiff as dask array, assuming everything fits in RAM
    fixed = load_tiff(fixed_path)
    tdf = f * d
    fixed = crop_to_roi(fixed, roi_factor=1, margin_percent=0, divis_factor=tdf, minimum_size=min_size, maximum_size=max_size)  # Reduce size

    # Create dask array after cropping, otherwise the cropping messes with the chunk sizes.
    fixed = da.from_array(fixed, chunks=chunks)

    if tdf == 1:
        return np.array(fixed)

    # Smoothing and downsampling
    sigma = tdf / 2  # Rule of thumb for sigma
    fixed = dask_zoom(fixed, factor=tdf, sigma=sigma, truncate=4.0)

    return fixed


if __name__ == '__main__':

    # Define paths
    base_path = "C:/Users/aulho/OneDrive - Danmarks Tekniske Universitet/Dokumenter/Github/Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/"
    sample_path = base_path + "Larch_A_bin1x1/"
    moving_path = sample_path + "Larch_A_bin1x1_LFOV_80kV_7W_air_2p5s_6p6mu_bin1_recon.tiff"  # LR image is the moving image
    fixed_path = sample_path + "Larch_A_bin1x1_4X_80kV_7W_air_1p5_1p67mu_bin1_pos1_recon.tif"  # HR image is the reference
    #fixed_path = sample_path + "Larch_A_bin1x1_4X_80kV_7W_air_1p5_1p67mu_bin1_pos2_recon.tiff" # registration works

    f = 4  # Resolution factor between the two images
    d = 2  # Downsampling factor
    tdf = f * d  # Total downsampling factor

    moving = get_moving_image(moving_path, chunks=(400, 200, 200), f=f, d=d)  # (256, 256, 256)
    fixed = get_fixed_image(fixed_path, chunks=(400, 200, 200), f=f, d=d)  # (256, 256, 256)

    # Visualize
    Dm = moving.shape[0]
    viz_slices(moving, [Dm - 50, Dm - 100, Dm - 150], title="Moving image slices")
    Df = fixed.shape[0]
    viz_slices(fixed, [Df - 50, Df - 100, Df - 150], title="Fixed image slices")

    # Save as tiff
    write_tiff(moving, sample_path + "test2/" + "Larch_A_LFOV_crop.tiff")
    write_tiff(fixed, sample_path + "test2/" + "Larch_A_4x_crop_pos1.tiff")

    write_nifti(moving, sample_path + "test2/" + "Larch_A_LFOV_crop.nii.gz")
    write_nifti(fixed, sample_path + "test2/" + "Larch_A_4x_crop_pos1.nii.gz")

    if False:

        moving_array = load_tiff(moving_path)
        fixed_array = load_tiff(fixed_path)

        #max_roi = np.array([min(dim1, dim2) for dim1, dim2 in zip(moving_array.shape, fixed_array.shape)])  # Set minimum size
        #max_roi = np.array(fixed_array.shape)  # Set minimum size
        #max_roi = (max_roi // tdf * tdf).astype(int)  # Ensure shape is divisible by tdf

        margin_LR = 0.50  # X% size increase margin of moving image
        center_roi_moving = np.array(moving_array.shape)
        center_roi_moving[1:] = ((center_roi_moving[1:] // f) * (1 + margin_LR)).astype(int)  # Reduce size
        center_roi_moving = (center_roi_moving // d * d).astype(int)  # Ensure shape is divisible by d

        center_roi_fixed = np.array(fixed_array.shape)
        center_roi_fixed = (center_roi_fixed // tdf * tdf).astype(int)  # Ensure shape is divisible by tdf

        moving_array = center_crop(moving_array, center_roi_moving)
        fixed_array = center_crop(fixed_array, center_roi_fixed)

        write_downsampled_tiff(moving_array, sample_path + "test2/" + "Larch_A_LFOV_crop.tiff", factor=d)
        write_downsampled_tiff(fixed_array, sample_path + "test2/" + "Larch_A_4x_crop_pos1.tiff", factor=tdf)


