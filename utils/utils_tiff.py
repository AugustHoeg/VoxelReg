import tifffile
import numpy as np
from scipy.ndimage import zoom
import argparse
import os
from multiprocessing.pool import ThreadPool, Pool


def crop_tiff_slice(tiff_slice, start_row, end_row, start_col, end_col):

    return tiff_slice[start_row:end_row, start_col:end_col]


def parallel_crop_tiff(tiff_path, start_row, end_row, start_col, end_col, start_depth, end_depth, n_proc=8):
    """Estimate percentiles using multiprocessing."""

    with tifffile.TiffFile(tiff_path) as tif:

        depth = len(tif.pages)
        print(f"Number of slices: {depth}")

        num_read = 0
        num_write = 0

        # Read N slices
        image_stack = ...

        # Create multiprocessing pool
        with Pool(n_proc) as pool:

            # Start N workers
            results_async = [
                pool.apply_async(crop_tiff_slice, args=(image_stack[idx], start_row, end_row, start_col, end_col))
                for idx in range(len(image_stack))
            ]

            for idx in range(start_depth, end_depth):

                cropped_slice = results_async[0].get()
                results_async.pop()

                next_image = tif.pages[idx].asarray()
                results_async.append(pool.apply_async(crop_tiff_slice, args=(next_image, start_row, end_row, start_col, end_col)))

                num_read += 1

                # Save cropped slice


def load_tiff(input_path, dtype=np.float32):
    print(f"Reading input file: {input_path}")
    image = tifffile.imread(input_path, dtype=dtype)
    print(f"tiff shape: {image.shape}")
    return image


def center_crop(image, target_shape):

    """
    Center crop a 3D image to the target shape.

    Args:
        image (ndarray): Input 3D image.
        target_shape (tuple): Target shape for cropping.

    Returns:
        ndarray: Cropped image.
    """

    if image.shape == tuple(target_shape):
        return image

    D, H, W = image.shape
    target_shape = [image.shape[i] if target_shape[i] == -1 else target_shape[i] for i in range(3)]

    center = (D // 2, H // 2, W // 2)

    crop_start = [max(0, center[i] - target_shape[i] // 2) for i in range(3)]
    crop_end = [min(image.shape[i], center[i] + target_shape[i] // 2) for i in range(3)]

    cropped_image = image[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1], crop_start[2]:crop_end[2]]
    return cropped_image, crop_start, crop_end


def write_downsampled_tiff(image, output_path, factor, ret=False):
    """
    Downsamples a 3D TIFF image by the given factor and saves the result.

    Args:
        input_path (str): Path to the input 3D TIFF file.
        output_path (str): Path to save the downsampled TIFF file.
        factor (float or tuple of 3 floats): Downsampling factor(s) for (Z, Y, X).
    """
    image_dtype = image.dtype

    if isinstance(factor, (int, float)):
        factor = (factor, factor, factor)


    ########
    if False:
        import matplotlib.pyplot as plt
        from scipy import ndimage

        D, H, W = image.shape
        c = image[:, H // 2 - 250:H // 2 + 250, W // 2 - 250:W // 2 + 250]

        plt.figure(figsize=(20, 20))
        plt.imshow(c[1976 // 2, :, :])
        plt.show()

        c_smooth = ndimage.gaussian_filter(c, sigma=3)  # type: ignore
        plt.figure(figsize=(20, 20))
        plt.imshow(c_smooth[1976 // 2, :, :])
        plt.show()

        c_ds = zoom(c, zoom=1 / np.array(factor), order=1)  # linear interpolation
        plt.figure(figsize=(20, 20))
        plt.imshow(c_ds[247 // 2, :, :])
        plt.show()

        c_smooth_ds = zoom(c_smooth, zoom=1 / np.array(factor), order=1)  # linear interpolation
        plt.figure(figsize=(20, 20))
        plt.imshow(c_smooth_ds[247 // 2, :, :])
        plt.show()
    ########

    print(f"Downsampling with factor: {factor}")
    image = zoom(image, zoom=1/np.array(factor), order=1)  # linear interpolation
    print("New shape: ", image.shape)

    print(f"Downsampled shape: {image.shape}")
    tifffile.imwrite(output_path, image.astype(image_dtype))
    print(f"Saved downsampled image to: {output_path}")

    if ret:
        return image


def write_tiff(image, output_path, ret=False):
    """
    saves a tiff from numpy array.

    Args:
        input_path (str): Path to the input 3D TIFF file.
        output_path (str): Path to save the TIFF file.
    """

    tifffile.imwrite(output_path, image.astype(image.dtype))
    print(f"Saved downsampled image to: {output_path}")

    if ret:
        return image




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample a 3D TIFF image.")
    parser.add_argument("input_path", help="Path to input TIFF stack")
    parser.add_argument("output_path", help="Path to save downsampled TIFF")
    parser.add_argument("--factor", type=float, nargs='+', default=[2.0],
                        help="Downsampling factor (single value or 3 values for Z, Y, X)")

    args = parser.parse_args()

    # Handle single factor or list of 3
    factor = args.factor
    if len(factor) == 1:
        factor = factor[0]
    elif len(factor) == 3:
        factor = tuple(factor)
    else:
        raise ValueError("Provide either one factor or three values for Z, Y, X")

    write_downsampled_tiff(args.input_path, args.output_path, factor)
