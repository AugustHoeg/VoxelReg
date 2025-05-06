import os
import numpy as np
import torch
from skimage.transform import downscale_local_mean
from skimage.filters import threshold_otsu

from utils.utils_plot import viz_slices, viz_multiple_images
from utils.utils_tiff import load_tiff, write_tiff, center_crop
from utils.utils_nifti import write_nifti, compute_affine_crop, compute_affine_scale
from utils.utils_txm import load_txm, get_affine_txm


def norm(image):
    # image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image_min = np.min(image)
    image_max = np.max(image)
    image -= image_min
    image /= (image_max - image_min)


def masked_norm(image, mask):
    # Get the min and max of the masked image
    masked_image = image[mask > 0]
    masked_image_min = np.min(masked_image)
    masked_image_max = np.max(masked_image)

    # Normalize the image using the mask values
    masked_image -= masked_image_min
    masked_image /= (masked_image_max - masked_image_min)

    # Set values outside mask to zero
    image[mask == 0] = 0

    # Set values inside mask to normalized values
    image[mask > 0] = masked_image


def crop_to_roi(image, roi_factor, margin_percent=0.50, divis_factor=2, minimum_size=(2000, 2000, 2000), maximum_size=(2000, 2000, 2000)):

    # Define roi
    roi = np.array(image.shape)
    roi[1:] = ((roi[1:] // roi_factor) * (1 + margin_percent)).astype(int)  # Reduce size
    roi = (roi // divis_factor * divis_factor).astype(int)  # Ensure shape is divisible by d

    #minimum_size = [np.minimum(val1, val2) for val1, val2 in zip(roi, minimum_size)]

    if minimum_size is not None:
        roi = np.maximum(roi, minimum_size)  # Ensure minimum size
    if maximum_size is not None:
        roi = np.minimum(roi, maximum_size)  # Ensure maximum size

    # Center crop to roi
    crop_image = center_crop(image, roi)
    print("Final crop shape: ", crop_image.shape)

    return crop_image


def define_roi(input_size, reduce_factor, margin_percent=0.50, divis_factor=2, minimum_size=(2000, 2000, 2000), maximum_size=(2000, 2000, 2000)):

    # Define roi
    roi = np.array(input_size)
    roi[1:] = ((roi[1:] // reduce_factor) * (1 + margin_percent)).astype(int)  # Reduce size
    roi = (roi // divis_factor * divis_factor).astype(int)  # Ensure shape is divisible by d

    # minimum_size = [np.minimum(val1, val2) for val1, val2 in zip(roi, minimum_size)]

    if minimum_size is not None:
        roi = np.maximum(roi, minimum_size)  # Ensure minimum size
    if maximum_size is not None:
        roi = np.minimum(roi, maximum_size)  # Ensure maximum size

    return roi


def preprocess(scan_path, out_path, out_name, f, margin_percent, divis_factor, min_size, max_size, pyramid_depth=3, mask_threshold=None):

    filename, file_extension = os.path.splitext(os.path.basename(scan_path))
    if out_name is None:
        out_name = filename

    # Define output directory
    if out_path is None:
        out_path = os.path.join(os.path.dirname(scan_path), "processed")
    os.makedirs(out_path, exist_ok=True)

    if file_extension == ".tiff" or file_extension == ".tif":
        image = load_tiff(scan_path)
        affine = np.eye(4)  # Identity matrix for TIFF
    elif file_extension == ".txm":
        image, metadata = load_txm(scan_path)
        print("######### TXM metadata ########## \n", metadata)
        affine = get_affine_txm(metadata)
    else:
        assert False, "Unsupported file format."

    roi = define_roi(image.shape, f, margin_percent, divis_factor, minimum_size=min_size, maximum_size=max_size)
    image, crop_start, crop_end = center_crop(image, roi)
    print(f"crop start: {crop_start}, crop end: {crop_end}, crop shape: {image.shape}")
    affine = compute_affine_crop(affine, crop_start, crop_end)  # Compute new affine based on crop roi

    # convert to float
    image = image.astype(np.float32)

    # Create image pyramid
    pyramid = []
    pyramid.append(image)
    pyramid_affines = []
    pyramid_affines.append(affine)

    # Create pyramid images
    for depth in range(pyramid_depth - 1):
        print(f"Creating pyramid level: {depth+1}/{pyramid_depth-1}")
        pyramid.append(downscale_local_mean(pyramid[depth], (2, 2, 2)).astype(np.float32))
        pyramid_affines.append(compute_affine_scale(pyramid_affines[depth], scale=2))

    for i in range(len(pyramid)):

        if mask_threshold is not None:
            mask_image = pyramid[i]
            if mask_threshold == "otsu":
                mask_threshold = threshold_otsu(mask_image)
                print("Otsu threshold: ", mask_threshold)
            else:
                mask_threshold = float(mask_threshold)
                print("Custom threshold: ", mask_threshold)
            mask = np.zeros_like(mask_image)
            mask[mask_image > mask_threshold] = 1
            mask = mask.astype(np.uint8)
            # np.save(os.path.join(out_path, out_name + f"_scale_{2 ** i}_mask.npy"), mask)
            # write_tiff(mask, os.path.join(sample_path, filename + "_mask.tiff"))
            print("Saving mask for pyramid level: ", i)
            write_nifti(mask, pyramid_affines[i], os.path.join(out_path, out_name + f"_scale_{2 ** i}_mask.nii.gz"))

            # Normalize the image based on the mask
            masked_norm(pyramid[i], mask)  # Ensure range is between [0, 1]
        else:
            # Normalize the image
            norm(pyramid[i])

        # Save downscaled images
        # write_tiff(down, os.path.join(sample_path, filename + f"_down_{2**(i+1)}.tiff"))
        # np.save(os.path.join(out_path, out_name + f"_scale_{2**i}.npy"), pyramid[i])
        print("Saving image for pyramid level: ", i)
        write_nifti(pyramid[i], pyramid_affines[i], os.path.join(out_path, out_name + f"_scale_{2**i}.nii.gz"))  # for itk-snap

    return pyramid, pyramid_affines