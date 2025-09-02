import os
import numpy as np
import torch
from skimage.transform import downscale_local_mean
from skimage.filters import threshold_otsu

from utils.utils_plot import viz_slices, viz_multiple_images
from utils.utils_tiff import load_tiff, write_tiff, center_crop, top_center_crop
from utils.utils_nifti import write_nifti, get_crop_origin, set_origin, set_affine_scale, compute_affine_scale, compute_affine_crop
from utils.utils_txm import load_txm, get_affine_txm
from utils.utils_image import load_image, create_cylinder_mask, normalize_std


def norm(image):
    # image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image_min = np.min(image)
    image_max = np.max(image)
    image -= image_min
    image /= (image_max - image_min)


def masked_norm(image, mask, apply_mask=True):
    # Get the min and max of the masked image
    masked_image = image[mask > 0]
    masked_image_min = np.min(masked_image)
    masked_image_max = np.max(masked_image)

    # Normalize the image using the mask values
    masked_image -= masked_image_min
    masked_image /= (masked_image_max - masked_image_min)

    # Set values outside mask to zero
    if apply_mask:
        image[mask == 0] = 0

    # Set values inside mask to normalized values
    image[mask > 0] = masked_image


def masked_norm_std(image, mask, standard_deviations=3, mode='rescale', apply_mask=True):
    # Get the min and max of the masked image
    masked_image = image[mask > 0]
    mean = masked_image.mean()
    std = masked_image.std()
    vmin = mean - standard_deviations * std
    vmax = mean + standard_deviations * std

    # Normalize the image using the mask values
    masked_image -= vmin
    masked_image /= (vmax - vmin)

    if mode == 'clip':
        np.clip(masked_image, 0, 1, out=masked_image)
    elif mode == 'rescale':
        vmin = masked_image.min(initial=0)
        vmax = masked_image.max(initial=1)
        if vmax > vmin:
            masked_image -= vmin
            masked_image /= (vmax - vmin)
        else:
            masked_image.fill(0)

    # Set values outside mask to zero
    if apply_mask:
        image[mask == 0] = 0

    # Set values inside mask to normalized values
    image[mask > 0] = masked_image


def norm_std(image, standard_deviations=3, mode='rescale'):
    # Get the min and max of the masked image
    mean = image.mean()
    std = image.std()
    vmin = mean - standard_deviations * std
    vmax = mean + standard_deviations * std

    # Normalize the image using the mask values
    image -= vmin
    image /= (vmax - vmin)

    if mode == 'clip':
        np.clip(image, 0, 1, out=image)
    elif mode == 'rescale':
        vmin = image.min(initial=0)
        vmax = image.max(initial=1)
        if vmax > vmin:
            image -= vmin
            image /= (vmax - vmin)
        else:
            image.fill(0)


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

    minimum_size = [input_size[i] if minimum_size[i] == -1 else minimum_size[i] for i in range(len(minimum_size))]
    maximum_size = [input_size[i] if maximum_size[i] == -1 else maximum_size[i] for i in range(len(maximum_size))]

    if minimum_size is not None:
        roi = np.maximum(roi, minimum_size)  # Ensure minimum size
    if maximum_size is not None:
        roi = np.minimum(roi, maximum_size)  # Ensure maximum size

    return roi


def preprocess(scan_path, out_path, out_name, f, margin_percent, divis_factor, min_size, max_size, pyramid_depth=3, mask_threshold=None):

    filename, file_extension = os.path.basename(scan_path).split('.', 1)
    if out_name is None:
        out_name = filename

    # Define output directory
    if out_path is None:
        out_path = os.path.join(os.path.dirname(scan_path), "processed")
    os.makedirs(out_path, exist_ok=True)

    if file_extension == ".tiff" or file_extension == ".tif":
        image = load_tiff(scan_path)
        nifti_affine = np.eye(4)  # Identity matrix for TIFF
    elif file_extension == ".txm":
        image, metadata = load_txm(scan_path)
        print("######### TXM metadata ########## \n", metadata)
        nifti_affine = get_affine_txm(metadata, custom_origin=(0, 0, 0))
    else:
        assert False, "Unsupported file format."

    roi = define_roi(image.shape, f, margin_percent, divis_factor, minimum_size=min_size, maximum_size=max_size)
    image, crop_start, crop_end = center_crop(image, roi)
    print(f"crop start: {crop_start}, crop end: {crop_end}, crop shape: {image.shape}")
    nifti_affine = compute_affine_crop(nifti_affine, crop_start)  # Compute new affine based on crop roi
    print("Nifti origin after crop: ", nifti_affine)

    # convert to float
    image = image.astype(np.float32)

    # Create image pyramid
    pyramid = []
    pyramid.append(image)
    pyramid_affines = []
    pyramid_affines.append(nifti_affine)

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


def get_image_and_affine(scan_path, custom_origin=(0, 0, 0), pixel_size_mm=(None, None, None), dtype=np.float32):

    #filename, file_extension = os.path.splitext(os.path.basename(scan_path))
    filename, file_extension = os.path.basename(scan_path).split('.', 1)
    print("file name: ", filename)
    print("file extension: ", file_extension)

    nifti_affine = None
    if file_extension == "tiff" or file_extension == "tif":
        # If path has glob wildcards, parse flag to load all files
        image = load_tiff(scan_path, dtype=dtype, image_sequence=True if '*' in scan_path else False)
    elif file_extension == "txm":
        image, metadata = load_txm(scan_path)
        print("######### TXM metadata ########## \n", metadata)
        nifti_affine = get_affine_txm(metadata)
    elif file_extension == "nii" or file_extension == "nii.gz":
        image, nifti_data = load_image(scan_path, dtype=dtype, return_metadata=True)
        nifti_affine = nifti_data.affine
    elif file_extension == "npy":
        image = load_image(scan_path, dtype=dtype)
    elif file_extension == "h5":
        image = load_image(scan_path, dtype=dtype, dataset_name='/exchange/data')
        print("Loading HDF5 dataset: /exchange/data")
        image = np.array(image)
    else:
        assert False, f"Unsupported file format: {file_extension}"

    if nifti_affine is None:
        if None in pixel_size_mm or pixel_size_mm == (None, None, None):
            nifti_affine = np.eye(4)  # Identity matrix for TIFF
        else:
            nifti_affine = np.diag([pixel_size_mm[0], pixel_size_mm[1], pixel_size_mm[2], 1])  # set pixel size in mm
    nifti_affine[:3, 3] = np.array(custom_origin)  # Set custom origin

    print("Image shape: ", image.shape)
    print("Nifti affine: \n", nifti_affine)
    return image, nifti_affine

def compute_crop_bounds(image, roi, top_index="last", slice_axis=0):
    start_crop = [0] * image.ndim
    end_crop = [0] * image.ndim

    for i in range(len(roi)):
        diff = min(0, roi[i] - image.shape[i])  # ensure diff is negative for cropping

        if i == slice_axis:  # For the slice dimension, crop from the top or bottom
            if top_index == "first":
                crop_before = 0
                crop_after = np.abs(diff)
            else:
                crop_before = np.abs(diff)
                crop_after = 0
        else:
            crop_before = np.abs(diff) // 2
            crop_after = np.abs(diff) - crop_before

        start_crop[i] = int(crop_before)  # Set the start coordinate for this dimension
        end_crop[i] = int(crop_after)

    return start_crop, end_crop


def image_crop_pad(image, roi, top_index="last", slice_axis=0):
    """
    Pads or crops an N-dimensional image to match the specified ROI size.

    Parameters:
    - image: np.ndarray, the input N-dimensional image.
    - roi: tuple or list of ints, the desired output shape for each dimension.

    Returns:
    - np.ndarray: The cropped or padded image.
    """

    start_coords = [0] * image.ndim
    end_coords = [image.shape[i] for i in range(image.ndim)]

    for i in range(len(roi)):
        diff = roi[i] - image.shape[i]

        if diff > 0:
            if i == slice_axis:
                if top_index == "first":
                    pad_before = 0
                    pad_after = diff
                else:
                    pad_before = diff
                    pad_after = 0
            else:
                pad_before = diff // 2
                pad_after = diff - pad_before

            pad_width = [(0, 0)] * image.ndim
            pad_width[i] = (pad_before, pad_after)
            image = np.pad(image, pad_width, mode='constant', constant_values=0)

            start_coords[i] = -pad_before  # Set the start coordinate for this dimension
            end_coords[i] = image.shape[i]  # Set end coords to the padded image shape

        elif diff < 0:
            if i == slice_axis:  # For the first dimension, crop from the top or bottom
                if top_index == "first":
                    crop_before = 0
                    crop_after = np.abs(diff)
                else:
                    crop_before = np.abs(diff)
                    crop_after = 0
            else:
                crop_before = np.abs(diff) // 2
                crop_after = np.abs(diff) - crop_before

            slices = [slice(None)] * image.ndim
            slices[i] = slice(crop_before, image.shape[i] - crop_after)
            image = image[tuple(slices)]

            start_coords[i] = crop_before  # Set the start coordinate for this dimension
            end_coords[i] = image.shape[i]  # Set end coords to the cropped image shape

    return image, start_coords, end_coords


def define_image_space(image, nifti_affine, f, margin_percent, divis_factor, min_size, max_size, top_index="last"):

    roi = define_roi(image.shape, f, margin_percent, divis_factor, minimum_size=min_size, maximum_size=max_size)

    if True:
        image, start_coords, end_coords = image_crop_pad(image, roi, top_index)
    else:
        image, start_coords, end_coords = top_center_crop(image, roi, top_index)
        print(f"start coords: {start_coords}, end coords: {end_coords}, new shape: {image.shape}")

    nifti_affine = compute_affine_crop(nifti_affine, start_coords)  # Compute new affine based on crop roi
    print("Nifti affine after crop/pad: \n", nifti_affine)

    return image, nifti_affine, start_coords, end_coords

def mask_with_threshold(image, mask_threshold):

    # print(f"Creating threshold mask for pyramid level: {i}")
    if mask_threshold is None:  # Use Otsu's method for thresholding if threshold not specified
        mask_threshold = threshold_otsu(image)
        print("Otsu threshold: ", mask_threshold)
    else:
        mask_threshold = float(mask_threshold)
        print("Custom threshold: ", mask_threshold)
    mask = np.zeros(image.shape, dtype=np.uint8)  # Create a mask of zeros
    mask[image > mask_threshold] = 1  # Set values above threshold to 1
    return mask

def mask_with_cylinder(image, cylinder_radius, cylinder_offset):

    # print(f"Creating cylinder mask for pyramid level: {i}")
    if cylinder_radius is None:
        raise ValueError("Pixel radius must be specified for cylindrical mask method.")
    # offset = (cylinder_offset[0] / 2 ** i, cylinder_offset[1] / 2 ** i)
    # mask = create_cylinder_mask(image.shape, cylinder_radius / 2 ** i, offset)  # Example radius
    mask = create_cylinder_mask(image.shape, cylinder_radius, cylinder_offset)  # Example radius
    return mask

def get_image_pyramid(image, nifti_affine, pyramid_depth=3, mask_method='threshold', mask_threshold=None, cylinder_radius=None, cylinder_offset=(0, 0), apply_mask=False):

    # convert to float
    image = image.astype(np.float32)

    if mask_method == 'threshold':
        mask = mask_with_threshold(image, mask_threshold)
    elif mask_method == 'cylinder':
        mask = mask_with_cylinder(image, cylinder_radius, cylinder_offset)
    else:
        mask = None

    if mask is None:
        # Normalize the image and ensure range is between [0, 1]
        norm_std(image, standard_deviations=3, mode='rescale')
    else:
        # Normalize the image using values inside mask and ensure range is between [0, 1]
        masked_norm_std(image, mask, standard_deviations=3, mode='rescale', apply_mask=apply_mask)

    # Create image/mask pyramid
    image_pyramid = []
    image_pyramid.append(image)
    mask_pyramid = []
    mask_pyramid.append(mask)

    affines = []
    affines.append(nifti_affine.copy())

    # Create pyramid images
    for depth in range(pyramid_depth - 1):
        print(f"Creating pyramid level: {depth + 1}/{pyramid_depth - 1}")
        down = downscale_local_mean(image_pyramid[depth], (2, 2, 2)).astype(np.float32)
        image_pyramid.append(down)

        affines.append(compute_affine_scale(affines[depth], scale=2))

        if mask_method == 'threshold':
            mask = mask_with_threshold(down, mask_threshold)
        elif mask_method == 'cylinder':
            offset = [(val / 2 ** (depth + 1)) for val in cylinder_offset]
            radius = [(val / 2 ** (depth + 1)) for val in cylinder_radius]
            mask = mask_with_cylinder(down, radius, offset)
        else:
            mask = None

        mask_pyramid.append(mask)

    return image_pyramid, mask_pyramid, affines


def save_image_pyramid(image_pyramid, mask_pyramid, affines, scan_path, out_path, out_name):

    filename, file_extension = os.path.basename(scan_path).split('.', 1)
    if out_name is None:
        out_name = filename

    # Define output directory
    if out_path is None:
        out_path = os.path.join(os.path.dirname(scan_path), "processed")
    os.makedirs(out_path, exist_ok=True)

    for i in range(0, len(image_pyramid)):
        if image_pyramid[i] is None:
            continue
        # Save downscaled images
        # write_tiff(down, os.path.join(sample_path, filename + f"_down_{2**(i+1)}.tiff"))
        # np.save(os.path.join(out_path, out_name + f"_scale_{2**i}.npy"), pyramid[i])
        print(f"Writing pyramid image level: {i} with shape {image_pyramid[i].shape}")
        write_nifti(image_pyramid[i], affines[i], os.path.join(out_path, out_name + f"_scale_{2 ** i}.nii.gz"))

    for i in range(0, len(mask_pyramid)):
        if mask_pyramid[i] is None:
            continue
        # np.save(os.path.join(out_path, out_name + f"_scale_{2 ** i}_mask.npy"), mask)
        # write_tiff(mask, os.path.join(sample_path, filename + "_mask.tiff"))
        print(f"Writing pyramid mask level: {i} with shape {mask_pyramid[i].shape}")
        write_nifti(mask_pyramid[i], affines[i], os.path.join(out_path, out_name + f"_scale_{2 ** i}_mask.nii.gz"))