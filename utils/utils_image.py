import os
import time
import glob
import numpy as np
import nibabel as nib
#import dask.array as da
import dask_image.imread
import zarr
import h5py
import tifffile
import dask.array as da
from monai.transforms import LoadImage
import matplotlib.pyplot as plt
import ants
import SimpleITK as sitk

from utils.utils_tiff import load_tiff, bigtiff2dask
from utils.utils_txm import load_txm

def load_image(image_path,
               backend="Numpy",
               dtype=None,
               chunk_shape=(1, -1, -1),
               dataset_name='/exchange/data',
               return_metadata=False,
               as_contiguous=False,
               nifti_backend="nibabel",
               **kwargs):
    """
    Load an image from various formats, optionally as a Dask array.

    Parameters
    ----------
    image_path : str
        Path to the image file or directory (for DICOM).
    dtype : numpy dtype
        Data type of the output array.
    dataset_name : str
        Dataset name for HDF5 files.
    return_metadata : bool
        Whether to return metadata (e.g., nibabel object or ants image).
    as_contiguous : bool
        Whether to return a contiguous numpy array.
    as_dask_array : bool
        Whether to return the image as a Dask array.
    nifti_backend : str
        Which backend to use for NIfTI: "nibabel" or "antspyx".
    """

    image = None
    metadata = None

    # Zarr
    if "zarr" in image_path:
        zarr_data = zarr.open(image_path, mode='r')
        if backend == "Dask":
            image = da.from_zarr(zarr_data, chunks=chunk_shape).astype(dtype)
        elif backend == "Numpy":
            image = np.array(zarr_data, dtype=dtype)

    # DICOM folder (no file extension)
    elif '.' not in os.path.basename(image_path):
        if glob.glob(os.path.join(image_path, '*.dcm')):
            if backend == "Dask":
                print("Dask backend for DICOM not supported. Loading full image and returning dask array...")
                reader = LoadImage(dtype=dtype, image_only=True)
                image = reader(image_path).numpy()  # MONAI returns torch tensor
                image = da.from_array(image, chunks=chunk_shape)
            elif backend == "Numpy":
                reader = LoadImage(dtype=dtype, image_only=True)
                image = reader(image_path).numpy()  # MONAI returns torch tensor

    else:
        filename, file_extension = os.path.basename(image_path).split('.', 1)

        # NIfTI
        if file_extension in ("nii", "nii.gz"):
            if backend == "Dask":
                nifti_data = nib.load(image_path)
                metadata = nifti_data
                image = da.from_array(nifti_data.dataobj, chunks=chunk_shape).astype(dtype)
            elif backend == "Numpy":
                if nifti_backend == "antspyx":
                    nifti_data = ants.image_read(image_path)
                    image = nifti_data.numpy().astype(dtype)
                    metadata = nifti_data  # Keep the ANTs image as metadata
                elif nifti_backend == "nibabel":
                    nifti_data = nib.load(image_path)
                    #image = nifti_data.get_fdata(dtype=dtype)
                    image = np.asanyarray(nifti_data.dataobj).astype(dtype)
                    metadata = nifti_data
                elif nifti_backend == "simpleitk" or nifti_backend == "sitk":
                    nifti_data = sitk.ReadImage(image_path)
                    image = sitk.GetArrayFromImage(nifti_data).astype(dtype)
                    metadata = nifti_data  # Keep the SimpleITK image as metadata
                else:
                    raise ValueError("nifti_backend must be either 'nibabel' or 'antspyx'.")

            if as_contiguous:
                image = np.ascontiguousarray(image)


        # TIFF
        elif file_extension in ("tiff", "tif"):
            if backend == "Dask":
                file = tifffile.TiffFile(image_path)
                if file.is_bigtiff:
                    image = bigtiff2dask(image_path)
                else:
                    image = dask_image.imread.imread(image_path, nframes=1).astype(dtype)
                    # from dask.array.image import imread as da_imread
                    # image = da_imread(image_path)
            elif backend == "Numpy":
                image = load_tiff(image_path, dtype=dtype)

        # TXM
        elif file_extension == "txm":
            if backend == "Dask":
                print("Dask backend for .txm not supported. Loading full image and returning dask array...")
                image, metadata = load_txm(image_path, dtype=dtype)
                image = da.from_array(image, chunks=chunk_shape)
            elif backend == "Numpy":
                image, metadata = load_txm(image_path, dtype=dtype)

        # NPY
        elif file_extension == "npy":
            if backend == "Dask":
                print("Dask backend for .npy not supported. Loading full image and returning dask array...")
                image = np.load(image_path).astype(dtype)
                image = da.from_array(image, chunks=chunk_shape)
            if backend == "Numpy":
                image = np.load(image_path).astype(dtype)

        # HDF5
        elif file_extension == "h5":
            data = h5py.File(image_path, 'r')[dataset_name]
            d, h, w = data.shape
            print(f"HDF5 shape: (D={d}, H={h}, W={w})")

            if backend == "Dask":
                image = da.from_array(data, chunks=chunk_shape).astype(dtype)
            elif backend == "Numpy":
                image = np.array(data, dtype=dtype)

        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")

    if return_metadata:
        return image, metadata
    else:
        return image


def normalize(volume, global_min=0.0, global_max=1.0, dtype=np.float16):

    if global_min is None:
        global_min = volume.min()
    if global_max is None:
        global_max = volume.max()

    normalized = volume - global_min
    normalized = normalized / (global_max - global_min)
    normalized = normalized.astype(dtype)

    return normalized


def normalize_std(img, standard_deviations=3, mode='rescale', mask=None, apply_mask=True):
    """
    Normalize image using N standard deviations and clip to range [0, 1].
    Normalization is applied only to pixels where mask == 1.

    :param img: Input image (numpy array)
    :param standard_deviations: Number of std deviations from mean
    :param mode: 'clip' or 'rescale'
    :param mask: Optional binary mask (same shape as img) indicating pixels to normalize
    :param apply_mask: Whether to set values outside mask to zero
    :return: Normalized image (numpy array)
    """
    img = img.astype(np.float32, copy=False)

    # define mask
    if mask is None:
        mask_arr = np.ones(img.shape, dtype=bool)
    else:
        mask_arr = mask.astype(bool, copy=False)

    values = img[mask_arr]
    if values.size == 0:
        return img.copy()  # nothing to normalize, return a copy for safety

    mean = values.mean()
    std = values.std()
    vmin = mean - standard_deviations * std
    vmax = mean + standard_deviations * std

    # output array (same dtype, same shape, uninitialized)
    norm_img = np.empty_like(img, dtype=np.float32)

    # copy original if mask not applied, else fill zeros directly
    if apply_mask:
        norm_img.fill(0)
    else:
        norm_img[...] = img

    # normalize only masked values (in-place on a view)
    norm_vals = norm_img[mask_arr]
    np.subtract(img[mask_arr], vmin, out=norm_vals)
    np.divide(norm_vals, vmax - vmin, out=norm_vals)

    if mode == 'clip':
        np.clip(norm_vals, 0, 1, out=norm_vals)
    elif mode == 'rescale':
        min_val = norm_vals.min(initial=0)
        max_val = norm_vals.max(initial=1)
        if max_val > min_val:
            np.subtract(norm_vals, min_val, out=norm_vals)
            np.divide(norm_vals, max_val - min_val, out=norm_vals)
        else:
            norm_vals.fill(0)

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


def calc_histogram(image, data_min=None, data_max=None, num_bins=256, title="Histogram", color="darkgray", savefig=False, log_scale=False, show_plot=True):
    """
    Compute and plot the histogram of an image/volume.

    Parameters
    ----------
    image : np.ndarray
        Input 2D or 3D image (any dtype, will be flattened).
    num_bins : int
        Number of bins for the histogram (default: 256).
    title : str
        Title for the plot.
    color : str
        Color of the histogram bars.
    """
    # Flatten the image/volume
    values = np.ravel(image)

    if data_max == None:
        data_max = values.max()
    if data_min == None:
        data_min = values.min()

    # Compute histogram
    hist, bin_edges = np.histogram(values, bins=num_bins, range=(data_min, data_max))

    if show_plot:
        plot_histogram(hist, bin_edges, title=title, color=color, savefig=savefig, log_scale=log_scale)

    return hist, bin_edges

def plot_histogram(hist, bin_edges, title="Histogram", save_dir="figures", color="darkgray", savefig=True, log_scale=False, density=False, low=None, high=None):

    if density:  # Normalize histogram to probabilities (optional, looks cleaner)
        hist = hist.astype(float) / hist.sum()

    # Plot
    plt.figure(figsize=(16, 10))
    # plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align="edge", color=color, alpha=0.7)
    plt.step(bin_edges[:-1], hist, where="mid", color=color, linewidth=2, alpha=0.7, label='Histogram')
    plt.fill_between(bin_edges[:-1], hist, step="mid", alpha=0.3, color=color)

    plt.title(title, fontsize=14, weight="bold")
    plt.xlabel("Intensity", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    if low is not None:
        plt.axvline(x=low, color='red', linestyle='--', label='Low Threshold')
    if high is not None:
        plt.axvline(x=high, color='green', linestyle='--', label='High Threshold')
    if log_scale:
        plt.yscale('log')
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.legend(fontsize=14)

    if savefig:
        plt.savefig(os.path.join(save_dir, f"{title}.pdf"), dpi=300, bbox_inches='tight')
    else:
        plt.show()


def compare_histograms(image1, image2,
                       data_min=0.0, data_max=1.0, num_bins=256,
                       labels=("Image 1", "Image 2"),
                       colors=("steelblue", "darkorange"),
                       title="Histogram Comparison"):
    """
    Compute and plot the histograms of two images/volumes for comparison.

    Parameters
    ----------
    image1, image2 : np.ndarray
        Input 2D or 3D images (any dtype, will be flattened).
    data_min, data_max : float
        Range for histogram computation.
    num_bins : int
        Number of bins for the histograms.
    labels : tuple
        Labels for the two histograms.
    colors : tuple
        Colors for the two histograms.
    title : str
        Title for the plot.
    """
    # Flatten
    values1 = np.ravel(image1)
    values2 = np.ravel(image2)

    # Compute histograms
    hist1, bin_edges = np.histogram(values1, bins=num_bins, range=(data_min, data_max))
    hist2, _ = np.histogram(values2, bins=num_bins, range=(data_min, data_max))

    # Normalize to probability
    hist1 = hist1.astype(float) / hist1.sum()
    hist2 = hist2.astype(float) / hist2.sum()

    # Plot
    plt.figure(figsize=(16, 10))
    plt.step(bin_edges[:-1], hist1, where="mid", color=colors[0], label=labels[0], linewidth=2)
    plt.step(bin_edges[:-1], hist2, where="mid", color=colors[1], label=labels[1], linewidth=2)
    plt.fill_between(bin_edges[:-1], hist1, step="mid", alpha=0.3, color=colors[0])
    plt.fill_between(bin_edges[:-1], hist2, step="mid", alpha=0.3, color=colors[1])

    plt.title(title, fontsize=14, weight="bold")
    plt.xlabel("Intensity", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    return (hist1, hist2), bin_edges


def match_histogram_3d(source, reference, num_bins=256):
    """
    Match the histogram of a 3D source volume to that of a reference volume.
    Both inputs are expected to be floats in [0,1].

    notes:

    z_da = da.from_array(z['HR']['0'])
    da.histogram(z_da, )

    """

    # Flatten
    src_values = np.ravel(source)
    ref_values = np.ravel(reference)

    # Histogram bins in [0,1]
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)

    # Histograms
    src_hist, _ = np.histogram(src_values, bins=bin_edges, density=True)
    ref_hist, _ = np.histogram(ref_values, bins=bin_edges, density=True)

    # CDFs
    src_cdf = np.cumsum(src_hist); src_cdf /= src_cdf[-1]
    ref_cdf = np.cumsum(ref_hist); ref_cdf /= ref_cdf[-1]

    # Mapping: for each source bin’s CDF value, find corresponding ref intensity
    mapping = np.interp(src_cdf, ref_cdf, bin_edges[:-1])

    # Digitize source values into bins
    src_bin_idx = np.digitize(src_values, bin_edges[:-1], right=True)
    src_bin_idx = np.clip(src_bin_idx, 0, num_bins-1)

    # Map using the lookup
    matched = mapping[src_bin_idx].reshape(source.shape).astype(source.dtype)

    return matched

def match_histogram_3d_continuous(source, reference, ref_sorted=None):
    """
    Histogram match a 3D source volume to a reference volume using
    continuous CDF-to-CDF mapping (no binning, smoother result).

    Both inputs are expected to be floats in [0,1].

    Parameters
    ----------
    source : np.ndarray
        3D numpy array (D, H, W), float in [0,1]
    reference : np.ndarray
        3D numpy array (D, H, W), float in [0,1]

    Returns
    -------
    matched : np.ndarray
        Histogram-matched 3D volume, same shape as source
    """

    # Sort source values
    src_values = np.ravel(source).astype(np.float32)
    src_sorted = np.sort(src_values)

    # Same for ref values
    if ref_sorted is None:
        ref_values = np.ravel(reference).astype(np.float32)
        ref_sorted = np.sort(ref_values)

    # Quantiles (uniformly spaced between 0 and 1)
    src_quantiles = np.linspace(0, 1, len(src_sorted))
    ref_quantiles = np.linspace(0, 1, len(ref_sorted))

    # Map source quantiles to reference intensities
    ref_interp = np.interp(src_quantiles, ref_quantiles, ref_sorted)

    # Build continuous mapping from source intensities to reference intensities
    matched_values = np.interp(src_values, src_sorted, ref_interp)

    # Reshape back to volume
    matched = matched_values.reshape(source.shape).astype(source.dtype)

    return matched


def match_histogram_3d_continuous_sampled(source, reference, max_sample_size=4e9):
    """
    Histogram match a 3D source volume to a reference volume using
    continuous CDF-to-CDF mapping (no binning, smoother result).

    Both inputs are expected to be floats in [0,1].

    Parameters
    ----------
    source : np.ndarray
        3D numpy array (D, H, W), float in [0,1]
    reference : np.ndarray
        3D numpy array (D, H, W), float in [0,1]
    sample_size : int
        sample size to approximate CDF, default is 16GB sample size maximum (assuming float32 precision)

    Returns
    -------
    matched : np.ndarray
        Histogram-matched 3D volume, same shape as source
    """

    # Sort source values
    if source.size > int(max_sample_size):
        src_sample = np.random.choice(source.reshape(-1), size=int(max_sample_size), replace=False)
    else:
        src_sample = source.reshape(-1)  # view

    if reference.size > int(max_sample_size):
        ref_sample = np.random.choice(reference.reshape(-1), size=int(max_sample_size), replace=False)
    else:
        ref_sample = reference.reshape(-1)  # view

    src_sorted = np.sort(src_sample)
    ref_sorted = np.sort(ref_sample)

    # Quantiles (uniformly spaced between 0 and 1)
    src_quantiles = np.linspace(0, 1, len(src_sorted))
    ref_quantiles = np.linspace(0, 1, len(ref_sorted))

    # Map source quantiles to reference intensities
    ref_interp = np.interp(src_quantiles, ref_quantiles, ref_sorted)

    # Build continuous mapping from source intensities to reference intensities
    matched_values = np.interp(source.reshape(-1), src_sorted, ref_interp)

    # Reshape back to volume
    matched = matched_values.reshape(source.shape).astype(source.dtype)

    return matched




if __name__ == "__main__":

    base_path = "../../Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/"
    image_path = os.path.join(base_path, "fixed_scale_8.nii.gz")

    start = time.time()
    image, metadata = load_image(image_path, dtype=np.float32, nifti_backend="nibabel", return_metadata=True)
    stop = time.time()
    print("Time elapsed:", stop - start)

    image = np.ascontiguousarray(image)
    matched = match_histogram_3d_continuous_sampled(image, image*2, max_sample_size=4e9)

    start = time.time()
    image, metadata = load_image(image_path, dtype=np.float32, nifti_backend="antspyx", return_metadata=True)
    stop = time.time()
    print("Time elapsed:", stop - start)

    start = time.time()
    image, metadata = load_image(image_path, dtype=np.float32, nifti_backend="sitk", return_metadata=True)
    stop = time.time()
    print("Time elapsed:", stop - start)




