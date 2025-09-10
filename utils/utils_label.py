
import numpy as np
from scipy import ndimage
from skimage.morphology import binary_dilation
from skimage.filters import threshold_otsu, gaussian
from skimage.measure import find_contours

def find_coutours():

    # dilate to avoid loosing small elements on edges
    image_dil = binary_dilation(image, np.ones((5, 5)))
    # make the image smooth
    image_gauss = gaussian(image_dil, sigma=30)
    # automatic threshold
    image_th = threshold_otsu(image_gauss)
    # find contour
    contour = find_contours(image_gauss > image_th, level=0.5)

def threshold_image(volume, threshold=None):

    if threshold is None:
        print("No threshold provided, computing otsu threshold...")
        threshold = threshold_otsu(volume)  # if threshold not provided, use otsu threshold
    print(f"Using threshold: {threshold}")

    # Step 1: Threshold to detect object
    mask = volume > threshold

    return mask

def thresholded_floodfill(volume, threshold=None):
    """
    Create a segmentation mask with exactly one connected bone component
    using outside flood fill.

    Parameters
    ----------
    volume : np.ndarray
        3D CT or micro-CT image (intensity values).
    threshold :
        Intensity threshold to separate object from background with large intensity difference.

    Returns
    -------
    mask : np.ndarray (bool)
        3D mask where object = True, background = False.
    """

    if threshold is None:
        print("No threshold provided, computing otsu threshold...")
        threshold = threshold_otsu(volume)  # if threshold not provided, use otsu threshold
    print(f"Using threshold: {threshold} for flood filling.")

    # Step 1: Threshold to detect object
    obj_mask = volume > threshold

    # Optional: Morphological opening to remove small objects/noise
    # obj_mask = ndimage.binary_opening(obj_mask, structure=np.ones((3, 3, 3)))

    # Step 2: Invert -> background = 1, obj = 0
    background_mask = ~obj_mask

    # Step 3: Flood fill from outside
    # Create a seed mask along the borders of the volume
    seeds = np.zeros_like(background_mask, dtype=bool)
    seeds[0 ,: ,:]   = True
    seeds[-1 ,: ,:]  = True
    seeds[: ,0 ,:]   = True
    seeds[: ,-1 ,:]  = True
    seeds[: ,: ,0]   = True
    seeds[: ,: ,-1]  = True

    # Restrict flood fill to air voxels
    background_exterior = ndimage.binary_propagation(seeds, mask=background_mask, structure=np.ones((5, 5, 5)))

    # Step 4: Everything not reached by flood fill must be object
    object_filled = ~background_exterior

    return object_filled


if __name__ == "__main__":

    path = "../../Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/Vertebrae_A/Vertebrae_A_80kV_cropped.tif"

    from utils.utils_image import load_image
    from utils.utils_plot import viz_orthogonal_slices
    from utils.utils_preprocess import rescale

    image = load_image(path, dtype=np.float32)
    image = rescale(image)  # rescale to [0, 1]
    print("Image shape:", image.shape)
    viz_orthogonal_slices(image, slice_indices=[min(image.shape)//2, min(image.shape)//2, min(image.shape)//2], title="Input Image", savefig=False)

    mask = threshold_image(image, threshold=None)

    #mask = thresholded_floodfill(image, threshold=0.22)
    viz_orthogonal_slices(mask, slice_indices=[min(mask.shape)//2, min(mask.shape)//2, min(mask.shape)//2], title="Bone Mask", savefig=False)


    print("Done")