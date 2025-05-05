import nibabel as nib
import numpy as np

def write_nifti(image, affine=np.eye(4), output_path="", ret=False):
    """
    Write a numpy array to a NIfTI file.
    """

    # Create a NIfTI image
    nifti_image = nib.Nifti1Image(image, affine=affine)

    # Save the NIfTI image
    nib.save(nifti_image, output_path)
    print(f"Saved nifti image to: {output_path}")

    if ret:
        return image

def load_nifti(input_path):
    """
    Read a NIfTI file and return the image data.
    """
    nifti_image = nib.load(input_path)
    image_data = nifti_image.get_fdata()
    print(f"Read nifti image from: {input_path}")
    return image_data, nifti_image, nifti_image.affine

def crop_nifti(data, affine, crop_start, crop_end):

    # Define crop indices (example)
    x0, y0, z0 = crop_start  # starting voxel indices
    x1, y1, z1 = crop_end  # ending voxel indices (exclusive)

    # Crop the image data
    cropped_data = data[x0:x1, y0:y1, z0:z1]

    # Compute new affine
    # The new origin in world coordinates = affine @ [x0, y0, z0, 1]
    new_origin = affine @ [x0, y0, z0, 1]
    new_affine = affine.copy()
    new_affine[:3, 3] = new_origin[:3]

    return cropped_data, new_affine
