import nibabel as nib
import numpy as np

def write_nifti(image, output_path, ret=False):
    """
    Write a numpy array to a NIfTI file.
    """

    # Create a NIfTI image
    nifti_image = nib.Nifti1Image(image, affine=np.eye(4))

    # Save the NIfTI image
    nib.save(nifti_image, output_path)
    print(f"Saved nifti image to: {output_path}")

    if ret:
        return image