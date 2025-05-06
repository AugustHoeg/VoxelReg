import nibabel as nib
import numpy as np

# Load the image
sample_path = "/dtu/3d-imaging-center/projects/2025_DANFIX_163_VoDaSuRe/raw_data_extern/stitched/processed/Bamboo_A_bin1x1/"
nifti_path = sample_path + "Bamboo_A_bin1x1.nii.gz"
img = nib.load(nifti_path)

# Modify the affine: keep voxel spacing and orientation, but move origin to (0, 0, 0)
new_affine = img.affine.copy()
new_affine[:3, 3] = 0  # Set the translation (origin) to (0, 0, 0)

# Create a new NIfTI image with the modified affine
new_img = nib.Nifti1Image(img.dataobj, affine=new_affine, header=img.header)

# Save the modified image
nib.save(new_img, nifti_path)
