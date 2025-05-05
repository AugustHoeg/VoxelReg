import itk
import nibabel as nib
import numpy as np

from utils.utils_nifti import load_nifti

if __name__ == "__main__":

    nifti_path = "../Vedrana_master_project/3D_datasets\datasets/HCP_1200/train/134829/T1w/T1w_acpc_dc.nii"

    data, img, affine = load_nifti(nifti_path)

    itk_image = itk.imread(nifti_path)
    origin = itk_image.GetOrigin()
    spacing = itk_image.GetSpacing()
    direction = itk_image.GetDirection()

    print("Test")