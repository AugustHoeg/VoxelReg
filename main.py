import itk
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

from utils.utils_tiff import load_tiff, write_downsampled_tiff, center_crop


if __name__ == '__main__':

    # Define paths
    base_path = "C:/Users/aulho/OneDrive - Danmarks Tekniske Universitet/Dokumenter/Github/Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/"
    sample_path = base_path + "Larch_A_bin1x1/"
    moving_path = sample_path + "Larch_A_bin1x1_LFOV_80kV_7W_air_2p5s_6p6mu_bin1_recon.tiff"  # LR image is the moving image
    fixed_path = sample_path + "Larch_A_bin1x1_4X_80kV_7W_air_1p5_1p67mu_bin1_pos1_recon.tif"  # HR image is the reference
    #fixed_path = sample_path + "Larch_A_bin1x1_4X_80kV_7W_air_1p5_1p67mu_bin1_pos2_recon.tiff" # registration works

    resolution_factor = 4  # Resolution factor between the two images
    down_factor = 6  # Downsampling factor

    moving_array = load_tiff(moving_path)
    fixed_array = load_tiff(fixed_path)

    max_roi = np.array([min(dim1, dim2) for dim1, dim2 in zip(moving_array.shape, fixed_array.shape)])  # Set minimum size
    max_roi = (max_roi // resolution_factor * resolution_factor).astype(int)  # Ensure shape is divisible by factor

    margin_LR = 0.50  # X% size increase margin of moving image
    center_roi_moving = max_roi
    center_roi_moving[1:] = ((center_roi_moving[1:] // resolution_factor) * (1 + margin_LR)).astype(int)  # Increase size by 25%

    center_roi_fixed = max_roi

    moving_array = center_crop(moving_array, center_roi_moving)
    fixed_array = center_crop(fixed_array, center_roi_fixed)

    write_downsampled_tiff(moving_array, sample_path + "Larch_A_LFOV_crop.tiff", down_factor/resolution_factor)
    write_downsampled_tiff(fixed_array, sample_path + "Larch_A_4x_crop_pos1.tiff", down_factor)


