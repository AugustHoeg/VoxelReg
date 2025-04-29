import os
import argparse

from utils.utils_plot import viz_slices, viz_multiple_images
from utils.utils_preprocess import crop_to_roi, preprocess

project_path = "C:/Users/aulho/OneDrive - Danmarks Tekniske Universitet/Dokumenter/Github/Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/"
sample_path = project_path + "Larch_A_bin1x1/"

#scan_path = sample_path + "Larch_A_bin1x1_LFOV_80kV_7W_air_2p5s_6p6mu_bin1_recon.tiff"  # LR image is the moving image
#out_name = "Larch_A_LFOV_crop_full_height"  # Name of the output file

scan_path = sample_path + "Larch_A_bin1x1_4X_80kV_7W_air_1p5_1p67mu_bin1_pos1_recon.tif"  # HR image is the reference
out_name = "Larch_A_4x_pos1"  # Name of the output file

def parse_arguments():

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Preprocess 3D image data for registration.")
    parser.add_argument("--sample_path", type=str, required=False, help="Path to the sample directory.")
    parser.add_argument("--scan_path", type=str, required=False, help="Path to the scan file.")
    parser.add_argument("--out_name", type=str, required=False, help="Output name for the processed image.")
    parser.add_argument("--run_type", type=str, default="HOME PC", help="Run type: HOME PC or DTU HPC.")

    parser.add_argument("--min_size", type=int, nargs=3, default=(2000, 1900, 1900), help="Minimum size for cropping.")
    parser.add_argument("--max_size", type=int, nargs=3, default=(2000, 1900, 1900), help="Maximum size for cropping.")

    parser.add_argument("--margin_percent", type=float, default=0.0, help="Margin percentage for cropping.")
    parser.add_argument("--divis_factor", type=int, default=4, help="Divisibility factor for cropping.")
    parser.add_argument("--save_downscaled", default=True, help="Save downscaled image.")
    parser.add_argument("--f", type=int, default=1, help="Resolution factor.")
    parser.add_argument("--mask_threshold", default="otsu", help="Threshold for binary mask image, default 'otsu'.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_arguments()

    if args.run_type == "HOME PC":
        project_path = "C:/Users/aulho/OneDrive - Danmarks Tekniske Universitet/Dokumenter/Github/Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/"
    elif args.run_type == "DTU HPC":
        project_path = "/dtu/3d-imaging-center/projects/2025_DANFIX_163_VoDaSuRe/raw_data_extern/"

    # Assign paths
    if args.sample_path is not None:
        sample_path = os.path.join(project_path, args.sample_path)
    if args.scan_path is not None:
        scan_path = os.path.join(sample_path, args.scan_path)
    if args.out_name is not None:
        out_name = args.out_name  # os.path.join(sample_path, args.out_name)

    # Define other parameters
    min_size = tuple(args.min_size)
    max_size = tuple(args.max_size)
    f = args.f  # Resolution factor between the two images
    margin_percent = args.margin_percent  # X% size increase margin of moving image
    divis_factor = args.divis_factor  # Ensure shape is divisible by d
    mask_threshold = args.mask_threshold  # Threshold for masking
    save_downscaled = args.save_downscaled  # Save downscaled image

    image, down = preprocess(scan_path, out_name, f, margin_percent, divis_factor, min_size, max_size, save_downscaled, mask_threshold)  # Preprocess moving image

    # Visualize
    slices = [image.shape[0] - 8, image.shape[0] - 16, image.shape[0] - 24]
    viz_slices(image, slices, savefig=True, title=out_name + "_preprocessed")
    if down is not None:
        slices = [down.shape[0] - 2, down.shape[0] - 4, down.shape[0] - 6]
        viz_slices(down, slices, savefig=True, title=out_name + "down_4_preprocessed")

    #
    # # Load moving image tiff as dask array, assuming everything fits in RAM
    # moving = load_tiff(moving_path)
    # moving = crop_to_roi(moving, roi_factor=f, margin_percent=0.5, divis_factor=d,
    #                      minimum_size=(2000, 800, 800), maximum_size=(2000, 800, 800))  # Reduce size
    #
    # write_tiff(moving, sample_path + "Larch_A_LFOV_crop_full_height.tiff")
    # write_nifti(moving, sample_path + "Larch_A_LFOV_crop_full_height.nii.gz")
    # del moving
    #
    # fixed = load_tiff(fixed_path)
    # fixed = crop_to_roi(fixed, roi_factor=1, margin_percent=0, divis_factor=d,
    #                     minimum_size=(1900, 1900, 1900), maximum_size=(1900, 1900, 1900))  # Reduce size
    #
    #
    #
    # # Downsample fixed w. torch
    # #pool = torch.nn.AvgPool3d(kernel_size=4, stride=4)
    # #downscaled = pool(torch.from_numpy(fixed).unsqueeze(0).float())
    #
    # down_2 = downscale_local_mean(fixed, (2, 2, 2))
    # down_4 = downscale_local_mean(down_2, (2, 2, 2))
    #
    # write_tiff(down_4.astype(np.uint16), sample_path + "Larch_A_4x_pos1_down_4.tiff")
    # write_nifti(down_4.astype(np.uint16), sample_path + "Larch_A_4x_pos1_down_4.nii.gz")

    print("Done")
