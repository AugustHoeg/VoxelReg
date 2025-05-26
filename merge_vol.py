import os
import itk
import nibabel as nib
import numpy as np
import argparse
from utils.utils_tiff import write_tiff
from utils.utils_image import load_image
def parse_arguments():

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Merge same-sized volumes")
    parser.add_argument("--sample_path", type=str, required=False, help="Path to the sample directory.")
    parser.add_argument("--vol1_path", type=str, required=False, help="path to vol 1")
    parser.add_argument("--vol2_path", type=str, required=False, help="path to vol 2")
    parser.add_argument("--out_path", type=str, required=False, help="Path to the output file.")
    parser.add_argument("--out_name", type=str, required=False, help="Output name for the registered output image.")
    parser.add_argument("--run_type", type=str, default="HOME PC", help="Run type: HOME PC or DTU HPC.")
    parser.add_argument("--merge_axis", type=int, required=False, help="Axis to merge along (0, 1, or 2).")

    args = parser.parse_args()
    return args

def merge_volume(vol1, vol2, merge_axis):
    """Merge two 3D volumes along the specified axis."""

    # Check that all dimensions other than merge_axis are equal
    for axis in range(vol1.ndim):
        if axis != merge_axis and vol1.shape[axis] != vol2.shape[axis]:
            raise ValueError(f"Dimension mismatch at axis {axis}: {vol1.shape[axis]} != {vol2.shape[axis]}")

    # Concatenate along the specified axis
    return np.concatenate((vol1, vol2), axis=merge_axis)


if __name__ == "__main__":

    args = parse_arguments()

    if args.run_type == "HOME PC":
        project_path = "C:/Users/aulho/OneDrive - Danmarks Tekniske Universitet/Dokumenter/Github/Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/"
    elif args.run_type == "DTU_HPC":
        project_path = "/dtu/3d-imaging-center/projects/2025_DANFIX_163_VoDaSuRe/raw_data_extern/"

    # Assign paths
    if args.sample_path is not None:
        sample_path = os.path.join(project_path, args.sample_path)
        print("Sample path: ", sample_path)
    if args.vol1_path is not None:
        vol1_path = os.path.join(sample_path, args.vol1_path)
        print("Volume 1 path: ", vol1_path)
    if args.vol2_path is not None:
        vol2_path = os.path.join(sample_path, args.vol2_path)
        print("Volume 2 path: ", vol2_path)
    if args.out_path is not None:
        out_path = os.path.join(sample_path, args.out_path)  # os.path.join(sample_path, args.out_name)
        print("Output path: ", out_path)
    if args.out_name is not None:
        out_name = args.out_name  # os.path.join(sample_path, args.out_name)
        print("Output name: ", out_name)

    print(f"Loading {vol1_path}")
    vol1 = load_image(vol1_path)
    #vol1 = np.load(vol1_path).astype(np.float32)

    print(f"Loading {vol2_path}")
    vol2 = load_image(vol2_path)
    #vol2 = np.load(vol2_path).astype(np.float32)

    merged_volume = merge_volume(vol1, vol2, args.merge_axis)

    # Create output directory if it doesn't exist
    os.makedirs(out_path, exist_ok=True)

    # Save as tiff
    print("Saving merged volume to tiff format...")
    write_tiff(merged_volume, os.path.join(out_path, f"{out_name}_merged.tiff"))



