import os
import argparse
import numpy as np
from utils.utils_tiff import write_tiff


def parse_arguments():

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Save tiff.")
    parser.add_argument("--sample_path", type=str, required=False, help="Path to the sample directory.")
    parser.add_argument("--image_path", type=str, required=False, help="Path to fixed image.")
    parser.add_argument("--out_name", type=str, required=False, help="Output name for the registered output image.")
    parser.add_argument("--run_type", type=str, default="HOME PC", help="Run type: HOME PC or DTU HPC.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_arguments()

    if args.run_type == "HOME PC":
        project_path = "C:/Users/aulho/OneDrive - Danmarks Tekniske Universitet/Dokumenter/Github/Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/"
    elif args.run_type == "DTU_HPC":
        project_path = "/dtu/3d-imaging-center/projects/2025_DANFIX_163_VoDaSuRe/raw_data_extern/"

    # Assign paths
    if args.sample_path is not None:
        sample_path = os.path.join(project_path, args.sample_path, "processed/")
    if args.image_path is not None:
        image_path = os.path.join(sample_path, args.image_path)
    if args.out_name is not None:
        out_name = args.out_name  # out_name = os.path.join(sample_path, args.out_name)

    print(f"Loading {image_path}")
    image = np.load(image_path).astype(np.float32)

    print(f"Loading {image_path}")
    write_tiff(image, os.path.join(sample_path, out_name))

