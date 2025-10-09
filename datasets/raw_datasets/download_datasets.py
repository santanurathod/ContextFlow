import os, subprocess
import argparse


def download_GSE232025(output_dir="GSE232025"):
    """Download Axolotl Brain Regeneration Dataset (GSE232025)"""
    print("Downloading GSE232025 (Axolotl Brain Regeneration)...")
    stages = ["Stage44.h5ad", "Stage54.h5ad", "Stage57.h5ad", "Juvenile.h5ad", "Adult.h5ad"]
    base_url = "https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000056/stomics/{stage}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, stage in enumerate(stages):
        url = base_url.format(stage=stage)
        out = f"{output_dir}/d{i}_spatial_scRNAseq.h5ad"  
        print(f"Downloading {url} → {out}")
        subprocess.run([
            "wget", "--progress=bar:force", "-O", out, url
        ])


def download_GSE062025(output_dir="GSE062025"):
    """Download Mouse Organogenesis Dataset (GSE062025)"""
    print("Downloading GSE062025 (Mouse Organogenesis)...")
    stages = ["E9.5", "E10.5", "E11.5", "E12.5", "E13.5", "E14.5", "E15.5", "E16.5"]
    base_url = "https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/{stage}_E1S1.MOSTA.h5ad"
    
    os.makedirs(output_dir, exist_ok=True)
    
    for stage in stages:
        url = base_url.format(stage=stage)
        out = f"{output_dir}/{stage}.h5ad" 
        print(f"Downloading {url} → {out}")
        subprocess.run([
            "wget", "--progress=bar:force", "-O", out, url
        ])


def download_GSE092025(output_dir="GSE092025"):
    """Download Liver Regeneration Dataset (GSE092025)"""
    print("GSE092025 (Liver Regeneration):")
    print("Please download manually from: https://zenodo.org/records/6035873")
    print(f"Save the data in the {output_dir} folder")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download spatial omics datasets")
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["GSE232025", "GSE062025", "GSE092025", "all"],
        default="GSE232025",
        help="Which dataset to download by GSE number (default: GSE232025)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as dataset GSE number)"
    )
    
    args = parser.parse_args()
    
    if args.dataset == "GSE232025" or args.dataset == "all":
        out_dir = args.output_dir if args.output_dir else "GSE232025"
        download_GSE232025(out_dir)
    
    if args.dataset == "GSE062025" or args.dataset == "all":
        out_dir = args.output_dir if args.output_dir else "GSE062025"
        download_GSE062025(out_dir)
    
    if args.dataset == "GSE092025" or args.dataset == "all":
        out_dir = args.output_dir if args.output_dir else "GSE092025"
        download_GSE092025(out_dir)