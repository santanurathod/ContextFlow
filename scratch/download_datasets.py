import os, subprocess


os.makedirs("MOSTA", exist_ok=True)
# list of stages
# stages = ["E9.5", "E10.5", "E11.5", "E12.5", "E13.5", "E14.5", "E15.5", "E16.5"]
stages = ["E14.5"]
# base URL pattern
base_url = "https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/{stage}_E1S1.MOSTA.h5ad"
for stage in stages:
    url = base_url.format(stage=stage)
    out = f"MOSTA/{stage}.h5ad" 
    print(f"Downloading {url} → {out}")
    subprocess.run([
        "wget", "--progress=bar:force", "-O", out, url
    ])



# os.makedirs("Flysta3D", exist_ok=True)
# filenames = [
#     "E14-16h_a_count_normal_stereoseq.h5ad",
#     "E16-18h_a_count_normal_stereoseq.h5ad",
#     "L1_a_count_normal_stereoseq.h5ad",
#     "L2_a_count_normal_stereoseq.h5ad",
#     "L3_b_count_normal_stereoseq.h5ad",
# ]
# # base URL pattern
# base_url = "https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000060/stomics/"
# for filename in filenames:
#     url = base_url + filename
#     out = f"Flysta3D/{filename}" 
#     print(f"Downloading {url} → {out}")
#     subprocess.run(["wget", "--progress=bar:force", "-O", out, url])
