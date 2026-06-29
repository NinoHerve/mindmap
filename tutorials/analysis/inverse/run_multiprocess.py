import mindmap as mm
from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path



if __name__ == "__main__":
    # 1. Define paths and methods
    dm = mm.io.TutoManager("../../config.yml")
    sub = "simf10"
    AVE_DIR = dm.roots["mm"]/f"sub-{sub}"/"ave"
    methods = ["dspm", "sloreta", "cset", "wcset"]
    lambda2 = [0.01, 0.01, 0.1, 0.1]

    # 2. Gather all files
    all_files = [
        f for f in AVE_DIR.glob("*.fif")
        if "template" not in f.name
    ]
    total_dataset_size = len(all_files)

    # 3. Filter out what is already done
    print("Checking disk for existing results...")
    files_ave = mm.analysis.inverse.filter_remaining_files(all_files, methods)

    already_done = total_dataset_size - len(files_ave)
    print(f"Dataset Status: {already_done}/{total_dataset_size} files already completed.")
    print(f"Files left to process: {len(files_ave)}")

    if len(files_ave) == 0:
        print("All files processed! Exiting.")
        exit()

    # Generate matching covariance file paths
    files_cov = [Path(str(f).replace("ave", "cov")) for f in files_ave]

    for fcov in files_cov:
        if not fcov.exists():
            raise FileNotFoundError(f"Missing expected covariance file: {fcov}")

    # 4. Load static variables
    forward = dm.load(sub, "fwd", desc="inv")
    eig = dm.load(sub, "eig")

    # 5. Initialize process factory
    process_one_file = mm.analysis.inverse.make_process_sample(
        methods = methods,
        lambda2 = lambda2,
        forward = forward,
        eig = eig,
        crop = [-0.05, 0.13],
    )

    # 6. Launch Parallel Processing with a tqdm Progress Bar
    print("Spawning parallele workers across available cores...")

    results = Parallel(n_jobs=-1)(
        delayed(process_one_file)(fave, fcov) 
        for fave, fcov in tqdm(
            zip(files_ave, files_cov),
            total=len(files_ave),
            desc="Reconstructing Sources",
        )
    )

    print("\nProcessing cycle complete!")