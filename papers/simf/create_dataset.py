"""
Create a dataset using the simulations.

Requirements
------------
- Dataset folder with template subjects.
- FreeSurfer reconstruction for every subject.
- Parcellation for every subject.
"""

from pathlib import Path
import pandas as pd
import mindmap as mm



############## Parameters #################
# files
FOLDER = "/media/nino/DATA/SIMF/derivatives"

# simulation
N        = 20
DURATION = 1000

# source spike
ONSET    = 500
WIDTH    = 100

# spike propagation
CONDUCTION = 0.25
DELAY      = 30

# noise generator
TARGET_SNR  = 15
NOISE_RATIO = 0.9
############################################

def main():
    """
    main
    """
    reader   = mm.io.dataset_reader("configuration.yml")
    subjects = reader.config["SUBS"]
    summary  = pd.DataFrame(columns=["subject", "snr", "location"])

    folder = Path(FOLDER)
    if not folder.exists():
        raise FileNotFoundError("Can't find folder.")
    
    folder = folder/"simulations"
    folder.mkdir()

    for sub in subjects:
        data = reader(sub, "eeg", "fwd", "sc", "parc", "cras")

        sim = mm.utils.Simulation(
            forward      = data["fwd"],
            connectome   = data["sc"],
            info         = data["eeg"].info,
            duration     = DURATION,
            parcellation = data["parc"],
            cras         = data["cras"]
        )

        sim.set_spike_attr(width=WIDTH, onset=ONSET)
        sim.set_prop_attr(conduction=CONDUCTION, delay=DELAY)
        sim.set_noise_attr(snr=TARGET_SNR, noise_ratio=NOISE_RATIO)

        for i in range(N):
            sim.set_spike_attr(location="random")
            sim.run()

            # save eeg and groundtruths
            eeg           = sim.get_eeg(noise=True)
            gt_noise      = sim.get_activity(noise=True)
            gt_noise_free = sim.get_activity(noise=False)

            subfolder = folder/sub/f"sample{i+1:03}"
            subfolder.mkdir(parents=True)
            eeg.save(subfolder/"eeg_ave.fif", verbose="warning")
            gt_noise.save(subfolder/"ground-truth-noise", verbose="warning")
            gt_noise_free.save(subfolder/"ground-truth-noisefree", verbose="warning")

            # stack info
            summary.loc[i] = [sub, sim.adj_snr, sim.loc]

    # save summary info
    summary.to_csv(folder/"summary.csv")

    # add the samples in the configuration
    reader.config["SAMP"] = [f"sample{i+1:03}" for i in range(N)]
    mm.io.write_config("configuration_test.yml", reader.config)


if __name__ == "__main__":
    main()