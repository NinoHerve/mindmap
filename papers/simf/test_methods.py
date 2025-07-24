"""
Repeat simulations multiple times and evaluate each reconstruction method every time.

Requirements
------------
- FreeSurfer reconstruction
- Parcellation
- Connectome
- Forward solution

Outputs
-------
- Table of time correlations
"""

import numpy as np
import pandas as pd
import pygsp as pg
import mindmap as mm

############### Parameters ##############
# configuration
SUBJECT  = "vepcon_sub-10"
OUT_FILE = "test_methods.csv"

# experiment
N_REGIONS   = 100
REPETITIONS = 10

# simulation
DURATION    = 1000
TARGET_SNR  = 15
RANGE       = [0.5, 0.63]

WIDTH       = 100
ONSET       = 500
CONDUCTION  = 0.25
DELAY       = 30
NOISE_RATIO = 0.9

# reconstructions
METHODS     = ["cset", "wcset", "sloreta"]
LAM_CSET    = 0.1
LAM_WCSET   = 0.1
LAM_SLORETA = 0.1
#########################################

def main():
    """
    main
    """
    reader = mm.io.dataset_reader("configuration.yml")
    data   = reader(SUBJECT, "fwd", "eeg", "cst", "sc", "parc", "cras")

    # initialize simulation
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

    # select regions
    regions = sim.get_regions()
    n       = min(len(regions), N_REGIONS)
    regions = np.random.choice(regions, n, replace=False)

    # compute weights for wcset
    if "wcset" in METHODS:
        pos   = mm.utils.get_src_position(data["fwd"]["src"])
        graph = pg.graphs.Graph(data["sc"], lap_type="normalized", coords=pos)
        graph.compute_fourier_basis()

        eigvals = graph.e
        weights = mm.utils.harmonic_weights(eigvals, relaxation=0.2)

    # run repetitions
    times   = data["eeg"].times
    times   = times[(times>=RANGE[0]) & (times<RANGE[1])]  # select only range
    results = pd.DataFrame(columns=["timepoint", "correlation", "method", "region", "half", "peak"])

    i = 0
    for j, r in enumerate(regions):
        sim.set_spike_attr(location=r)
        for _ in range(REPETITIONS):
            sim.run()
            eeg = sim.get_eeg(noise=True).crop(tmin=RANGE[0], tmax=RANGE[1], include_tmax=False)
            gt  = sim.get_activity(noise=False).crop(tmin=RANGE[0], tmax=RANGE[1], include_tmax=False)

            for m in METHODS:
                if m == "cset":
                    stc = mm.cset(eeg, data["fwd"], data["cst"], lam=LAM_CSET)
                elif m == "wcset":
                    stc = mm.wcset(eeg, data["fwd"], data["cst"], weights=weights, lam=LAM_WCSET)
                elif m == "sloreta":
                    stc = mm.sloreta(eeg, data["fwd"], lam=LAM_SLORETA)
                else:
                    print("\033[93mNone implemented method\033[0m")
                    continue

                corr = mm.utils.compute_stc_correlation(stc, gt)
                half = mm.utils.eeg_rising_phase(eeg, ratio=0.5)
                peak = mm.utils.eeg_rising_phase(eeg, ratio=1)

                # store in DataFrame
                size = len(times)
                sample = pd.DataFrame({
                    "timepoint": times,
                    "correlation": corr,
                    "method": [m]*size,
                    "region": [r]*size,
                    "half": times==half,
                    "peak": times==peak,
                })
                results = pd.concat([results, sample])
                i += 1

        print(f"\033[94mNumber of regions tested: [{j+1}/{len(regions)}]\033[0m")


    # save results
    results.to_csv(OUT_FILE, index=False)



if __name__ == "__main__":
    main()
