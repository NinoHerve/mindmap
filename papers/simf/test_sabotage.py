"""
Repeat simulations multiple times and evaluate CSET while sabotaging the connectome.
We implement reconstructions with 0%, 20%, 40%, 60%, 80% and 100% permutations on the original connectome used for simulating.
Additionally, we reconstruct using the geometry connectome and the atlas-based connectome.

Requirements
------------
- FreeSurfer reconstruction
- Parcellation
- Connectome
- Forward solution

Outputs
-------
- Table of peak phase correlation.
"""
import pandas as pd
import pygsp as pg
import mindmap as mm


############# Parameters ##############
OUT_FILE    = "sabotage.csv"                # Saving file
SUB         = "vepcon_sub-10"               # template subject
REPS        = 1000                           # number of repetitions
PERM        = [0, 0.2, 0.4, 0.6, 0.8, 1]    # percentage of permutations
DMAX        = 10                            # maximal distance for neighbor connections (brain geometry)
LAM_CSET    = 0.1                           # CSET regularizer parameter
LAM_WCSET   = 0.1                           # wCSET regularizer parameter
LAM_SLORETA = 0.001                         # sLORETA regularizer parameter
#######################################

def main():
    """
    main
    """
    # load data
    reader = mm.io.dataset_reader("configuration.yml")
    data   = reader(SUB, "sc", "cst", "cst_tmp", "fwd", "eeg", "parc", "cras")

    # instanciate simulation
    sim = mm.utils.Simulation(data["fwd"], data["sc"], data["eeg"].info, data["parc"], data["cras"])

    # build geometry
    geometry = mm.utils.build_geometry(data["fwd"]["src"], dmax=10, conn_hemis=True)
    geometry = pg.graphs.Graph(geometry, lap_type="normalized")
    geometry.compute_fourier_basis()

    # results
    results = pd.DataFrame(columns=["subject", "method", "location", "correlation", "sabotage"])

    # iterate through repetitions
    for rep in range(REPS):
        sim.set_spike_attr(location="random")
        sim.run()

        eeg = sim.get_eeg(noise=True)
        gt  = sim.get_activity(noise=False)

        # select only peak phase
        peak_time     = mm.utils.eeg_rising_phase(eeg, ratio=1)
        eeg.crop(tmin = peak_time,tmax=peak_time)
        gt.crop(tmin=peak_time, tmax=peak_time)

        stc_geom    = mm.cset(eeg, data["fwd"], cst=geometry.U.T, lam=LAM_CSET)
        stc_temp    = mm.cset(eeg, data["fwd"], cst=data["cst_tmp"], lam=LAM_CSET)
        stc_sloreta = mm.sloreta(eeg, data["fwd"], lam=LAM_SLORETA)

        corr_geom    = mm.utils.compute_stc_correlation(stc_geom, gt)[0]
        corr_temp    = mm.utils.compute_stc_correlation(stc_temp, gt)[0]
        corr_sloreta = mm.utils.compute_stc_correlation(stc_sloreta, gt)[0]

        sub_results = pd.DataFrame({
            "subject"    : [SUB]*3,
            "method"     : ["geometry", "template", "sloreta"],
            "location"   : [sim.loc]*3,
            "correlation": [corr_geom, corr_temp, corr_sloreta],
            "sabotage"   : [0]*3
        })
        results = pd.concat([results, sub_results])

        # sabotage connectome with different degrees of permutations
        for sabo in PERM:
            permutations = int(sabo * data["sc"].sum() /2)
            sc           = mm.utils.sabotage_connectome(data["sc"], deletions=permutations, additions=permutations)

            sc = pg.graphs.Graph(sc, lap_type="normalized")
            sc.compute_fourier_basis()
            weights = mm.utils.harmonic_weights(eigvals=sc.e, relaxation=0.2)

            stc_cset  = mm.cset(eeg, data["fwd"], cst=sc.U.T, lam=LAM_CSET)
            stc_wcset = mm.wcset(eeg, data["fwd"], cst=sc.U.T, weights=weights, lam=LAM_WCSET)

            corr_cset  = mm.utils.compute_stc_correlation(stc_cset, gt)[0]
            corr_wcset = mm.utils.compute_stc_correlation(stc_wcset, gt)[0]

            sub_results = pd.DataFrame({
                "subject"    : [SUB]*2,
                "method"     : ["cset", "wcset"],
                "location"   : [sim.loc]*2,
                "correlation": [corr_cset, corr_wcset],
                "sabotage"   : [sabo]*2,
            })
            results = pd.concat([results, sub_results])

        print(f"\033[94mNumber of repetitions: [{rep+1}/{REPS}]\033[0m")

    # save results
    results.to_csv(OUT_FILE, index=False)

if __name__ == "__main__":
    main()