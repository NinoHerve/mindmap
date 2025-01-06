import mindmap as mm
import mne
import numpy as np

import pyxu.abc as pxa
import pyxu.operator as pxo
import pyxu.opt.stop as pxst
from pyxu.opt.solver import PGD

def recon_cset(
        eeg: mne.Evoked,
        fwd: mne.Forward, 
        cst: np.ndarray, 
        l_cst1: float   = 1, 
        l_cst2: float   = None, 
        l_dw: float     = 0.8, 
        cut_freq: int   = None,
        win_size: float = None, 
        n_iter: int     = 1000
) -> mne.SourceEstimate:
    """
    Source reconstruction using Connectome Spectrum Electromagnetic Tomography (CSET)

    Parameters
    ----------
    eeg: mne.Evoked
        Electroencephalography recordings.

    fwd: mne.Forward
        Forward solution.

    cst: np.ndarray
        Connectome spectrum transform.

    l_cst1: float
        L1 connectome spectrum regularizer parameter.

    cst2: float
        L2 connectome spectrum regularizer parameter
    
    l_dw: float
        L2 source depth weight regularizer parameter.
    
    cut_freq: int
        Cutoff frequency for connectome spectrum.

    win_size: float
        Time window in [s]. If None, we only reconstruct the 50% rising and peak phase of the EEG.

    n_iter: int
        Number of iterations.

    Returns
    -------
    recon: mne.SourceEstimate 
        Source reconstruction.
    """
    
    # prepare eeg
    if win_size is None:
        _, ht = mm.utils.eeg_rising_phase(eeg, ratio=0.5)       # 50% rising phase
        _, pt = mm.utils.eeg_rising_phase(eeg, ratio=1)         # peak phase
        b     = eeg.data[:, [ht, pt]]

        sfreq   = eeg.info["sfreq"]
        t_start = ht/sfreq
        t_end   = pt/sfreq 
        sfreq   = 2/(t_end-t_start)
    else:
        sfreq = eeg.info["sfreq"]
        _, pt = mm.utils.eeg_rising_phase(eeg, ratio=1)         # peak phase
        dt    = int(win_size * sfreq /2)
        b     = eeg.data[:, pt-dt:pt+dt]

        t_start = pt/sfreq-(win_size/2) 

    bnorms = np.linalg.norm(b, axis=0)
    b      = b/bnorms[None,:]           # normalize data

    # forward operator 
    A = mne.convert_forward_solution(fwd, force_fixed=True, verbose="warning")
    A = A["sol"]["data"].astype("float64")

    # filter connectome spectrum
    if cut_freq is None:
        cst = cst
    else:
        cst = cst[:cut_freq,:]

    # sparse prior for connectome spectrum
    if l_cst1 is None:
        priorCST1 = None 
    else:
        priorCST1 = l_cst1 * pxo.L1Norm(dim_shape=cst.shape[0])

    # smooth prior for connectome spectrum
    if l_cst2 is None:
        priorCST2 = None 
    else:
        priorCST2 = l_cst2*pxo.SquaredL2Norm(dim_shape=cst.shape[0])

    # depth weighting
    if l_dw is None:
        priorW = None 
    else:
        W = np.linalg.norm(A, axis=0)
        W = np.diag(W) @ cst.T 
        W = pxa.LinOp.from_array(W)

        priorW = l_dw * pxo.SquaredL2Norm(dim_shape=cst.shape[1])*W 


    # modified forward operator
    A = pxa.LinOp.from_array(A @ cst.T)

    # prepare inverse problem
    alpha0 = np.random.random(A.dim_shape)*10**(-9)
    stop   = pxst.MaxIter(n_iter)

    # reconstruct time window
    recon = np.zeros((cst.shape[1], b.shape[1]))
    for t in range(b.shape[1]):
        sample = b[:,t]

        likelihood = pxo.SquaredL2Norm(dim_shape=A.codim_shape).argshift(-sample)*A

        if l_cst2 is not None:
            likelihood = likelihood + priorCST2 
        if l_dw is not None:
            likelihood = likelihood + priorW

        likelihood.diff_lipschitz = likelihood.estimate_diff_lipschitz()

        solver = PGD(f=likelihood, g=priorCST1, show_progress=False)

        solver.fit(x0=alpha0, stop_crit=stop)

        alpha      = solver.solution().squeeze()
        recon[:,t] = (cst.T @ alpha) * bnorms[t]

    # arrange result in an mne.SourceEstimate object
    vertices = [fwd["src"][0]["vertno"], fwd["src"][1]["vertno"]]
    stc      = mne.SourceEstimate(recon, vertices, tmin=t_start, tstep=1/sfreq)

    return stc



def recon_sloreta(eeg, fwd, lam=0.1, win_size=None):
    """
    Source reconstruction using standardized low resolution electromagnetic tomography.

    Parameters
    ----------
    eeg: mne.Evoked
        EEG recordings.

    fwd: mne.Forward
        Forward solution.

    lam: float
        L2 regularizer parameter.

    
    win_size: float
        Time window size to reconstruct around peak.

    Returns
    -------
    recon: mne.SourceEstimate
        Source reconstruction.
    """

    noise_cov = mne.make_ad_hoc_cov(eeg.info, std={"eeg":1})
    
    # prepare eeg
    eeg.set_eeg_reference(projection=True)

    # solve inverse problem
    inverse = mne.minimum_norm.make_inverse_operator(
        eeg.info,
        fwd,
        noise_cov,
        fixed=True,
        depth=0,
        verbose="warning",
    )

    stc = mne.minimum_norm.apply_inverse(
        eeg, 
        inverse, 
        lambda2=lam,
        method="sLORETA",
        verbose="warning",
    )

    # crop time window
    if win_size is None:
        _, ht = mm.utils.eeg_rising_phase(eeg, ratio=0.5)       # 50% rising phase
        _, pt = mm.utils.eeg_rising_phase(eeg, ratio=1)         # peak phase
        
        stc.data = stc.data[:, [ht, pt]]
    else:
        _, pt    = mm.utils.eeg_rising_phase(eeg, ratio=1)
        dt       = int(win_size * eeg.info["sfreq"] /2)
        stc.data = eeg.data[:, pt-dt:pt+dt]

    return stc
    