import mindmap as mm
import mne
import numpy as np

import pyxu.abc as pxa
import pyxu.operator as pxo
import pyxu.opt.stop as pxst
from pyxu.opt.solver import PGD

def cset(
        eeg      : mne.Evoked,
        fwd      : mne.Forward,
        cst      : np.ndarray,
        lam      : float,
        dw       : float = 0.5,
        times    : list = None,
        sub      : str = None,
)-> mne.SourceEstimate:
    """
    Solve the inverse problem using Connectome Spectrum Electromagnetic Tomography (CSET).

    eeg: mne.Evoked
        Electroencephalograms. 

    fwd: mne.Forward 
        Forward solution (contains lead field matrix)

    cst: np.ndarray (N,N)
        Connectome Spectrum Transform 

    lam: float
        Regularizer parameter

    dw: float
        Depth weighting scale 

    times: list(float) 
        Reconstruct specific time points (in [ms]) only.

    Returns
    -------
    stc: mne.SourceEstimate 
        Source Time Currents
    """

    sfreq = eeg.info["sfreq"]
    if times is not None:
        idx   = eeg.time_as_index(times)
        Y     = eeg.data[:, idx]
        tstep = (times[1]-times[0]) if len(times)>1 else 0
    else:
        Y     = eeg.data
        times = eeg.times
        tstep = 1/sfreq

    # extract lead field matrix
    fwd = mne.convert_forward_solution(fwd, force_fixed=True, verbose="warning")
    L   = fwd["sol"]["data"].astype("float64")

    # standardize
    Ynorms = np.linalg.norm(Y, axis=0)
    Y      = Y/Ynorms[None,:]

    L      = L - np.mean(L, axis=0)
    Lnorms = np.linalg.norm(L, axis=0)**dw
    L      = L / Lnorms[None,:] 

    # component-wise transformation
    L = L @ cst.T

    # pyxu components
    L     = pxa.LinOp.from_array(L)
    prior = lam*pxo.L1Norm(dim_shape=cst.shape[0])
    recon = np.zeros((cst.shape[1], Y.shape[1]))
    sc    = pxst.MaxIter(1000)

    # ----- Optimization solving -----
    for t in range(Y.shape[1]):
        y     = Y[:,t]                                   # data time point
        s_hat = np.random.random(L.dim_shape[0])*10**(-9)    # initial guess

        likelihood = pxo.SquaredL2Norm(dim_shape=L.codim_shape).argshift(-y)*L
        likelihood.diff_lipschitz = likelihood.estimate_diff_lipschitz()

        solver = PGD(f=likelihood, g=prior, show_progress=False)

        solver.fit(x0=s_hat, stop_crit=sc)

        s_hat = solver.solution().squeeze()
        recon[:,t] = (cst.T @ s_hat) * Ynorms[t]

    # Store in MNE object
    vertices = [fwd["src"][0]["vertno"], fwd["src"][1]["vertno"]]
    stc      = mne.SourceEstimate(recon, vertices, tmin=times[0], tstep=tstep, subject=sub)

    return stc


def enet(
        eeg: mne.Evoked,
        fwd: mne.Forward,
        cst: np.ndarray,
        alpha: float,
        dw: float = 0.5,
        win_size: float = 0.1,
        sub: str = None,
        super: bool = False,
) -> mne.SourceEstimate:
    """
    Solve the inverse problem using a combination of elastic NET (eNET) and Connectome Spectrum Electromagnetic Tomography (CSET). eNET solves an L1 and
    L2 problem at the same time, forcing sparsity and smoothness at the same time.

    eeg: mne.Evoked
        Electroencephalograms. 

    fwd: mne.Forward 
        Forward solution (contains lead field matrix)

    cst: np.ndarray (N,N)
        Connectome Spectrum Transform 

    lam: float
        Regularizer parameter

    dw: float
        Depth weighting scale 

    win_size: float 
        Time window size in [s]

    super: bool
        Use super ENet if True !

    Returns
    -------
    stc: mne.SourceEstimate 
        Source Time Currents
    """
    # select time window 
    sfreq = eeg.info["sfreq"]
    _, pt = mm.utils.eeg_rising_phase(eeg, ratio=1)
    dt = int(win_size*sfreq/2)
    Y  = eeg.data[:, pt-dt:pt+dt]

    # extract lead field matrix
    fwd = mne.convert_forward_solution(fwd, force_fixed=True, verbose="warning")
    L   = fwd["sol"]["data"].astype("float64")

    # standardize
    Ynorms = np.linalg.norm(Y, axis=0)
    Y      = Y/Ynorms[None,:]

    if not super: 
        L      = L - np.mean(L, axis=0)
        Lnorms = np.linalg.norm(L, axis=0)**dw
        L      = L / Lnorms[None,:] 

        # component-wise transformation
        L = L @ cst.T

        # define artificial dataset
        lambda1 = 1-alpha
        lambda2 = alpha
        L = (1 + lambda2)**(-1/2) * np.vstack([L, lambda2**(1/2)*np.eye(L.shape[1])])
        Y = np.vstack([Y, np.zeros((L.shape[1], Y.shape[1]))])

        # regularization parameter
        gamma = lambda1 / (1+lambda2)**(1/2)

        # pyxu components
        L     = pxa.LinOp.from_array(L)
        prior = gamma*pxo.L1Norm(dim_shape=cst.shape[0])
        recon = np.zeros((cst.shape[1], Y.shape[1]))
        sc    = pxst.MaxIter(1000)

        # ----- Optimization solving -----
        for t in range(Y.shape[1]):
            y     = Y[:,t]                                   # data time point
            s_hat = (1+lambda2)**(1/2)*np.random.random(L.dim_shape[0])*10**(-9)    # initial guess

            likelihood = pxo.SquaredL2Norm(dim_shape=L.codim_shape).argshift(-y)*L
            likelihood.diff_lipschitz = likelihood.estimate_diff_lipschitz()

            solver = PGD(f=likelihood, g=prior, show_progress=False)

            solver.fit(x0=s_hat, stop_crit=sc)

            s_hat = solver.solution().squeeze()
            s_hat = (1+lambda2)**(1/2) * s_hat 
            recon[:,t] = (cst.T @ s_hat) * Ynorms[t]

    if super:

        # depth prior
        W = np.diag(np.linalg.norm(L, axis=0)**dw)

        # define artificial data set
        lambda1 = 1-alpha
        lambda2 = alpha
        L = (1 + lambda2)**(-1/2) * np.vstack([L, lambda2**(1/2)*W])
        Y = np.vstack([Y, np.zeros((L.shape[1], Y.shape[1]))])

        # regularization parameter
        gamma = lambda1 / (1+lambda2)**(1/2)

        # component-wise transformation
        L = L @ cst.T

        # pyxu components
        L     = pxa.LinOp.from_array(L)
        prior = gamma*pxo.L1Norm(dim_shape=cst.shape[0])
        recon = np.zeros((cst.shape[1], Y.shape[1]))
        sc    = pxst.MaxIter(1000)

        # ---- optimization ----
        for t in range(Y.shape[1]):
            y     = Y[:,t]
            s_hat = np.random.random(L.dim_shape[0])*10**(-9)

            likelihood = pxo.SquaredL2Norm(dim_shape=L.codim_shape).argshift(-y)*L
            likelihood.diff_lipschitz = likelihood.estimate_diff_lipschitz()

            solver = PGD(f=likelihood, g=prior, show_progress=False)

            solver.fit(x0=s_hat, stop_crit=sc)

            s_hat      = solver.solution().squeeze()
            s_hat      = (1+lambda2)**(1/2) * s_hat      # enet correction
            recon[:,t] = (cst.T @ s_hat) * Ynorms[t]

    # Store in MNE object
    t_peak = pt/sfreq
    t_start  = t_peak-(win_size/2)
    vertices = [fwd["src"][0]["vertno"], fwd["src"][1]["vertno"]]
    stc      = mne.SourceEstimate(recon, vertices, tmin=t_start, tstep=1/sfreq, subject=sub)

    return stc


def wcset(
        eeg: mne.Evoked,
        fwd: mne.Forward,
        cst: np.ndarray,
        gap: np.ndarray,
        lam: float,
        dw: float = 1/2,
        hw: float = 1/3,
        win_size: float = 0.1,
        scale: float = 1,
        sub: str = None,
):
    """
    Solve the inverse problem using weighted Connectome Spectrum Electromagnetic Tomography (wCSET). This technique applies unbalanced weights on the harmonics 
    according to frequency gaps.

    eeg: mne.Evoked
        Electroencephalograms. 

    fwd: mne.Forward 
        Forward solution (contains lead field matrix)

    cst: np.ndarray (N,N)
        Connectome Spectrum Transform 

    gap: np.ndarray (N,)
        Frequency gap between harmonics

    lam: float
        Regularizer parameter

    dw: float
        Depth weighting scale 

    hw: float
        Harmonic weighting

    win_size: float 
        Time window size in [s]

    scale: float
        Scale to scale value in case values are too small and pyxu always returns 0 due to computer resolution.

    Returns
    -------
    stc: mne.SourceEstimate 
        Source Time Currents
    """
    # select time window 
    sfreq = eeg.info["sfreq"]
    _, pt = mm.utils.eeg_rising_phase(eeg, ratio=1)
    dt = int(win_size*sfreq/2)
    Y  = eeg.data[:, pt-dt:pt+dt]

    # extract lead field matrix
    fwd = mne.convert_forward_solution(fwd, force_fixed=True, verbose="warning")
    L   = fwd["sol"]["data"].astype("float64")

    # standardize
    Ynorms = np.linalg.norm(Y, axis=0)
    Y      = Y/Ynorms[None,:]

    L      = L - np.mean(L, axis=0)
    Lnorms = np.linalg.norm(L, axis=0)**dw
    L      = L / Lnorms[None,:] 

    # harmonic weights
    arr1 = np.abs(gap - np.roll(gap, 1))
    arr2 = np.abs(gap - np.roll(gap, -1))
    D    = np.minimum(arr1, arr2)**hw
    D    = D/np.linalg.norm(D)
    D    = np.diag(D)

    # component-wise transformation
    L = L @ cst.T @ D

    # pyxu components
    L     = pxa.LinOp.from_array(L)
    prior = lam * pxo.SquaredL2Norm(dim_shape=cst.shape[0])
    recon = np.zeros((cst.shape[1], Y.shape[1]))
    sc    = pxst.MaxIter(1000)

    # ----- Optimization solving -----
    for t in range(Y.shape[1]):
        y     = Y[:,t]                                   # data time point
        s_hat = np.random.random(L.dim_shape[0])*10**(-9)    # initial guess

        likelihood = pxo.SquaredL2Norm(dim_shape=L.codim_shape).argshift(-y)*L + prior
        likelihood.diff_lipschitz = likelihood.estimate_diff_lipschitz()

        solver = PGD(f=likelihood, g=prior, show_progress=False)

        solver.fit(x0=s_hat, stop_crit=sc)

        s_hat = solver.solution().squeeze()
        recon[:,t] = (cst.T @ s_hat) * Ynorms[t] / scale

    # Store in MNE object
    t_peak = pt/sfreq
    t_start  = t_peak-(win_size/2)
    vertices = [fwd["src"][0]["vertno"], fwd["src"][1]["vertno"]]
    stc      = mne.SourceEstimate(recon, vertices, tmin=t_start, tstep=1/sfreq, subject=sub)

    return stc





def sloreta(eeg, fwd, noise_cov=None, lam=0.1, times=None):
    """
    Source reconstruction using standardized low resolution electromagnetic tomography (sLORETA).

    Parameters
    ----------
    eeg: mne.Evoked
        EEG recordings.

    fwd: mne.Forward
        Forward solution.

    lam: float
        L2 regularizer parameter.

    win_size: float
        Time window size to reconstruct around peak. If None, and time_points is None, reconstruct whole time series.
    
    times: list(float)
        Times to reconstruct.

    Returns
    -------
    recon: mne.SourceEstimate
        Source reconstruction.
    """
    
    if noise_cov is None:
        noise_cov = mne.make_ad_hoc_cov(eeg.info, std={"eeg":1})    # identity matrix
    
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
    if times is not None: 
        idx      = stc.time_as_index(times)
        data     = stc.data[:,idx]
        vertices = [fwd["src"][0]["vertno"], fwd["src"][1]["vertno"]]

        tstep = (times[1]-times[0]) if len(times)>1 else 0

        return mne.SourceEstimate(data, vertices, tmin=times[0], tstep=tstep, subject=stc.subject)

    else:
        return stc