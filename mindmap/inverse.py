"""
Module Name: inverse.py
Description: Class to run inverse problems.

This module provides the InverseOperator class to run inverse problems.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

from abc import ABC, abstractmethod

import numpy as np
import mne

import pyxu.abc as pxa
import pyxu.operator as pxo
import pyxu.opt.stop as pxst
from pyxu.opt.solver import PGD

from mindmap.head.source_space import make_stc
from mindmap.head.connectome import compute_harmonic_prior


class InverseOperator(ABC):
    """
    Class for inverse operators
    """
    # m defines the number of sensors while n defines the number of sources
    info: mne._fiff.meas_info.Info      # Sensor info object

    # Core data
    lead  : mne.Forward                      # Leadfield matrix (a.k.a forward solution) (m,n)
    ncov  : mne.cov.Covariance              # Noise Covariance matrix (sensor noise) (m,m)
    prior : np.ndarray                      # Prior Mixing matrix (n,n)
    basis : np.ndarray                      # Basis transformation matrix (a.k.a component-wise extraction matrix) (n,n)

    # whitening
    white : np.ndarray                       # Whitener (m,m)
    color : np.ndarray                       # Colorer (m,m)

    # depth prior
    depth : np.ndarray                      # Depth Prior matrix

    # verbosity (global attribute)
    verbose = "warning"

    def __init__(
        self,
        info      : mne.Info,
        leadfield : mne.Forward,
        noise_cov : mne.Covariance = None,
        depth     : float = 0.8,
    ):
        # EEG info
        self.info = info

        # lead field
        self.lead  = mne.convert_forward_solution(leadfield, force_fixed=True, verbose="warning")  # we always work with fixed orientation

        if noise_cov is None:
            self.ncov  = mne.make_ad_hoc_cov(self.info, std={"eeg":1})
            self.white = np.eye(self.lead["nchan"])
            self.color = np.eye(self.lead["nchan"])
        else:
            self.ncov = noise_cov
            self.white, self.color = self.compute_whitener()

        # average projector
        self.avg_proj = self.make_average_projector()

        # depth prior
        self.depth = self.compute_depth_prior(exponent = depth)


    @abstractmethod
    def __call__(self):
        pass

    def compute_whitener(self):
        """
        Computes whitener and colorer.

        New attributes
        ----------
        W: np.ndarray (m,m)
            Whitener. Let UDU.T be the eigdendecomposition of Nc. Then W = D^(-1/2)U.T

        C: np.ndarray (m,m)
            Colorer. C = W^(-1)

        whiten_method: str
            Method used for whitening.
        """

        white, _, color = mne.cov.compute_whitener(
            noise_cov      = self.ncov,
            info           = self.info,
            picks          = self.select_channels(),
            pca            = "white",
            return_colorer = True,
            verbose        = self.verbose,
        )

        return white, color



    def scaling(self, nave):
        pass

    def compute_depth_prior(self, exponent):
        """
        Computes the depth priors. The depth prior is a diagonal matrix whose elements serve to 
        weight sources according to their depth, as measured by the columns of the leadfield 
        matrix. The purpose is to counteract the minimum estimate bias toward superficial sources.

        Parameters
        ----------
        exponent: float [0-1]
            Exponent for depth weighting.

        limit: float
            I don't know what limit does but it changes a lot.

        New attributes
        ----------
        Dp: np.ndarray (n,n)
            Depth prior matrix.

        exp: float [0-1]
            Exponent used for depth weighting.
        """
        lead  = self.lead["sol"]["data"].astype("float64")
        norms = np.linalg.norm(lead, axis=0)

        depth = np.diag(1/norms**exponent)

        return depth

    def make_average_projector(self):
        """
        Make average projector for average referencing.

        Returns
        -------
        avg_proj: np.ndarray (n,n)
            Average projector.
        """
        n = self.lead["nchan"]

        # average projector = I - (1/n)11.T
        avg_proj = np.eye(n) - (1/n)*np.ones(shape=(n,n))

        return avg_proj
        
    def select_channels(self):
        """
        Select channels available in both info and ncov.
        """
        info_names = set(self.info["ch_names"])
        info_bads  = set(self.info["bads"])

        ncov_names = set(self.ncov["names"])
        ncov_bads  = set(self.ncov["bads"])

        ch_names = info_names & ncov_names
        bads     = info_bads | ncov_bads

        ch_names -= bads

        return list(ch_names)


class CSET(InverseOperator):
    """
    Implements Connectome Spectrum Electromagnetic Tomography.
    """
    info      : mne.Info        # EEG metadata
    leadfield : mne.Forward     # forward operator
    eig_vec   : np.ndarray      # connectome spectrum transform
    noise_cov : mne.Covariance  # Noise covariance
    depth     : float           # exponent for depth weighting

    def __init__(
        self,
        info      : mne.Info,
        leadfield : mne.Forward,
        eig_vec   : np.ndarray,          # required
        noise_cov : np.ndarray = None,
        depth     : float = 0,
    ):
        super().__init__(
            info      = info,
            leadfield = leadfield,
            noise_cov = noise_cov,
            depth     = depth,
        )

        self.basis = eig_vec

    def __call__(self, data, lam):
        """
        Applies the Connectome Spectrum Electromagnetic Tomography (CSET) source reconstruction
        method. 

        Parameter
        ---------
        data: mne.Evoked
            Sensor data to project to source space.

        lam: float
            Regularization parameter.

        Return
        ------
        stc: mne.SourceEstimate 
            Source Time Currents.
        """
        n = self.lead["nsource"]   # number of sources
        t = len(data.times)     # number of time points

        # -------------------------------------------------------------
        # Prepare leadfield
        # ¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
        # L = WPLD⁻¹U
        #
        # where W is a whitener, P is the average projection matrix,
        # D is the depth weighting prior and U are the connectome's
        # laplacian eigenvectors
        # -------------------------------------------------------------

        # average projection
        dat  = self.avg_proj @ data.data
        lead = self.avg_proj @ self.lead["sol"]["data"].astype("float64")

        # whiten (L = WL)
        dat  = self.white @ dat
        lead = self.white @ lead

        # depth prior
        inv_depth = np.linalg.inv(self.depth)
        lead      = lead @ inv_depth

        # connectome spectrum transform
        lead = lead @ self.basis

        # -----------------------
        # Solve CSET using pyxu
        # ¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
        # -----------------------

        # Instanciate pyxu objects
        lead  = pxa.LinOp.from_array(lead)          # forward matrix
        prior = lam * pxo.L1Norm(dim_shape=n)       # l1-regularizer ||ŝ||_1
        sol   = np.zeros((n, t))                    # solution holder
        sc    = pxst.MaxIter(1000)                  # stopping criterion

        # iterate over time points
        for tp in range(t):
            # data
            y     = dat[:,tp]
            s_hat = np.random.random(n)*10**(-9)    # initial guess

            # likelihood (||Lŝ - y||_2)²
            likelihood = pxo.SquaredL2Norm(dim_shape=lead.codim_shape).argshift(-y)*lead
            likelihood.diff_lipschitz = likelihood.estimate_diff_lipschitz()

            # solver = Proximal Gradient Descent
            solver = PGD(f=likelihood, g=prior, show_progress=False)

            solver.fit(x0=s_hat, stop_crit=sc)

            # store timepoint's solution
            sol[:,tp] = solver.solution().squeeze()

        # revert back to source space
        sol = inv_depth @ self.basis @ sol

        # store result in MNE object
        vertices = [self.lead["src"][0]["vertno"], self.lead["src"][1]["vertno"]]
        stc      = mne.SourceEstimate(sol, vertices, tmin=data.times[0], tstep=1/data.info["sfreq"])

        return stc

class wCSET(InverseOperator):
    """
    Applies the weighted Connectome Spectrum Electromagnetic Tomography (CSET) source 
    reconstruction method. 

    Attributes
    ---------
    data: mne.Evoked
        Sensor data to project to source space.

    Return
    ------
    stc: mne.SourceEstimate 
        Source Time Currents.
    """
    def __init__(
        self,
        info,
        leadfield : mne.Forward,
        eig_vec   : np.ndarray,
        eig_val   : np.ndarray,
        noise_cov : mne.Covariance = None,
        depth     : float = 0.5,
        harm      : float = 0.3,
    ):
        """
        Init

        Parameters
        ----------
        info: mne.Info
            Info variable of MNE.

        leadfield: mne.Forward
            Forward operator, a.k.a leadfield matrix.

        eig_vec: np.ndarray
            Connectome laplacian eigenvectors.

        eig_val: np.ndarray
            Connectome laplacian eigenvalues.

        noise_cov: mne.Covariance
            Noise covariance matrix. Default is identity matrix.

        lambda_: float
            Regularization parameter.

        depth: float [0-1]
            Depth exponent for depth weighting.

        harm: float [0-1]
            Harmonic exponent for harmonic weighting.
        """
        super().__init__(
            info      = info,
            leadfield = leadfield,
            noise_cov = noise_cov,
            depth     = depth,
        )

        # Harmonic prior
        self.basis = eig_vec
        self.harm  = compute_harmonic_prior(eig_val, relaxation=harm)


    def __call__(
            self,
            data : mne.Evoked,
            lam  : float = 0.01,
        ) -> mne.SourceEstimate:
        """
        Applies the weighted Connectome Spectrum Electromagnetic Tomography (wCSET) 
        source reconstruction method. 

        Parameters
        ----------
        data: mne.Evoked 
            Sensor data. 

        lam_harm: float
            Harmonic weighting regularization parameter.

        lam_depth: float
            Depth weighting regularization parameter.

        Return
        ------
        stc: mne.SourceEstimate 
            Source Time Currents.
        """
        # apply average referencing
        lead = self.avg_proj @ self.lead["sol"]["data"].astype("float64")
        y    = self.avg_proj @ data.data

        # build kernel
        inv_op = self.build_kernel(lead, lam)

        # compute solution
        # - Solution is computed in the graph spectrum space
        # - Basis transform is used to revert back to dipole space
        #sol = self.basis @ inv_op @ data.data
        sol = inv_op @ y

        # store result in MNE object
        stc = make_stc(self.lead["src"], sol, data.tmin, 1/data.info["sfreq"])

        return stc


    # def build_kernel(self, proj):
    #     """
    #     Computes the linear inverse operator (kernel). In CSET we reconstruct the
    #     connectome spectrum.

    #     Parameters
    #     ----------
    #     proj: np.ndarray (m,m)
    #         Projection matrix (includes average referencing).

    #     Attributes
    #     ----------
    #     lead: mne.Forward 
    #         Leadfield (forward solution)
        
    #     white: np.ndarray 
    #         Whitener

    #     depth: np.ndarray 
    #         Depth prior

    #     basis: np.ndarray 
    #         Basis transform (a.k.a component wise extraction)

    #     prior: np.ndarray 
    #         Prior mixing matrix

    #     lam: float
    #         Regularizer parameter.

    #     Returns
    #     -------
    #     T: np.ndarray
    #         Inverse operator.
    #     """
    #     # sensor weighting (whitening @ projections)
    #     sw = np.einsum("ik,kj->ij", self.white, proj)

    #     # leadfield weighting (sensor weighting @ leadfield @ depth weighting @ basis transform)
    #     lw = self.lead["sol"]["data"].astype("float64")
    #     lw = np.einsum("ik,kj->ij", sw, lw)
    #     lw = np.einsum("ij,j->ij", lw, self.depth)
    #     lw = np.einsum("ik,kj->ij", lw, self.basis)

    #     # kernel (L.T@L + lam*D.T@D)⁻¹ @ L.T
    #     inv_op = np.linalg.inv(lw.T @ lw + self.lam * (self.prior.T @ self.prior)) @ lw.T

    #     return inv_op

    def build_kernel(self, lead, lam):
        """
        Build kernel
        """
        # prior: D_h@U.T@D_d -> (D_h@U.T@D_d).T @ (D_h@U.T@D_d) = D_d@U@D_h**2@U.T@ D_d
        prior = self.depth @ self.basis @ self.harm**2 @ self.basis.T @ self.depth

        # inverse prior
        inv_prior = np.linalg.inv(prior)     # sensor inversion might make no sense due to this inverse

        # kernel with sensor inversion (R⁻¹ L.T(L R⁻¹ L.T + lam*W)⁻¹)
        inv_op = inv_prior @ lead.T @ np.linalg.inv(lead @ inv_prior @ lead.T + lam*self.white)

        return inv_op



    def make_resolution_matrix(self, lam):
        """
        Make resolution matrix.

        Parameters
        ----------
        info: mne.Info
            Evoked info object. Needed for eventual projectors.

        Returns
        -------
        resmat: np.ndarray (n,n)
            Resolution matrix.
        """
        # apply average referencing
        lead = self.avg_proj @ self.lead["sol"]["data"].astype("float64")

        # inverse operator
        inv_op = self.build_kernel(lead, lam)

        # resolution matrix
        resmat = inv_op @ lead

        return resmat

    
    def make_harmonic_resolution_matrix(self, lam):
        """
        Make harmonic resolution matrix.

        Parameters
        ----------
        info: mne.Info
            Evoked info object. Needed for eventual projectors.

        Returns
        -------
        harmmat: np.ndarray (n,n)
            Resolution matrix
        """
        # apply average referencing
        lead = self.avg_proj @ self.lead["sol"]["data"].astype("float64")

        # inverse operator
        inv_op = self.build_kernel(lead, lam)

        # resolution matrix
        resmat = self.basis.T @ inv_op @ lead @ self.basis

        return resmat




# OLD CODE

# def cset(
#         eeg      : mne.Evoked,
#         fwd      : mne.Forward,
#         cst      : np.ndarray,
#         lam      : float,
#         dw       : float = 0.5,
#         times    : list = None,
# )-> mne.SourceEstimate:
#     """
#     Solve the inverse problem using Connectome Spectrum Electromagnetic Tomography (CSET).

#     eeg: mne.Evoked
#         Electroencephalograms. 

#     fwd: mne.Forward 
#         Forward solution (contains lead field matrix)

#     cst: np.ndarray (N,N)
#         Connectome Spectrum Transform 

#     lam: float
#         Regularizer parameter

#     dw: float
#         Depth weighting scale 

#     times: list(float) 
#         Reconstruct specific time points (in [ms]) only.

#     Returns
#     -------
#     stc: mne.SourceEstimate 
#         Source Time Currents
#     """

#     sfreq = eeg.info["sfreq"]
#     if times is not None:
#         idx   = eeg.time_as_index(times)
#         Y     = eeg.data[:, idx]
#         tstep = (times[1]-times[0]) if len(times)>1 else 0
#     else:
#         Y     = eeg.data
#         times = eeg.times
#         tstep = 1/sfreq

#     # extract lead field matrix
#     fwd = mne.convert_forward_solution(fwd, force_fixed=True, verbose="warning")
#     L   = fwd["sol"]["data"].astype("float64")

#     # standardize
#     Ynorms = np.linalg.norm(Y, axis=0)
#     Y      = Y/Ynorms[None,:]

#     L      = L - np.mean(L, axis=0)
#     Lnorms = np.linalg.norm(L, axis=0)**dw
#     L      = L / Lnorms[None,:] 

#     # component-wise transformation
#     L = L @ cst.T

#     # pyxu components
#     L     = pxa.LinOp.from_array(L)
#     prior = lam*pxo.L1Norm(dim_shape=cst.shape[0])
#     recon = np.zeros((cst.shape[1], Y.shape[1]))
#     sc    = pxst.MaxIter(1000)

#     # ----- Optimization solving -----
#     for t in range(Y.shape[1]):
#         y     = Y[:,t]                                   # data time point
#         s_hat = np.random.random(L.dim_shape[0])*10**(-9)    # initial guess

#         likelihood = pxo.SquaredL2Norm(dim_shape=L.codim_shape).argshift(-y)*L
#         likelihood.diff_lipschitz = likelihood.estimate_diff_lipschitz()

#         solver = PGD(f=likelihood, g=prior, show_progress=False)

#         solver.fit(x0=s_hat, stop_crit=sc)

#         s_hat = solver.solution().squeeze()
#         recon[:,t] = (cst.T @ s_hat) * Ynorms[t]

#     # Store in MNE object
#     vertices = [fwd["src"][0]["vertno"], fwd["src"][1]["vertno"]]
#     stc      = mne.SourceEstimate(recon, vertices, tmin=times[0], tstep=tstep)

#     return stc


# def enet(
#         eeg: mne.Evoked,
#         fwd: mne.Forward,
#         cst: np.ndarray,
#         alpha: float,
#         dw: float = 0.5,
#         win_size: float = 0.1,
#         sub: str = None,
#         super: bool = False,
# ) -> mne.SourceEstimate:
#     """
#     Solve the inverse problem using a combination of elastic NET (eNET) and Connectome Spectrum Electromagnetic Tomography (CSET). eNET solves an L1 and
#     L2 problem at the same time, forcing sparsity and smoothness at the same time.

#     eeg: mne.Evoked
#         Electroencephalograms. 

#     fwd: mne.Forward 
#         Forward solution (contains lead field matrix)

#     cst: np.ndarray (N,N)
#         Connectome Spectrum Transform 

#     lam: float
#         Regularizer parameter

#     dw: float
#         Depth weighting scale 

#     win_size: float 
#         Time window size in [s]

#     super: bool
#         Use super ENet if True !

#     Returns
#     -------
#     stc: mne.SourceEstimate 
#         Source Time Currents
#     """
#     # select time window 
#     sfreq = eeg.info["sfreq"]
#     _, pt = mm.utils.eeg_rising_phase(eeg, ratio=1)
#     dt = int(win_size*sfreq/2)
#     Y  = eeg.data[:, pt-dt:pt+dt]

#     # extract lead field matrix
#     fwd = mne.convert_forward_solution(fwd, force_fixed=True, verbose="warning")
#     L   = fwd["sol"]["data"].astype("float64")

#     # standardize
#     Ynorms = np.linalg.norm(Y, axis=0)
#     Y      = Y/Ynorms[None,:]

#     if not super: 
#         L      = L - np.mean(L, axis=0)
#         Lnorms = np.linalg.norm(L, axis=0)**dw
#         L      = L / Lnorms[None,:] 

#         # component-wise transformation
#         L = L @ cst.T

#         # define artificial dataset
#         lambda1 = 1-alpha
#         lambda2 = alpha
#         L = (1 + lambda2)**(-1/2) * np.vstack([L, lambda2**(1/2)*np.eye(L.shape[1])])
#         Y = np.vstack([Y, np.zeros((L.shape[1], Y.shape[1]))])

#         # regularization parameter
#         gamma = lambda1 / (1+lambda2)**(1/2)

#         # pyxu components
#         L     = pxa.LinOp.from_array(L)
#         prior = gamma*pxo.L1Norm(dim_shape=cst.shape[0])
#         recon = np.zeros((cst.shape[1], Y.shape[1]))
#         sc    = pxst.MaxIter(1000)

#         # ----- Optimization solving -----
#         for t in range(Y.shape[1]):
#             y     = Y[:,t]                                   # data time point
#             s_hat = (1+lambda2)**(1/2)*np.random.random(L.dim_shape[0])*10**(-9)    # initial guess

#             likelihood = pxo.SquaredL2Norm(dim_shape=L.codim_shape).argshift(-y)*L
#             likelihood.diff_lipschitz = likelihood.estimate_diff_lipschitz()

#             solver = PGD(f=likelihood, g=prior, show_progress=False)

#             solver.fit(x0=s_hat, stop_crit=sc)

#             s_hat = solver.solution().squeeze()
#             s_hat = (1+lambda2)**(1/2) * s_hat 
#             recon[:,t] = (cst.T @ s_hat) * Ynorms[t]

#     if super:

#         # depth prior
#         W = np.diag(np.linalg.norm(L, axis=0)**dw)

#         # define artificial data set
#         lambda1 = 1-alpha
#         lambda2 = alpha
#         L = (1 + lambda2)**(-1/2) * np.vstack([L, lambda2**(1/2)*W])
#         Y = np.vstack([Y, np.zeros((L.shape[1], Y.shape[1]))])

#         # regularization parameter
#         gamma = lambda1 / (1+lambda2)**(1/2)

#         # component-wise transformation
#         L = L @ cst.T

#         # pyxu components
#         L     = pxa.LinOp.from_array(L)
#         prior = gamma*pxo.L1Norm(dim_shape=cst.shape[0])
#         recon = np.zeros((cst.shape[1], Y.shape[1]))
#         sc    = pxst.MaxIter(1000)

#         # ---- optimization ----
#         for t in range(Y.shape[1]):
#             y     = Y[:,t]
#             s_hat = np.random.random(L.dim_shape[0])*10**(-9)

#             likelihood = pxo.SquaredL2Norm(dim_shape=L.codim_shape).argshift(-y)*L
#             likelihood.diff_lipschitz = likelihood.estimate_diff_lipschitz()

#             solver = PGD(f=likelihood, g=prior, show_progress=False)

#             solver.fit(x0=s_hat, stop_crit=sc)

#             s_hat      = solver.solution().squeeze()
#             s_hat      = (1+lambda2)**(1/2) * s_hat      # enet correction
#             recon[:,t] = (cst.T @ s_hat) * Ynorms[t]

#     # Store in MNE object
#     t_peak = pt/sfreq
#     t_start  = t_peak-(win_size/2)
#     vertices = [fwd["src"][0]["vertno"], fwd["src"][1]["vertno"]]
#     stc      = mne.SourceEstimate(recon, vertices, tmin=t_start, tstep=1/sfreq, subject=sub)

#     return stc



def wcset(
        eeg:mne.Evoked,
        fwd:mne.Forward,
        cst:np.ndarray,
        weights:np.ndarray = None,
        lam:float          = 0.1
) -> mne.SourceEstimate:
    """
    Solve the inverse problem using weighted Connectome Spectrum Electromagnetic Tomography (wCSET)

    eeg: mne.Evoke (M,T)
        EEG

    fwd: mne.Forward (M,N)
        Forward solution.

    cst: np.ndarray (N,N)
        Connectome Spectrum Transform.

    weights: np.ndarray (N,)
        Harmonic (cst components) weights.

    Returns
    -------
    stc: mne.SourceEstimate 
        Source Time Currents.
    
    """
    # extract data
    Y = eeg.data

    # extract lead field matrix with fixed source orientation
    fwd = mne.convert_forward_solution(fwd, force_fixed=True, verbose="warning")
    L   = fwd["sol"]["data"].astype("float64")      # Lead field matrix

    # standardize   !!! I'm a bit scepticle about these standardizations !!!
    Ynorms = np.linalg.norm(Y, axis=0)
    Y      = Y/Ynorms

    #L      = L - np.mean(L, axis=0)
    #Lnorms = np.linalg.norm(L, axis=0)
    #L      = L/Lnorms[None,:]

    # component-wise transformation
    L = L @ cst.T

    # weight matrix
    D = np.diag(1/weights)**2

    # inverse operator
    T = np.linalg.inv(L.T@L + lam*D)@L.T

    # compute inverse
    recon = T@Y

    # revert to dipole space
    recon = cst.T@recon*Ynorms

    # Store in MNE object
    vertices = [fwd["src"][0]["vertno"], fwd["src"][1]["vertno"]]
    tmin     = eeg.tmin
    tstep    = 1/eeg.info["sfreq"]
    stc      = mne.SourceEstimate(recon, vertices, tmin=tmin, tstep=tstep)

    return stc





# def sloreta(eeg, fwd, noise_cov=None, lam=0.1, times=None):
#     """
#     Source reconstruction using standardized low resolution electromagnetic tomography (sLORETA).

#     Parameters
#     ----------
#     eeg: mne.Evoked
#         EEG recordings.

#     fwd: mne.Forward
#         Forward solution.

#     lam: float
#         L2 regularizer parameter.

#     win_size: float
#         Time window size to reconstruct around peak. If None, and time_points is None, reconstruct whole time series.
    
#     times: list(float)
#         Times to reconstruct.

#     Returns
#     -------
#     recon: mne.SourceEstimate
#         Source reconstruction.
#     """
    
#     if noise_cov is None:
#         noise_cov = mne.make_ad_hoc_cov(eeg.info, std={"eeg":1})    # identity matrix
    
#     # prepare eeg
#     eeg.set_eeg_reference(projection=True)

#     # solve inverse problem
#     inverse = mne.minimum_norm.make_inverse_operator(
#         eeg.info,
#         fwd,
#         noise_cov,
#         fixed=True,
#         depth=0,
#         verbose="warning",
#     )

#     stc = mne.minimum_norm.apply_inverse(
#         eeg, 
#         inverse, 
#         lambda2=lam,
#         method="sLORETA",
#         verbose="warning",
#     )

#     # crop time window
#     if times is not None: 
#         idx      = stc.time_as_index(times)
#         data     = stc.data[:,idx]
#         vertices = [fwd["src"][0]["vertno"], fwd["src"][1]["vertno"]]

#         tstep = (times[1]-times[0]) if len(times)>1 else 0

#         return mne.SourceEstimate(data, vertices, tmin=times[0], tstep=tstep, subject=stc.subject)

#     else:
#         return stc