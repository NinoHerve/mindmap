"""
Module Name: analysis/functional/inverse/solvers.py
Description: Class to run inverse problems.

This module provides the InverseOperator class to run inverse problems.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

from abc import ABC, abstractmethod

import numpy as np
import scipy.linalg as sla

import mne
from mne.proj import make_projector
from mne.cov import prepare_noise_cov, compute_whitener

import pyxu.abc as pxa
import pyxu.operator as pxo
import pyxu.opt.stop as pxst
from pyxu.opt.solver import PGD

from .simulation import make_stc
from .utils import compute_harmonic_prior, compute_depth_prior





class MNE:
    """Base inverse solver."""
    def __init__(
        self,
        forward: mne.Forward,
        noise_cov: mne.Covariance,
        depth: float = 0.8,
        noise_norm: str = None,
        verbose: str = "warning",
    ):
        """
        Base class for M/EEG inverse operators.

        Parameters
        ----------
        forward : mne.Forward
            The MNE-Python forward solution.

        noise_cov : mne.Cov
            The noise covariance matrix.

        depth : float
            Depth weighting coefficient.

        fixed : bool
            If True, use fixed orientations (normal to cortex)
        """
        # input data
        self.forward = mne.convert_forward_solution(forward, force_fixed=True, verbose=verbose)
        self.noise_cov = noise_cov
        self.gain_raw = self.forward["sol"]["data"]

        # parameters
        self.depth_coeff = depth
        self.noise_norm = noise_norm

        # variable properties
        self.rank = self.gain_raw.shape[0]
        self.lam = None

        # others
        self.verbose = verbose

    # ------------------
    # Component modules
    # ------------------

    def _setup_preprocessor(self, info):
        """
        Extracts whitener from noise covariance.

        Parameters
        ----------
        noise_cov: mne.Covariance
            Noise covariance matrix.

        nave: int
            Number of samples used for averaging.

        include_active: bool
            Whether to include active projections. Forward model
            needs all projections, Evoked only needs none active
            projections.

        Returns
        -------
        np.ndarray
            Whitener.
        """
        # projectors
        projs_gain, _, _ = make_projector(info["projs"], info.ch_names)
        projs_data, _, _ = make_projector(info["projs"], info.ch_names, include_active=False)

        # whitener
        ready_cov = prepare_noise_cov(self.noise_cov, info, verbose=self.verbose)
        whitener, _ = compute_whitener(self.noise_cov, info, verbose=self.verbose)

        # store rank-related info
        self.rank = np.sum(ready_cov["eig"] > 0)

        return projs_gain, projs_data, whitener


    def compute_source_prior(self, gain_whitened):
        """
        Default: standard depth weighting.
        """
        return compute_depth_prior(gain_whitened, self.depth_coeff)


    def compute_noise_norm(self, lambda2, kernel, method="dSPM"):
        """
        Computes the noise normalization vector (noise sensitivity).

        Note:
        - Since we only need the diagonal of the source covariance
          we can use the property that the diagonal of a product AB is the
          row-wise sum A*B^T.
        - Since we are dealing with M@V@D@V^T@M^T, we can just transform the
          basis vectors and take their norms: SUM(M@V@sqrt(D))²
        """
        # estimate sqrt(D)
        if method == "dSPM":
            # Weighting for dSPM: d_kk^2 = s / (s^2 + lambda2)
            d_vals = kernel["s"] / (kernel["s"]**2 + lambda2)
        elif method == "sLORETA":
            # Weighting for sLORETA: omega_kk = sqrt(s^2 / (s^2 + lambda2))
            d_vals = np.sqrt(kernel["s"]**2 / (kernel["s"]**2 + lambda2))
        else:
            raise ValueError(f"Noise normalization method {method} is unknown")

        v_weighted = d_vals[:, np.newaxis] * kernel["vh"]

        # Project through mixing matrix (M)
        if kernel["w_source"].ndim == 1:
            norm_t = v_weighted * kernel["w_source"]
        else:
            norm_t = v_weighted @ kernel["w_source"]

        # Comptue diagonal of (norm @ norm.T) efficiently
        norm_sq = np.sum(norm_t**2, axis=0)

        return np.sqrt(norm_sq)

    # ----------------
    # Inverse modules
    # ----------------


    def prepare_kernel(self, projs, whitener):
        """Builds kernel for source reconstruction"""
        # 1. Projectors and whitener
        gain_white = whitener @ projs @ self.gain_raw

        # 2. Get the prior
        # This returns either a 1D vector (diagonal) or 2D matrix
        w_source = self.compute_source_prior(gain_white)

        # 3. Apply prior and SVD
        if w_source.ndim == 1:
            gain_weighted = gain_white * w_source
        else:
            gain_weighted = gain_white @ w_source

        # 4. Scale source variance (to match EEG noise variance range)
        trace_GRGT = np.linalg.norm(gain_weighted, ord="fro")**2
        scale = np.sqrt(self.rank/trace_GRGT)

        gain_scaled = gain_weighted * scale
        w_source_scaled = w_source * scale

        # SVD decomposition
        u, s, vh = sla.svd(gain_scaled, full_matrices=False)

        return {
            "u": u, "s":s, "vh": vh,
            "w_source": w_source_scaled,
        }


    def solve(self, evoked, lambda2=None):
        """Solve inverse problem."""
        # Preprocessors
        projs_gain, projs_data, whitener = self._setup_preprocessor(evoked.info)

        # 1. Process gain
        kern = self.prepare_kernel(projs_gain, whitener)

        # 2. Process data
        y_white = whitener @ projs_data @ evoked.data

        # 3. Inverse logic
        if lambda2 is None:
            self.lam = self.estimate_lambda2(y_white, evoked.nave)
        else:
            self.lam = lambda2

        res_scaling = kern["s"] / (kern["s"]**2 + self.lam)
        sol = kern["vh"].T @ (res_scaling[:, np.newaxis] * (kern["u"].T @ y_white))

        # 4. Map back with prior
        if kern["w_source"].ndim == 1:
            sol = sol * kern["w_source"][:, np.newaxis]
        else:
            sol = kern["w_source"] @ sol

        # 5. Normalize
        if self.noise_norm is not None:
            norm = self.compute_noise_norm(
                self.lam,
                kernel = kern,
                method = self.noise_norm,
            )

            sol /= norm[:, np.newaxis]

        return make_stc(
            source_space = self.forward["src"],
            activity = sol,
            tmin = evoked.times[0],
            tstep = 1/evoked.info["sfreq"]
        )

    # ----------------
    # Utility modules
    # ----------------


    def estimate_lambda2(self, y_white, nave=1):
        """
        Estimates the regularization parameter lambda^2

        Parameters
        ----------
        ave : array (n_channels, n_times)
            The whitened sensor data.

        nave
        """
        # Total power (signal + noise)
        ms_amplitude = np.mean(np.square(y_white), axis=0) # Per time point

        # Subtract noise power (1 since data is whitened)
        signal_power = np.maximum(np.mean(ms_amplitude) - 1.0, 1e-8)

        # Ensure we don't divide by zero or negative
        return 1.0 / (signal_power*nave)


class wCSET(MNE):
    """weighted Connectome Electromagnetic Tomography"""
    def __init__(self, eigenvectors, eigenvalues, harm=0.7, **kwargs):
        """
        Parameters
        ----------

        """
        self.U = eigenvectors
        self.e = eigenvalues
        self.harm = harm

        super().__init__(**kwargs)


    def compute_source_prior(self, gain_whitened):
        """
        Override MNE source weighting.
        Compute Connectome Spectrum prior with depth and harmonic weightings.
        """
        # depth weighting
        w_depth = compute_depth_prior(gain_whitened, self.depth_coeff)

        # harmonic weighting
        w_harm = compute_harmonic_prior(self.e, relaxation=self.harm)

        # inverse prior mixing matrix
        return w_depth[:, np.newaxis] * (self.U * w_harm)


# class InverseOperator(ABC):
#     """
#     Class for inverse operators
#     """
#     # m defines the number of sensors while n defines the number of sources
#     info: mne._fiff.meas_info.Info      # Sensor info object

#     # Core data
#     lead  : mne.Forward                      # Leadfield matrix (a.k.a forward solution) (m,n)
#     ncov  : mne.cov.Covariance              # Noise Covariance matrix (sensor noise) (m,m)
#     prior : np.ndarray                      # Prior Mixing matrix (n,n)
#     basis : np.ndarray                      # Basis transformation matrix (a.k.a component-wise extraction matrix) (n,n)

#     # whitening
#     white : np.ndarray                       # Whitener (m,m)
#     color : np.ndarray                       # Colorer (m,m)

#     # depth prior
#     depth : np.ndarray                      # Depth Prior matrix

#     # verbosity (global attribute)
#     verbose = "warning"

#     def __init__(
#         self,
#         info      : mne.Info,
#         leadfield : mne.Forward,
#         noise_cov : mne.Covariance = None,
#         depth     : float = 0.8,
#     ):
#         # EEG info
#         self.info = info

#         # lead field
#         self.lead  = mne.convert_forward_solution(leadfield, force_fixed=True, verbose="warning")  # we always work with fixed orientation

#         if noise_cov is None:
#             self.ncov  = mne.make_ad_hoc_cov(self.info, std={"eeg":1})
#             self.white = np.eye(self.lead["nchan"])
#             self.color = np.eye(self.lead["nchan"])
#         else:
#             self.ncov = noise_cov
#             self.white, self.color = self.compute_whitener()

#         # average projector
#         self.avg_proj = self.make_average_projector()

#         # depth prior
#         self.depth = self.compute_depth_prior(exponent = depth)


#     @abstractmethod
#     def __call__(self):
#         pass

#     def compute_whitener(self):
#         """
#         Computes whitener and colorer.

#         New attributes
#         ----------
#         W: np.ndarray (m,m)
#             Whitener. Let UDU.T be the eigdendecomposition of Nc. Then W = D^(-1/2)U.T

#         C: np.ndarray (m,m)
#             Colorer. C = W^(-1)

#         whiten_method: str
#             Method used for whitening.
#         """

#         white, _, color = mne.cov.compute_whitener(
#             noise_cov      = self.ncov,
#             info           = self.info,
#             picks          = self.select_channels(),
#             pca            = "white",
#             return_colorer = True,
#             verbose        = self.verbose,
#         )

#         return white, color



#     def scaling(self, nave):
#         pass

#     def compute_depth_prior(self, exponent):
#         """
#         Computes the depth priors. The depth prior is a diagonal matrix whose elements serve to 
#         weight sources according to their depth, as measured by the columns of the leadfield 
#         matrix. The purpose is to counteract the minimum estimate bias toward superficial sources.

#         Parameters
#         ----------
#         exponent: float [0-1]
#             Exponent for depth weighting.

#         limit: float
#             I don't know what limit does but it changes a lot.

#         New attributes
#         ----------
#         Dp: np.ndarray (n,n)
#             Depth prior matrix.

#         exp: float [0-1]
#             Exponent used for depth weighting.
#         """
#         lead  = self.lead["sol"]["data"].astype("float64")
#         norms = np.linalg.norm(lead, axis=0)

#         depth = np.diag(1/norms**exponent)

#         return depth

#     def make_average_projector(self):
#         """
#         Make average projector for average referencing.

#         Returns
#         -------
#         avg_proj: np.ndarray (n,n)
#             Average projector.
#         """
#         n = self.lead["nchan"]

#         # average projector = I - (1/n)11.T
#         avg_proj = np.eye(n) - (1/n)*np.ones(shape=(n,n))

#         return avg_proj
        
#     def select_channels(self):
#         """
#         Select channels available in both info and ncov.
#         """
#         info_names = set(self.info["ch_names"])
#         info_bads  = set(self.info["bads"])

#         ncov_names = set(self.ncov["names"])
#         ncov_bads  = set(self.ncov["bads"])

#         ch_names = info_names & ncov_names
#         bads     = info_bads | ncov_bads

#         ch_names -= bads

#         return list(ch_names)


# class CSET(InverseOperator):
#     """
#     Implements Connectome Spectrum Electromagnetic Tomography.
#     """
#     info      : mne.Info        # EEG metadata
#     leadfield : mne.Forward     # forward operator
#     eig_vec   : np.ndarray      # connectome spectrum transform
#     noise_cov : mne.Covariance  # Noise covariance
#     depth     : float           # exponent for depth weighting

#     def __init__(
#         self,
#         info      : mne.Info,
#         leadfield : mne.Forward,
#         eig_vec   : np.ndarray,          # required
#         noise_cov : np.ndarray = None,
#         depth     : float = 0,
#     ):
#         super().__init__(
#             info      = info,
#             leadfield = leadfield,
#             noise_cov = noise_cov,
#             depth     = depth,
#         )

#         self.basis = eig_vec

#     def __call__(self, data, lam):
#         """
#         Applies the Connectome Spectrum Electromagnetic Tomography (CSET) source reconstruction
#         method. 

#         Parameter
#         ---------
#         data: mne.Evoked
#             Sensor data to project to source space.

#         lam: float
#             Regularization parameter.

#         Return
#         ------
#         stc: mne.SourceEstimate 
#             Source Time Currents.
#         """
#         n = self.lead["nsource"]   # number of sources
#         t = len(data.times)     # number of time points

#         # -------------------------------------------------------------
#         # Prepare leadfield
#         # ¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
#         # L = WPLD⁻¹U
#         #
#         # where W is a whitener, P is the average projection matrix,
#         # D is the depth weighting prior and U are the connectome's
#         # laplacian eigenvectors
#         # -------------------------------------------------------------

#         # average projection
#         dat  = self.avg_proj @ data.data
#         lead = self.avg_proj @ self.lead["sol"]["data"].astype("float64")

#         # whiten (L = WL)
#         dat  = self.white @ dat
#         lead = self.white @ lead

#         # depth prior
#         inv_depth = np.linalg.inv(self.depth)
#         lead      = lead @ inv_depth

#         # connectome spectrum transform
#         lead = lead @ self.basis

#         # -----------------------
#         # Solve CSET using pyxu
#         # ¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
#         # -----------------------

#         # Instanciate pyxu objects
#         lead  = pxa.LinOp.from_array(lead)          # forward matrix
#         prior = lam * pxo.L1Norm(dim_shape=n)       # l1-regularizer ||ŝ||_1
#         sol   = np.zeros((n, t))                    # solution holder
#         sc    = pxst.MaxIter(1000)                  # stopping criterion

#         # iterate over time points
#         for tp in range(t):
#             # data
#             y     = dat[:,tp]
#             s_hat = np.random.random(n)*10**(-9)    # initial guess

#             # likelihood (||Lŝ - y||_2)²
#             likelihood = pxo.SquaredL2Norm(dim_shape=lead.codim_shape).argshift(-y)*lead
#             likelihood.diff_lipschitz = likelihood.estimate_diff_lipschitz()

#             # solver = Proximal Gradient Descent
#             solver = PGD(f=likelihood, g=prior, show_progress=False)

#             solver.fit(x0=s_hat, stop_crit=sc)

#             # store timepoint's solution
#             sol[:,tp] = solver.solution().squeeze()

#         # revert back to source space
#         sol = inv_depth @ self.basis @ sol

#         # store result in MNE object
#         vertices = [self.lead["src"][0]["vertno"], self.lead["src"][1]["vertno"]]
#         stc      = mne.SourceEstimate(sol, vertices, tmin=data.times[0], tstep=1/data.info["sfreq"])

#         return stc

# class wCSET(InverseOperator):
#     """
#     Applies the weighted Connectome Spectrum Electromagnetic Tomography (CSET) source 
#     reconstruction method. 

#     Attributes
#     ---------
#     data: mne.Evoked
#         Sensor data to project to source space.

#     Return
#     ------
#     stc: mne.SourceEstimate 
#         Source Time Currents.
#     """
#     def __init__(
#         self,
#         info,
#         leadfield : mne.Forward,
#         eig_vec   : np.ndarray,
#         eig_val   : np.ndarray,
#         noise_cov : mne.Covariance = None,
#         depth     : float = 0,
#         harm      : float = 0.3,
#     ):
#         """
#         Init

#         Parameters
#         ----------
#         info: mne.Info
#             Info variable of MNE.

#         leadfield: mne.Forward
#             Forward operator, a.k.a leadfield matrix.

#         eig_vec: np.ndarray
#             Connectome laplacian eigenvectors.

#         eig_val: np.ndarray
#             Connectome laplacian eigenvalues.

#         noise_cov: mne.Covariance
#             Noise covariance matrix. Default is identity matrix.

#         lambda_: float
#             Regularization parameter.

#         depth: float [0-1]
#             Depth exponent for depth weighting.

#         harm: float [0-1]
#             Harmonic exponent for harmonic weighting.
#         """
#         super().__init__(
#             info      = info,
#             leadfield = leadfield,
#             noise_cov = noise_cov,
#             depth     = depth,
#         )

#         # Harmonic prior
#         self.basis = eig_vec
#         self.harm  = compute_harmonic_prior(eig_val, relaxation=harm)


#     def __call__(
#             self,
#             data : mne.Evoked,
#             lam  : float = 0.01,
#         ) -> mne.SourceEstimate:
#         """
#         Applies the weighted Connectome Spectrum Electromagnetic Tomography (wCSET) 
#         source reconstruction method. 

#         Parameters
#         ----------
#         data: mne.Evoked 
#             Sensor data. 

#         lam_harm: float
#             Harmonic weighting regularization parameter.

#         lam_depth: float
#             Depth weighting regularization parameter.

#         Return
#         ------
#         stc: mne.SourceEstimate 
#             Source Time Currents.
#         """
#         # apply average referencing
#         lead = self.avg_proj @ self.lead["sol"]["data"].astype("float64")
#         y    = self.avg_proj @ data.data

#         # build kernel
#         inv_op = self.build_kernel(lead, lam)

#         # compute solution
#         # - Solution is computed in the graph spectrum space
#         # - Basis transform is used to revert back to dipole space
#         #sol = self.basis @ inv_op @ data.data
#         sol = inv_op @ y

#         # store result in MNE object
#         stc = make_stc(self.lead["src"], sol, data.tmin, 1/data.info["sfreq"])

#         return stc


#     def build_kernel(self, lead, lam):
#         """
#         Build kernel
#         """
#         # prior: D_h@U.T@D_d -> (D_h@U.T@D_d).T @ (D_h@U.T@D_d) = D_d@U@D_h**2@U.T@ D_d
#         prior = self.depth @ self.basis @ self.harm**2 @ self.basis.T @ self.depth

#         # inverse prior
#         inv_prior = np.linalg.inv(prior)     # sensor inversion might make no sense due to this inverse

#         # kernel with sensor inversion (R⁻¹ L.T(L R⁻¹ L.T + lam*W)⁻¹)
#         inv_op = inv_prior @ lead.T @ np.linalg.inv(lead @ inv_prior @ lead.T + lam*self.white)

#         return inv_op



#     def make_resolution_matrix(self, lam):
#         """
#         Make resolution matrix.

#         Parameters
#         ----------
#         info: mne.Info
#             Evoked info object. Needed for eventual projectors.

#         Returns
#         -------
#         resmat: np.ndarray (n,n)
#             Resolution matrix.
#         """
#         # apply average referencing
#         lead = self.avg_proj @ self.lead["sol"]["data"].astype("float64")

#         # inverse operator
#         inv_op = self.build_kernel(lead, lam)

#         # resolution matrix
#         resmat = inv_op @ lead

#         return resmat

    
#     def make_harmonic_resolution_matrix(self, lam):
#         """
#         Make harmonic resolution matrix.

#         Parameters
#         ----------
#         info: mne.Info
#             Evoked info object. Needed for eventual projectors.

#         Returns
#         -------
#         harmmat: np.ndarray (n,n)
#             Resolution matrix
#         """
#         # apply average referencing
#         lead = self.avg_proj @ self.lead["sol"]["data"].astype("float64")

#         # inverse operator
#         inv_op = self.build_kernel(lead, lam)

#         # resolution matrix
#         resmat = self.basis.T @ inv_op @ lead @ self.basis

#         return resmat




# def wcset(
#         eeg:mne.Evoked,
#         fwd:mne.Forward,
#         cst:np.ndarray,
#         weights:np.ndarray = None,
#         lam:float          = 0.1
# ) -> mne.SourceEstimate:
#     """
#     Solve the inverse problem using weighted Connectome Spectrum Electromagnetic Tomography (wCSET)

#     eeg: mne.Evoke (M,T)
#         EEG

#     fwd: mne.Forward (M,N)
#         Forward solution.

#     cst: np.ndarray (N,N)
#         Connectome Spectrum Transform.

#     weights: np.ndarray (N,)
#         Harmonic (cst components) weights.

#     Returns
#     -------
#     stc: mne.SourceEstimate 
#         Source Time Currents.
    
#     """
#     # extract data
#     Y = eeg.data

#     # extract lead field matrix with fixed source orientation
#     fwd = mne.convert_forward_solution(fwd, force_fixed=True, verbose="warning")
#     L   = fwd["sol"]["data"].astype("float64")      # Lead field matrix

#     # standardize   !!! I'm a bit scepticle about these standardizations !!!
#     Ynorms = np.linalg.norm(Y, axis=0)
#     Y      = Y/Ynorms

#     #L      = L - np.mean(L, axis=0)
#     #Lnorms = np.linalg.norm(L, axis=0)
#     #L      = L/Lnorms[None,:]

#     # component-wise transformation
#     L = L @ cst.T

#     # weight matrix
#     D = np.diag(1/weights)**2

#     # inverse operator
#     T = np.linalg.inv(L.T@L + lam*D)@L.T

#     # compute inverse
#     recon = T@Y

#     # revert to dipole space
#     recon = cst.T@recon*Ynorms

#     # Store in MNE object
#     vertices = [fwd["src"][0]["vertno"], fwd["src"][1]["vertno"]]
#     tmin     = eeg.tmin
#     tstep    = 1/eeg.info["sfreq"]
#     stc      = mne.SourceEstimate(recon, vertices, tmin=tmin, tstep=tstep)

#     return stc

